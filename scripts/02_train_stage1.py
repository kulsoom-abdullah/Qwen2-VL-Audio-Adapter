"""
02_train_stage1.py
==================
Stage 1: Audio Projector Alignment
----------------------------------
Trains ONLY the linear projector to align Whisper features with Qwen's embedding space.
Uses the unified Stage 2 dataset (JSON + WAVs) for training.
"""

import sys
import os
import json
import torch
import librosa
import numpy as np
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    WhisperFeatureExtractor,
    Trainer,
    TrainingArguments
)

# --- SMART PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add Fork
FORK_PATH = os.path.join(PROJECT_ROOT, "transformers_fork", "src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
else:
    sys.path.insert(0, os.path.abspath("./transformers_fork/src"))

# Config
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "stage2_full")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage1_projector")
MODEL_SOURCE = os.path.join(PROJECT_ROOT, "model_base_grafted")

# Constants
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500

def load_json_data():
    """Loads JSON data from the prepare_data script."""
    train_path = os.path.join(DATA_DIR, "train.json")
    eval_path = os.path.join(DATA_DIR, "eval.json")
    
    if not os.path.exists(train_path):
        print(f"❌ Error: Data not found at {train_path}")
        print("   Run '01_prepare_data.py' first!")
        sys.exit(1)

    with open(train_path, 'r') as f:
        train_ds = Dataset.from_list(json.load(f))
    with open(eval_path, 'r') as f:
        eval_ds = Dataset.from_list(json.load(f))
        
    return train_ds, eval_ds

@dataclass
class Stage1Collator:
    tokenizer: AutoTokenizer
    feature_extractor: WhisperFeatureExtractor
    base_dir: str

    def __call__(self, features):
        audios = []
        texts = []
        
        for f in features:
            # 1. Load Audio from Path
            # JSON contains "audio/filename.wav", we join with base_dir
            audio_path = os.path.join(self.base_dir, f["audio"])
            try:
                wav, _ = librosa.load(audio_path, sr=16000, mono=True)
            except Exception:
                wav = np.zeros(16000)
            audios.append(wav)
            
            # 2. Get Text (Ground Truth)
            # We train the projector to simply predict the text caption
            text = f["ground_truth"] + self.tokenizer.eos_token
            texts.append(text)

        # Process Audio
        audio_features = self.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")["input_features"]
        
        # Process Text
        text_inputs = self.tokenizer(texts, padding="longest", truncation=True, max_length=512, return_tensors="pt", add_special_tokens=False)

        # Combine (Audio -> Text)
        batch_size = len(features)
        audio_ids = torch.full((batch_size, NUM_AUDIO_TOKENS), AUDIO_TOKEN_ID, dtype=torch.long)
        
        input_ids = torch.cat([audio_ids, text_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([torch.ones_like(audio_ids), text_inputs["attention_mask"]], dim=1)
        
        # Labels: Mask Audio (-100), Train on Text
        labels = torch.cat([torch.full_like(audio_ids, -100), text_inputs["input_ids"]], dim=1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": audio_features,
            "labels": labels
        }

def train_stage1():
    print(f"🚀 Starting Stage 1: Projector Alignment")
    
    # 1. Load Data
    train_ds, eval_ds = load_json_data()
    print(f"✅ Loaded Data: {len(train_ds)} train, {len(eval_ds)} eval")

    # 2. Load Base Model
    if not os.path.exists(MODEL_SOURCE):
        print(f"❌ Base model not found at {MODEL_SOURCE}")
        print("   Run '00_graft_architecture.py' first!")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_SOURCE,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    # 3. Freeze & Unfreeze
    model.requires_grad_(False)
    model.audio_projector.requires_grad_(True)
    print(f"🔒 Model Frozen. Training ONLY Audio Projector.")

    # 4. Train
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=1e-3,
        max_steps=5000,
        logging_steps=50,
        save_strategy="steps", save_steps=1000,
        bf16=True,
        report_to="wandb",
        run_name="Stage1-Projector-Unified"
    )

    collator = Stage1Collator(tokenizer, feature_extractor, base_dir=DATA_DIR)

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, 
        eval_dataset=eval_ds, data_collator=collator
    )
    
    trainer.train()
    
    # Save Projector
    final_path = os.path.join(OUTPUT_DIR, "final_projector")
    os.makedirs(final_path, exist_ok=True)
    torch.save(model.audio_projector.state_dict(), os.path.join(final_path, "audio_projector.pt"))
    print(f"✅ Saved aligned projector to {final_path}")

if __name__ == "__main__":
    train_stage1()