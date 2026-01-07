"""
04_train_stage2.py
==================
Stage 2: Full Instruction Tuning (QLoRA)
----------------------------------------
Fine-tunes the Projector + LLM on the transcription task.
Loads Stage 1 projector weights if available locally.
"""

import os
import sys
import json
import torch
import librosa
import wandb
from dataclasses import dataclass
from datasets import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    WhisperFeatureExtractor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- SMART PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add Fork
FORK_PATH = os.path.join(PROJECT_ROOT, "transformers_fork", "src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
else:
    # Fallback
    sys.path.insert(0, os.path.abspath("./transformers_fork/src"))

# Config
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "stage2_full")
TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
EVAL_JSON = os.path.join(DATA_DIR, "eval.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "stage2_checkpoints")
FINAL_DIR = os.path.join(PROJECT_ROOT, "output", "stage2_merged")

# Model Source
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "model_base_grafted")
STAGE1_PROJECTOR_PATH = os.path.join(PROJECT_ROOT, "output", "stage1_projector", "final_projector", "audio_projector.pt")
HF_FALLBACK = "kulsoom-abdullah/Qwen2-Audio-Stage1" 

RUN_NAME = "Stage2-Full-Training"
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500

def train_stage2():
    print("="*80)
    print("🚀 STAGE 2 FULL TRAINING")
    print("="*80)

    # 1. Load Data
    print(f"📂 Loading Data from {DATA_DIR}")
    if not os.path.exists(TRAIN_JSON):
        print(f"❌ Data not found. Run 01_prepare_data.py first.")
        return

    with open(TRAIN_JSON, 'r') as f:
        train_dataset = Dataset.from_list(json.load(f))
    with open(EVAL_JSON, 'r') as f:
        eval_dataset = Dataset.from_list(json.load(f))

    # 2. Load Model & Tokenizer
    print("\n📥 Loading Model...")
    
    tokenizer = AutoTokenizer.from_pretrained(HF_FALLBACK, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["audio_projector", "audio_encoder"]
    )

    # Logic: Prefer Local Stage 1 -> Fallback to Hub
    if os.path.exists(BASE_MODEL_PATH) and os.path.exists(STAGE1_PROJECTOR_PATH):
        print(f"✅ Found local Stage 1 weights. Loading from {BASE_MODEL_PATH}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        print(f"🔌 Injecting trained projector from {STAGE1_PROJECTOR_PATH}")
        model.audio_projector.load_state_dict(torch.load(STAGE1_PROJECTOR_PATH))
    else:
        print(f"⚠️  Local Stage 1 not found. Downloading from Hub: {HF_FALLBACK}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            HF_FALLBACK,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Config
    print("\n🔧 Configuring LoRA...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["audio_projector"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Collator
    collator = Stage2Collator(tokenizer, feature_extractor, base_dir=DATA_DIR)

    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="steps", eval_steps=200,
        save_strategy="steps", save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        bf16=True,
        report_to="wandb",
        gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print("\n🚀 Starting Training...")
    trainer.train()

    # 6. Save
    print("\n💾 Saving Final Model...")
    model = model.merge_and_unload()
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"✅ Saved to: {FINAL_DIR}")

# --- HELPER CLASSES ---
@dataclass
class Stage2Collator:
    tokenizer: AutoTokenizer
    feature_extractor: WhisperFeatureExtractor
    base_dir: str

    def __call__(self, examples):
        input_ids_batch = []
        labels_batch = []
        audio_values_batch = []

        for ex in examples:
            # Audio
            full_audio_path = os.path.join(self.base_dir, ex["audio"])
            try:
                y, sr = librosa.load(full_audio_path, sr=16000, mono=True)
            except:
                y = torch.zeros(16000).numpy()
            audio_values_batch.append(y)

            # Text
            user_content = ex["conversations"][0]["content"]
            if "<|audio_eos|>\n" in user_content:
                instruction = user_content.split("<|audio_eos|>\n")[-1]
            else:
                instruction = user_content.split("<|audio_eos|>")[-1].strip()

            assistant_content = ex["conversations"][1]["content"]

            prefix = "<|im_start|>user\n<|audio_bos|>"
            suffix = f"<|audio_eos|>\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            target = f" {assistant_content}<|im_end|>"

            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
            audio_ids = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
            suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)

            input_ids = prefix_ids + audio_ids + suffix_ids + target_ids
            labels = ([-100] * (len(prefix_ids) + len(audio_ids) + len(suffix_ids))) + target_ids

            input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))

        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True, padding_value=-100)
        attention_mask = (input_ids_padded != self.tokenizer.pad_token_id).long()
        audio_features = self.feature_extractor(audio_values_batch, sampling_rate=16000, return_tensors="pt")

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
            "input_features": audio_features["input_features"]
        }

if __name__ == "__main__":
    train_stage2()