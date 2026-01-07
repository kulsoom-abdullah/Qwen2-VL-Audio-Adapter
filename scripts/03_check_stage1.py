"""
03_check_stage1.py
==================
Sanity check for Stage 1 (Projector Alignment).
Loads the frozen Qwen model + your trained projector and runs inference.
"""

import os
import sys
import json
import random
import torch
import librosa
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, WhisperFeatureExtractor

# --- SMART PATH SETUP ---
# Get the absolute path of THIS script (scripts/02_check_stage1.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the Project Root (one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add Fork to Path (Robust)
FORK_PATH = os.path.join(PROJECT_ROOT, "transformers_fork", "src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
else:
    # Fallback for RunPod if repo structure isn't perfect yet
    sys.path.insert(0, os.path.abspath("./transformers_fork/src"))

# --- CONFIGURATION ---
# Now we point to data relative to the PROJECT ROOT, not the current working directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "full_data")
BASE_MODEL = os.path.join(PROJECT_ROOT, "model_base_grafted")
# Model output usually stays in root/output
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "stage1_projector", "final_projector")

def check_inference():
    print("="*60)
    print("🎧 Qwen2-Audio Stage 1: Inference Check")
    print(f"📂 Execution Context: {PROJECT_ROOT}")
    print("="*60)

    # 1. Validation: Ensure Data Exists
    json_path = os.path.join(DATA_DIR, "eval.json")
    if not os.path.exists(json_path):
        print(f"❌ Error: Data not found at {json_path}")
        print("   (Did you make sure to NOT run this on your Mac? Data is on RunPod!)")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Pick random samples to test
    samples = random.sample(data, 3)

    # 2. Validation: Ensure Base Model Exists
    if not os.path.exists(BASE_MODEL):
        print(f"❌ Error: Base model not found at {BASE_MODEL}")
        print("   Run '00_graft_architecture.py' first.")
        return

    # 3. Load Model & Inject Projector
    print("Loading Base Model...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        return
    
    print("Loading Stage 1 Projector Weights...")
    projector_path = os.path.join(MODEL_PATH, "audio_projector.pt")
    
    if os.path.exists(projector_path):
        # Load the trained linear layer weights
        model.audio_projector.load_state_dict(torch.load(projector_path))
        print("✅ Projector weights loaded successfully.")
    else:
        print(f"⚠️  WARNING: Projector weights not found at {projector_path}")
        print("   Using random initialization (Expect gibberish output).")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # 4. Run Inference
    AUDIO_TOKEN_ID = 151657
    
    for s in samples:
        print("\n" + "-"*40)
        print(f"Sample: {s['id']}")
        
        # Load Audio (Handle relative path in JSON)
        # JSON says "audio/train_0001.wav", we join with DATA_DIR
        audio_path = os.path.join(DATA_DIR, s['audio'])
        
        try:
            y, _ = librosa.load(audio_path, sr=16000, mono=True)
        except Exception as e:
            print(f"   ❌ Could not load audio: {e}")
            continue
        
        # Prepare Inputs
        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device).to(torch.bfloat16)
        
        audio_tokens = torch.full((1, 1500), AUDIO_TOKEN_ID, device=model.device, dtype=torch.long)
        
        # Generate
        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=audio_tokens,
                input_features=input_features,
                max_new_tokens=64,
                do_sample=False 
            )
        
        # Decode
        pred = tokenizer.decode(gen_ids[0, 1500:], skip_special_tokens=True)
        
        print(f"Ground Truth: {s['ground_truth']}")
        print(f"Prediction:   {pred.strip()}")

if __name__ == "__main__":
    check_inference()