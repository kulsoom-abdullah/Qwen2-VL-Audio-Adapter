"""
05_evaluate.py
==============
Final Model Evaluation Script
-----------------------------
Evaluates the model on the SpeechBrain Test Partition (200 unseen samples).
Loads model directly from Hugging Face.
Outputs detailed metrics and a results list.
"""

import os
import sys
import json
import torch
import librosa
from tqdm import tqdm
from jiwer import wer, cer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, WhisperFeatureExtractor

# --- CONFIGURATION ---
# 1. Use your Hugging Face Model ID
MODEL_PATH = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"

# 2. Use the Audit Test Data we just downloaded
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "audit_test")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "output", "final_evaluation_results.json")

# Constants
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500

def evaluate_model():
    print("="*80)
    print("🧪 FINAL MODEL EVALUATION (SpeechBrain Test Set)")
    print("="*80)

    # 1. Load Model (From Hugging Face)
    print(f"\n📥 Loading model: {MODEL_PATH}")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Hint: Run 'huggingface-cli login' if your model is private.")
        return

    # 2. Load Data
    print(f"\n📊 Loading test data from: {TEST_JSON}")
    if not os.path.exists(TEST_JSON):
        print("❌ Test data not found. Run 'python scripts/setup_audit_data.py' first.")
        return

    with open(TEST_JSON, 'r') as f:
        data = json.load(f)
    print(f"✅ Found {len(data)} samples.")

    # 3. Inference Loop
    results = []
    print("\n🚀 Running Evaluation...")
    
    for entry in tqdm(data):
        audio_abs_path = os.path.join(DATA_DIR, entry['audio'])
        ground_truth = entry['ground_truth']

        try:
            # Load Audio
            y, sr = librosa.load(audio_abs_path, sr=16000, mono=True)
            inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(model.device).to(torch.bfloat16)

            # Build Prompt
            audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
            input_ids_audio = torch.tensor([audio_tokens], device=model.device)
            
            p1 = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
            p2 = tokenizer.encode("<|audio_eos|>\nTranscribe this audio.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt").to(model.device)
            input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    input_features=input_features,
                    max_new_tokens=128,
                    do_sample=False
                )

            prediction = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Calculate Metrics
            sample_wer = wer(ground_truth.lower(), prediction.lower())
            
            results.append({
                "id": entry['id'],
                "audio": entry['audio'],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "wer": sample_wer
            })

        except Exception as e:
            print(f"⚠️ Error on {entry['id']}: {e}")

    # 4. Save Results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    # 5. Final Stats
    avg_wer = sum(r['wer'] for r in results) / len(results)
    
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Samples: {len(results)}")
    print(f"Average WER:   {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"Results List:  {OUTPUT_FILE}")
    print("="*80)

if __name__ == "__main__":
    evaluate_model()