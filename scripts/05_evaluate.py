"""
05_evaluate.py
==============
Stage 2 Final Model Evaluation
------------------------------
Comprehensive inference test on the fine-tuned model.
Calculates WER/CER on the held-out test set.
"""

import os
import sys
import json
import torch
import librosa
from tqdm import tqdm
from jiwer import wer, cer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, WhisperFeatureExtractor

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
EVAL_JSON = os.path.join(DATA_DIR, "eval.json")
# This points to the final merged model from 04_train_stage2.py
MODEL_PATH = os.path.join(PROJECT_ROOT, "output", "stage2_merged")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "output", "evaluation_results.json")

# Constants
NUM_SAMPLES = 50  # Test on first 50 eval samples
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500

def evaluate():
    print("="*80)
    print("🧪 STAGE 2 FINAL MODEL EVALUATION")
    print("="*80)

    # 1. Validation
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        print("   Make sure '04_train_stage2.py' completed successfully!")
        return

    if not os.path.exists(EVAL_JSON):
        print(f"❌ Eval data not found at: {EVAL_JSON}")
        return

    # 2. Load Model
    print(f"\n📥 Loading model from: {MODEL_PATH}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Ensure special tokens are set correctly for inference
    tokenizer.pad_token_id = 151643
    tokenizer.eos_token_id = 151645

    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # 3. Load Data
    print(f"\n📊 Loading eval data...")
    with open(EVAL_JSON, 'r') as f:
        eval_data = json.load(f)

    test_samples = eval_data[:NUM_SAMPLES]
    print(f"✅ Testing on {len(test_samples)} samples")

    # 4. Inference Loop
    results = []
    
    print("\n🚀 RUNNING INFERENCE")
    for idx, entry in enumerate(tqdm(test_samples, desc="Evaluating")):
        # Handle relative audio path
        audio_path = os.path.join(DATA_DIR, entry['audio'])
        ground_truth = entry['ground_truth']

        try:
            # Prepare Audio
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(model.device).to(torch.bfloat16)

            # Build Prompt
            audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
            input_ids_audio = torch.tensor([audio_tokens], device=model.device)

            p1 = tokenizer.encode(
                "<|im_start|>user\n<|audio_bos|>",
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            p2 = tokenizer.encode(
                "<|audio_eos|>\nTranscribe this audio.<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
                return_tensors="pt"
            ).to(model.device)

            input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)
            attention_mask = torch.ones_like(input_ids)

            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    input_features=input_features,
                    attention_mask=attention_mask,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            output_text = tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Simple Match Logic
            match_type = "mismatch"
            if output_text.lower() == ground_truth.lower():
                match_type = "exact"
            elif ground_truth.lower() in output_text.lower():
                match_type = "partial"

            results.append({
                "id": entry['id'],
                "ground_truth": ground_truth,
                "prediction": output_text,
                "match_type": match_type,
                "audio_path": entry['audio'] # Keep relative for cleaner JSON
            })

        except Exception as e:
            print(f"\n⚠️  Error on sample {idx}: {e}")

    # 5. Metrics & Reporting
    references = [r['ground_truth'] for r in results]
    hypotheses = [r['prediction'] for r in results]
    
    final_wer = wer(references, hypotheses)
    final_cer = cer(references, hypotheses)

    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)
    print(f"WER: {final_wer:.3f}")
    print(f"CER: {final_cer:.3f}")

    # Save
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {RESULTS_FILE}")
    print("   Run 'notebooks/02_error_analysis.ipynb' to visualize mismatches.")
    print("="*80)

if __name__ == "__main__":
    evaluate()
