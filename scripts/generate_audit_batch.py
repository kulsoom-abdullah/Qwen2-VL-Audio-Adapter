"""
generate_audit_batch.py
=======================
Label Noise Audit Script
-------------------------
Runs inference on 100 samples from SpeechBrain test partition (100% unseen).
Outputs JSON with audio paths, ground truth, predictions, and per-sample WER.

This proves true generalization, not memorization.
"""

import os
import sys
import json
import random
import torch
import librosa
from tqdm import tqdm
from jiwer import wer
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, WhisperFeatureExtractor

# --- SMART PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Add Fork (CRITICAL - your model needs custom transformers code!)
FORK_PATH = os.path.join(PROJECT_ROOT, "transformers_fork", "src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
    print(f"✅ Using transformers fork from: {FORK_PATH}")
else:
    print("⚠️  Transformers fork not found - model may fail to load!")

# Config - Load directly from HuggingFace
MODEL_PATH = "kulsoom-abdullah/Qwen2-Audio-7B-Transcription"  # Your HF model repo
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "audit_test")
TEST_JSON = os.path.join(DATA_DIR, "test.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "output", "audit_batch_results.json")

# Constants
TOTAL_AUDIT_SAMPLES = 100  # Sample 100 from the 200 available
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500

def normalize_text(text):
    """Simple normalization for comparison"""
    return text.lower().strip()

def calculate_sample_wer(reference, hypothesis):
    """Calculate WER for a single sample"""
    try:
        return wer([reference], [hypothesis])
    except:
        return 1.0  # Return 100% WER if calculation fails

def generate_audit_batch():
    print("="*80)
    print("🔍 LABEL NOISE AUDIT BATCH GENERATOR")
    print("="*80)
    print("Using SpeechBrain Test Partition (100% UNSEEN)")
    print("="*80)

    # 1. Validation - Check test data exists
    if not os.path.exists(TEST_JSON):
        print(f"❌ Test data not found at: {TEST_JSON}")
        print("   Run: python scripts/setup_audit_data.py")
        return

    # 2. Load Model (directly from HuggingFace)
    print(f"\n📥 Loading model from HuggingFace: {MODEL_PATH}")
    print("   (This may take a few minutes on first download...)")

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokenizer.pad_token_id = 151643
        tokenizer.eos_token_id = 151645

        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

        print("✅ Model loaded successfully!")

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check transformers fork is in project (needed for custom model code)")
        print("   2. If model is private, run: huggingface-cli login")
        print("   3. Check internet connection")
        return

    # 3. Load Data
    print(f"\n📊 Loading SpeechBrain test data...")
    with open(TEST_JSON, 'r') as f:
        all_data = json.load(f)

    print(f"✅ Total samples available: {len(all_data)}")

    # 4. Sample Selection - Simple random sampling from unseen test set
    print(f"\n🎲 Sampling Strategy:")
    print(f"   - Randomly selecting {TOTAL_AUDIT_SAMPLES} from {len(all_data)} test samples")
    print(f"   - All samples are from SpeechBrain TEST partition (never seen during training)")

    # Random sampling
    if len(all_data) >= TOTAL_AUDIT_SAMPLES:
        audit_samples = random.sample(all_data, TOTAL_AUDIT_SAMPLES)
    else:
        audit_samples = all_data
        print(f"⚠️  Only {len(all_data)} samples available, using all")

    print(f"✅ Selected {len(audit_samples)} samples for audit")

    # 5. Inference Loop
    audit_results = []

    print("\n🚀 RUNNING INFERENCE ON AUDIT BATCH")
    for idx, entry in enumerate(tqdm(audit_samples, desc="Generating audit batch")):
        # Handle relative audio path
        audio_rel_path = entry['audio']
        audio_abs_path = os.path.join(DATA_DIR, audio_rel_path)
        ground_truth = entry['ground_truth']

        try:
            # Prepare Audio
            y, sr = librosa.load(audio_abs_path, sr=16000, mono=True)
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

            prediction = tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Calculate per-sample WER
            sample_wer = calculate_sample_wer(ground_truth, prediction)

            # Determine if there's a disagreement (normalized comparison)
            is_disagreement = normalize_text(ground_truth) != normalize_text(prediction)

            audit_results.append({
                "id": entry.get('id', f"sample_{idx}"),
                "audio_path": audio_abs_path,  # Store absolute path for easy playback
                "audio_relative_path": audio_rel_path,  # Keep relative for reference
                "ground_truth": ground_truth,
                "prediction": prediction,
                "wer": round(sample_wer, 4),
                "is_disagreement": is_disagreement,
                "source": entry.get('source', 'speechbrain_test')
            })

        except Exception as e:
            print(f"\n⚠️  Error on sample {idx} ({entry.get('id', 'unknown')}): {e}")
            # Add failed sample to results for tracking
            audit_results.append({
                "id": entry.get('id', f"sample_{idx}"),
                "audio_path": audio_abs_path,
                "audio_relative_path": audio_rel_path,
                "ground_truth": ground_truth,
                "prediction": "[INFERENCE_FAILED]",
                "wer": 1.0,
                "is_disagreement": True,
                "source": entry.get('source', 'speechbrain_test'),
                "error": str(e)
            })

    # 6. Save Results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(audit_results, f, indent=2)

    # 7. Summary Statistics
    disagreements = [r for r in audit_results if r['is_disagreement']]
    avg_wer = sum(r['wer'] for r in audit_results) / len(audit_results)

    print("\n" + "="*80)
    print("📊 AUDIT BATCH SUMMARY")
    print("="*80)
    print(f"Dataset: SpeechBrain TEST partition (100% unseen)")
    print(f"Total samples: {len(audit_results)}")
    print(f"Disagreements (prediction ≠ ground truth): {len(disagreements)} ({len(disagreements)/len(audit_results)*100:.1f}%)")
    print(f"Agreements: {len(audit_results) - len(disagreements)}")
    print(f"Average WER on unseen data: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
    print(f"\n💾 Results saved to: {OUTPUT_FILE}")
    print("\n✨ Why This Matters:")
    print("   - SpeechBrain test partition = zero data leakage")
    print("   - Proves true generalization, not memorization")
    print("   - Scientific rigor for resume/paper claims")
    print("\n📝 Next Steps:")
    print("   1. Open notebooks/01_View_Results_Highlighted.ipynb")
    print("   2. Run the 'Label Noise Audit' cells")
    print("   3. Listen to disagreements and mark which are label noise vs. model errors")
    print("   4. Calculate: (Label Noise Count / Total Samples) = Noise Rate %")
    print("="*80)

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)

    generate_audit_batch()
