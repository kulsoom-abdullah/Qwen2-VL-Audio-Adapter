"""
setup_audit_data.py
===================
Downloads test samples from SpeechBrain LargeScaleASR (test partition).
Uses official test split - completely separate from training data.

This guarantees zero data leakage and proves true generalization.
"""

import os
import json
import soundfile as sf
import librosa
import io
from datasets import load_dataset
from tqdm import tqdm

# --- CONFIG ---
DATASET_NAME = "speechbrain/LargeScaleASR"
TEST_DATA_FILES = ["test/test-00000*"]  # Official test partition
NUM_SAMPLES = 200
OUTPUT_DIR = "data/audit_test"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio_clips")
JSON_PATH = os.path.join(OUTPUT_DIR, "test.json")

# Set cache directory for RunPod
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache"

def setup_data():
    print("="*80)
    print("🔬 LABEL NOISE AUDIT - DATA SETUP")
    print("="*80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Test partition: {TEST_DATA_FILES}")
    print(f"Target samples: {NUM_SAMPLES}")
    print("="*80)

    # Check if already exists
    if os.path.exists(JSON_PATH):
        print(f"\n⚠️  Data already exists at {JSON_PATH}")
        with open(JSON_PATH, 'r') as f:
            existing = json.load(f)
        print(f"   Found {len(existing)} existing samples")

        response = input("   Re-download? (y/n): ").lower()
        if response != 'y':
            print("\n✅ Using existing data. Run: python scripts/generate_audit_batch.py")
            return

    print(f"\n⬇️  Loading SpeechBrain test partition...")
    print("   (Using num_proc=1 to avoid server errors)")

    # Load test dataset with specific data files
    try:
        dataset = load_dataset(
            DATASET_NAME,
            data_files=TEST_DATA_FILES,
            num_proc=1,  # Avoid 502 errors
            cache_dir="/workspace/hf_cache"
        )
        # Extract the actual dataset
        test_data = dataset["train"]  # data_files loads into "train" key
        print(f"✅ Loaded {len(test_data)} total test samples")

    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        print("\n💡 TIP: Check your internet connection or try again")
        return

    os.makedirs(AUDIO_DIR, exist_ok=True)
    data_entries = []

    # Select samples to process
    num_available = min(len(test_data), NUM_SAMPLES)
    samples_to_process = test_data.select(range(num_available))

    print(f"\n💾 Processing {num_available} test samples...")
    saved_count = 0

    for idx, item in enumerate(tqdm(samples_to_process, desc="Processing test data")):
        try:
            # Extract audio (SpeechBrain format)
            if "wav" in item and "bytes" in item["wav"]:
                audio_bytes = item["wav"]["bytes"]
                y, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            elif "wav" in item and "array" in item["wav"]:
                y = item["wav"]["array"]
                if item["wav"]["sampling_rate"] != 16000:
                    y = librosa.resample(y, orig_sr=item["wav"]["sampling_rate"], target_sr=16000)
            else:
                continue

            # Extract text
            text = item.get("text", item.get("duration_text", "")).strip()
            if len(text) < 2:
                continue

            # Calculate duration
            duration = len(y) / 16000

            # Save .wav
            filename = f"test_{saved_count:03d}.wav"
            file_path = os.path.join(AUDIO_DIR, filename)
            sf.write(file_path, y, 16000)

            # JSON Entry
            data_entries.append({
                "id": f"test_{saved_count}",
                "audio": f"audio_clips/{filename}",
                "ground_truth": text,
                "duration": duration,
                "source": "speechbrain_test"
            })
            saved_count += 1

        except Exception as e:
            # Skip problematic samples
            continue

    # Save Index
    with open(JSON_PATH, 'w') as f:
        json.dump(data_entries, f, indent=2)

    print("\n" + "="*80)
    print("✅ AUDIT DATA SETUP COMPLETE")
    print("="*80)
    print(f"Downloaded: {len(data_entries)} test samples")
    print(f"Saved to: {JSON_PATH}")
    print(f"Audio directory: {AUDIO_DIR}")
    print("\n📊 Why This Matters:")
    print("   - SpeechBrain TEST partition = official held-out split")
    print("   - Zero overlap with training data (trained on 'small' partition)")
    print("   - Proves true generalization, not memorization")
    print("\n🚀 Next Step:")
    print("   python scripts/generate_audit_batch.py")
    print("="*80)

if __name__ == "__main__":
    setup_data()
