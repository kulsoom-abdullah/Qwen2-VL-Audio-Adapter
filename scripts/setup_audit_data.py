"""
setup_audit_data.py
===================
Downloads the FULL official test split of SpeechBrain LargeScaleASR (all 6 shards,
8,087 rows at the pinned revision), decodes audio to 16 kHz WAV, and writes test.json.

The test split is disjoint from the training data (which is drawn from the 'small' TRAIN
partition), so it is genuinely held-out — unlike data/stage2_full/eval.json, which is the
head of the train stream and is a validation split, not a test set.

NOTE: 8,087 audio samples is a large pull; run this on the GPU pod, not locally.
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
# All 6 shards of the official test split (test-00000..00005-of-00006 = 8,087 rows at
# the pinned revision). The previous glob "test/test-00000*" matched ONLY shard 0
# (1,348 rows), so the earlier draw was the first 200 rows of 1/6 of the split.
TEST_DATA_FILES = ["test/test-*.parquet"]
DATASET_REVISION = "0e84cdb9e4b826afaabca5d33ec9453b11aacef3"  # pinned; see PRE_REGISTRATION.md
NUM_SAMPLES = None  # None = FULL split; set an int ONLY for a quick smoke pull (head slice)
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
    print(f"Target samples: {'FULL split' if NUM_SAMPLES is None else NUM_SAMPLES}")
    print(f"Revision: {DATASET_REVISION}")
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
            revision=DATASET_REVISION,  # pinned for reproducibility
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

    # Process the FULL split by default (NUM_SAMPLES=None). A cap, if set, is a HEAD
    # slice for quick smoke pulls ONLY — never for a reported run (file order is not a
    # sample). For a reported subset, draw seeded random indices instead.
    if NUM_SAMPLES is None:
        samples_to_process = test_data
    else:
        samples_to_process = test_data.select(range(min(len(test_data), NUM_SAMPLES)))
    num_available = len(samples_to_process)

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

            # Save .wav  (5-digit pad: the full split is 8,087 rows)
            filename = f"test_{saved_count:05d}.wav"
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
