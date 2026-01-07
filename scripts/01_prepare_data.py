"""
01_prepare_data.py
==================
Downloads SpeechBrain data, decodes audio, and creates Qwen-formatted JSONs.
Used for BOTH Stage 1 and Stage 2 training.
"""

import os
import json
import soundfile as sf
import librosa
import numpy as np
import io
from tqdm import tqdm
from datasets import load_dataset, Audio

# --- SMART PATHS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "stage2_full")

TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
EVAL_JSON = os.path.join(DATA_DIR, "eval.json")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")

TRAIN_SAMPLES = 20000
EVAL_SAMPLES = 200

def prepare_data():
    if os.path.exists(TRAIN_JSON) and os.path.exists(EVAL_JSON):
        print(f"✅ Data already exists at {DATA_DIR}")
        return

    os.makedirs(AUDIO_DIR, exist_ok=True)

    print("="*80)
    print("📊 PREPARING DATASET (SpeechBrain)")
    print("="*80)

    # 1. Load Dataset
    print("🔄 Loading SpeechBrain Dataset (Streaming)...")
    dataset = load_dataset("speechbrain/LargeScaleASR", "small", streaming=True)
    # decode=False to bypass torchcodec issues
    train_stream = dataset["train"].cast_column("wav", Audio(decode=False))

    process_stream(train_stream, EVAL_SAMPLES, EVAL_JSON, "eval", skip_count=0)
    process_stream(train_stream, TRAIN_SAMPLES, TRAIN_JSON, "train", skip_count=EVAL_SAMPLES)
    
    print("\n✨ Data Prep Complete!")

def process_stream(stream, num_samples, json_path, prefix, skip_count=0):
    entries = []
    iterator = iter(stream)
    
    # Skip items used for other sets
    for _ in range(skip_count):
        next(iterator)

    saved_count = 0
    pbar = tqdm(total=num_samples, desc=f"Processing {prefix}")

    while saved_count < num_samples:
        try:
            item = next(iterator)
            
            # Audio Extraction
            if "wav" in item and "bytes" in item["wav"]:
                audio_bytes = item["wav"]["bytes"]
                y, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)
            elif "wav" in item and "array" in item["wav"]:
                y = item["wav"]["array"]
                if item["wav"]["sampling_rate"] != 16000:
                    y = librosa.resample(y, orig_sr=item["wav"]["sampling_rate"], target_sr=16000)
            else:
                continue

            # Text Extraction
            text = item.get("text", item.get("duration_text", "")).strip()
            if len(text) < 2: continue

            # Save .wav
            filename = f"{prefix}_{saved_count:05d}.wav"
            abs_path = os.path.join(AUDIO_DIR, filename)
            sf.write(abs_path, y, 16000)

            # Create JSON Entry
            entry = {
                "id": f"{prefix}_{saved_count}",
                "audio": f"audio/{filename}", # Relative to base_dir
                "ground_truth": text,
                "conversations": [
                    {
                        "role": "user", 
                        "content": "<|audio_bos|><|AUDIO|><|audio_eos|>\nTranscribe this audio."
                    },
                    {
                        "role": "assistant",
                        "content": text
                    }
                ]
            }
            entries.append(entry)
            saved_count += 1
            pbar.update(1)
            
        except StopIteration:
            break
        except Exception:
            continue
            
    pbar.close()

    with open(json_path, 'w') as f:
        json.dump(entries, f, indent=2)
    
    print(f"✅ Saved {len(entries)} to {json_path}")

if __name__ == "__main__":
    prepare_data()