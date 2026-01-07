"""
Test if Model is Using Audio
=============================
Tests if different audio produces different outputs.
If outputs are identical → model is deaf (ignoring audio).
"""

import os
import sys
import torch
import librosa
import json

# Load fork
FORK_PATH = os.path.abspath("transformers_fork/src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
else:
    FORK_PATH = "/workspace/transformers_fork/src"
    if os.path.exists(FORK_PATH):
        sys.path.insert(0, FORK_PATH)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, WhisperFeatureExtractor

MODEL_PATH = "./stage2_nano_bulletproof/merged_model"
DATA_JSON = "./stage2_nano_data/stage2_nano.json"

print("="*80)
print("🧪 TESTING AUDIO SENSITIVITY")
print("="*80)
print("\nThis test checks if the model produces different outputs")
print("for different audio with the SAME instruction.")
print("\nIf outputs are identical → Model is DEAF (ignoring audio)")
print("If outputs differ → Model is using audio")
print("="*80)

# Load model
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

# Load dataset
with open(DATA_JSON, 'r') as f:
    data = json.load(f)

# Test with 3 different audio files, SAME instruction
SAME_INSTRUCTION = "Transcribe this audio."
TEST_SAMPLES = [0, 10, 20]  # Different audio samples

outputs = []

for idx in TEST_SAMPLES:
    entry = data[idx]
    audio_path = os.path.join("./stage2_nano_data", entry['audio'])

    print(f"\n{'='*80}")
    print(f"Audio: {entry['id']}")
    print(f"Ground truth: {entry['ground_truth'][:60]}...")
    print("-"*80)

    # Load audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(model.device).to(torch.bfloat16)

    # Build prompt
    audio_tokens = [151657] * 1500
    input_ids_audio = torch.tensor([audio_tokens], device=model.device)

    p1 = tokenizer.encode("<|im_start|>user\\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
    p2 = tokenizer.encode(f"<|audio_eos|>\\n{SAME_INSTRUCTION}<|im_end|>\\n<|im_start|>assistant\\n", add_special_tokens=False, return_tensors="pt").to(model.device)

    input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)
    attention_mask = torch.ones_like(input_ids)

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=None,  # Disable sampling params
            top_p=None,
            top_k=None
        )

    output = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    outputs.append(output.strip())

    print(f"Output: {output.strip()}")

# Check if outputs are identical
print("\n" + "="*80)
print("🔍 ANALYSIS")
print("="*80)

all_same = all(out == outputs[0] for out in outputs)

if all_same:
    print("\n❌ CRITICAL FAILURE: All outputs are IDENTICAL!")
    print(f"\n   Same output for all 3 different audio files:")
    print(f"   \"{outputs[0]}\"")
    print("\n   This means the model is COMPLETELY IGNORING audio input.")
    print("\n   Possible causes:")
    print("   1. Audio encoder has LoRA (wasn't frozen correctly)")
    print("   2. Audio projector wasn't trained")
    print("   3. input_features not reaching model during generation")
    print("   4. Fork not loaded correctly")
else:
    print("\n✅ GOOD: Outputs are DIFFERENT for different audio!")
    print("\n   This confirms the model IS using audio input.")
    print("   The problem is likely:")
    print("   - Insufficient training")
    print("   - Wrong loss masking")
    print("   - Data format issues")

    print("\n   Outputs:")
    for i, out in enumerate(outputs):
        print(f"   {i+1}. \"{out[:50]}...\"")

print("\n" + "="*80)
