"""
Test generate() Fix
===================
Verifies that .generate() now works correctly after fork modifications.
Should get exact match on overtrained nano samples.
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
print("🧪 TESTING .generate() FIX")
print("="*80)

# Load model
print(f"\n📥 Loading model...")
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

print("✅ Model loaded")

# Load dataset
with open(DATA_JSON, 'r') as f:
    data = json.load(f)

# Test sample
entry = data[0]
audio_path = os.path.join("./stage2_nano_data", entry['audio'])
ground_truth = entry['ground_truth']

user_content = entry['conversations'][0]['content']
instruction = user_content.split("<|audio_eos|>\n")[-1]

print(f"\n📝 Ground Truth: {ground_truth}")
print(f"🎯 Testing .generate() method...")

# Prepare audio
y, sr = librosa.load(audio_path, sr=16000, mono=True)
inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(model.device).to(torch.bfloat16)

# Build prompt
audio_tokens = [151657] * 1500
input_ids_audio = torch.tensor([audio_tokens], device=model.device)

p1 = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
p2 = tokenizer.encode(f"<|audio_eos|>\n{instruction}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt").to(model.device)

input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)
attention_mask = torch.ones_like(input_ids)

# USE .generate() - THIS IS THE TEST!
print("\n🚀 Calling model.generate()...")
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        input_features=input_features,  # This should now work!
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

output_text = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

print(f"\n🤖 Model Output: {output_text.strip()}")

# Check result
print("\n" + "="*80)
if ground_truth.strip() == output_text.strip():
    print("✅ SUCCESS! .generate() FIX WORKS!")
    print("   Exact match on overtrained sample.")
    print("\n   The fork modifications are correct!")
    print("\n   You can now use .generate() for inference.")
elif ground_truth.strip().lower() in output_text.strip().lower():
    print("⚠️  PARTIAL SUCCESS")
    print("   Output contains ground truth but not exact.")
    print("   This is acceptable - model is using audio.")
else:
    print("❌ FAILED!")
    print("   Output doesn't match ground truth.")
    print("\n   Possible issues:")
    print("   - Fork edits not saved/reloaded")
    print("   - Wrong file edited")
    print("   - Model still cached old version")

print("="*80)
