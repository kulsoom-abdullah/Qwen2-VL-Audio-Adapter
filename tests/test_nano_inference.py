"""
Test Inference on Nano Dataset
================================
Tests the merged model on samples it was trained on.
With loss ~0.0001, should have nearly perfect transcriptions.
"""

import os
import sys
import torch
import librosa
import json

# Setup fork
FORK_PATH = os.path.abspath("transformers_fork/src")
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
else:
    FORK_PATH = "/workspace/transformers_fork/src"
    if os.path.exists(FORK_PATH):
        sys.path.insert(0, FORK_PATH)
    else:
        print("❌ Fork not found!")
        sys.exit(1)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, WhisperFeatureExtractor

# Config
MODEL_PATH = "./stage2_nano_bulletproof/merged_model"
DATA_JSON = "./stage2_nano_data/stage2_nano.json"
NUM_SAMPLES = 5  # Test first 5 samples

print("="*80)
print("🧪 TESTING NANO INFERENCE")
print("="*80)

# Load model
print(f"\n📥 Loading model from: {MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    print("   Did training complete successfully?")
    sys.exit(1)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

# Fix token IDs
tokenizer.pad_token_id = 151643
tokenizer.eos_token_id = 151645
model.config.pad_token_id = 151643
model.config.eos_token_id = 151645

print("✅ Model loaded")

# Load dataset
print(f"\n📊 Loading dataset...")
with open(DATA_JSON, 'r') as f:
    data_entries = json.load(f)

print(f"✅ Loaded {len(data_entries)} samples")

# Audio token config
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500

# Test loop
print("\n" + "="*80)
print("🔍 TESTING SAMPLES")
print("="*80)

matches = 0
partial_matches = 0

for i in range(min(NUM_SAMPLES, len(data_entries))):
    entry = data_entries[i]

    # Load audio
    audio_path = os.path.join("./stage2_nano_data", entry['audio'])
    ground_truth = entry['ground_truth']

    # Extract instruction
    user_content = entry['conversations'][0]['content']
    instruction = user_content.split("<|audio_eos|>\\n")[-1]

    print(f"\n{'='*80}")
    print(f"Sample {i+1}/{NUM_SAMPLES}")
    print(f"ID: {entry['id']}")
    print(f"Instruction: \"{instruction}\"")
    print("-"*80)

    # Prepare audio
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(model.device).to(torch.bfloat16)

    # Build prompt with correct format
    audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
    input_ids_audio = torch.tensor([audio_tokens], device=model.device, dtype=torch.long)

    # Tokenize parts
    p1 = tokenizer.encode("<|im_start|>user\\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
    p2 = tokenizer.encode(f"<|audio_eos|>\\n{instruction}<|im_end|>\\n<|im_start|>assistant\\n", add_special_tokens=False, return_tensors="pt").to(model.device)

    # Combine
    input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)
    attention_mask = torch.ones_like(input_ids)

    # Generate
    print("🤖 Generating...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            input_features=input_features,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=False,  # Greedy for exact match
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode output
    output_text = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Display results
    print(f"\n📝 Ground Truth:")
    print(f"   {ground_truth}")
    print(f"\n🤖 Model Output:")
    print(f"   {output_text.strip()}")

    # Check match
    gt_clean = ground_truth.strip().lower()
    out_clean = output_text.strip().lower()

    if gt_clean == out_clean:
        print("\n✅ EXACT MATCH!")
        matches += 1
    elif gt_clean in out_clean or out_clean in gt_clean:
        print("\n⚠️  PARTIAL MATCH")
        partial_matches += 1
    elif any(word in out_clean for word in gt_clean.split()[:5]):
        print("\n⚠️  SOME WORDS MATCH")
        partial_matches += 1
    else:
        print("\n❌ MISMATCH")

        # Check for common failure modes
        if "<|audio_eos|>" in output_text:
            print("   ⚠️  Model outputting special tokens")
        elif output_text.strip() == "":
            print("   ⚠️  Empty output")
        elif "1.0" in output_text or "0.0" in output_text:
            print("   ⚠️  Numeric gibberish (model may not be using audio)")

print("\n" + "="*80)
print("📊 RESULTS SUMMARY")
print("="*80)

total = min(NUM_SAMPLES, len(data_entries))
print(f"\nTested: {total} samples")
print(f"Exact matches: {matches}/{total} ({matches/total*100:.1f}%)")
print(f"Partial matches: {partial_matches}/{total} ({partial_matches/total*100:.1f}%)")
print(f"Mismatches: {total - matches - partial_matches}/{total}")

print("\n" + "="*80)
print("🎯 INTERPRETATION")
print("="*80)

if matches >= total * 0.8:
    print("""
✅ EXCELLENT! Model is working perfectly.
   - High exact match rate on overtrained samples
   - Audio encoder + projector functioning correctly
   - Ready for full training!

Next steps:
   1. Prepare full dataset: python prepare_stage2_full.py
   2. Run full training: python train_stage2_full_BULLETPROOF.py
""")
elif matches + partial_matches >= total * 0.6:
    print("""
⚠️  GOOD! Model is working but not perfectly memorized.
   - Audio is being used (partial matches)
   - May need more training steps or lower learning rate
   - Can proceed to full training with caution

Consider:
   - Increasing max_steps for full training
   - Monitoring loss curves closely
""")
elif partial_matches > 0:
    print("""
⚠️  MARGINAL. Model is using audio but struggling.
   - Some transcription capability present
   - May have underfitted on nano set
   - Or data format issues

Investigate:
   - Check if loss actually decreased during training
   - Verify audio files are correct
   - Consider re-running nano with more steps
""")
else:
    print("""
❌ PROBLEM! Model is not working correctly.
   - No matches suggest audio not being used
   - Or critical bug in inference

Debug:
   1. Check if model outputs special tokens
   2. Verify input_features is reaching the model
   3. Check for errors in collator/format
   4. May need to retrain with fixes
""")

print("="*80)
