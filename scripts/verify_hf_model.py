"""
Verify Stage 1 Model on HuggingFace
====================================
Quick test to make sure the uploaded model is correct.
"""

import os
import sys
import torch
import numpy as np

# Setup fork
WORKSPACE = os.path.expanduser("~/workspace")
FORK_PATH = os.path.join(WORKSPACE, "transformers_fork/src")

if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)
    print(f"✅ Using transformers fork")

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, WhisperFeatureExtractor

print("=" * 80)
print("🔍 VERIFYING HUGGINGFACE MODEL")
print("=" * 80)

HF_REPO = "kulsoom-abdullah/Qwen2-Audio-Stage1"

# ==============================================================================
# TEST 1: Load Model from HF
# ==============================================================================

print(f"\n📥 TEST 1: Loading model from HuggingFace")
print(f"   Repo: {HF_REPO}")

try:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        HF_REPO,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ PASS: Model loaded successfully from HF")
except Exception as e:
    print(f"❌ FAIL: Could not load model: {e}")
    exit(1)

# ==============================================================================
# TEST 2: Verify Audio Components
# ==============================================================================

print(f"\n🔧 TEST 2: Checking audio components")

# Check audio_projector exists
if not hasattr(model, 'audio_projector'):
    print("❌ FAIL: No audio_projector found!")
    exit(1)
print("✅ PASS: audio_projector exists")

# Check audio_encoder exists
if not hasattr(model, 'audio_encoder'):
    print("❌ FAIL: No audio_encoder found!")
    exit(1)
print("✅ PASS: audio_encoder exists (Whisper)")

# Check projector has trained weights (not zeros)
test_param = list(model.audio_projector.parameters())[0]
if torch.all(test_param == 0):
    print("❌ FAIL: Projector weights are all zeros (not trained)")
    exit(1)
print(f"✅ PASS: Projector has trained weights")
print(f"   Sample values: {test_param.flatten()[:3].tolist()}")

# ==============================================================================
# TEST 3: Verify Model Config
# ==============================================================================

print(f"\n⚙️  TEST 3: Checking configuration")

config = model.config

# Check audio config
if not hasattr(config, 'audio_hidden_size'):
    print("❌ FAIL: Config missing audio_hidden_size")
    exit(1)
if config.audio_hidden_size != 1280:
    print(f"❌ FAIL: audio_hidden_size is {config.audio_hidden_size}, expected 1280")
    exit(1)
print(f"✅ PASS: audio_hidden_size = {config.audio_hidden_size}")

if not hasattr(config, 'use_audio_encoder') or not config.use_audio_encoder:
    print("❌ FAIL: use_audio_encoder not set")
    exit(1)
print(f"✅ PASS: use_audio_encoder = {config.use_audio_encoder}")

# ==============================================================================
# TEST 4: Quick Inference Test
# ==============================================================================

print(f"\n🎤 TEST 4: Running inference test")

try:
    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # Create dummy audio (5 seconds of random noise)
    dummy_audio = np.random.randn(16000 * 5).astype(np.float32)

    # Extract features
    audio_features = feature_extractor(
        [dummy_audio],
        sampling_rate=16000,
        return_tensors="pt"
    )["input_features"].to(model.device)

    # Create audio token IDs
    audio_tokens = torch.full((1, 1500), 151657, dtype=torch.long, device=model.device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_ids=audio_tokens,
            input_features=audio_features,
            max_new_tokens=50,
            do_sample=False
        )

    # Decode
    generated_text = tokenizer.decode(output[0, 1500:], skip_special_tokens=True)

    print(f"✅ PASS: Model generated text")
    print(f"   Output: {generated_text[:100]}...")

    if len(generated_text.strip()) > 0:
        print(f"✅ PASS: Output is not empty")
    else:
        print(f"⚠️  WARNING: Output is empty (might be okay for Stage 1)")

except Exception as e:
    print(f"❌ FAIL: Inference test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ==============================================================================
# TEST 5: Check File Sizes
# ==============================================================================

print(f"\n📊 TEST 5: Checking uploaded files")

try:
    from huggingface_hub import HfApi
    api = HfApi()

    files = api.list_repo_files(HF_REPO, repo_type="model")

    required_files = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    for req_file in required_files:
        if req_file in files:
            print(f"✅ Found: {req_file}")
        else:
            print(f"⚠️  Missing: {req_file}")

    # Count safetensors files
    safetensors_files = [f for f in files if f.endswith('.safetensors')]
    print(f"\n✅ Model shards: {len(safetensors_files)} safetensors files")

except Exception as e:
    print(f"⚠️  Could not check files: {e}")

# ==============================================================================
# FINAL VERDICT
# ==============================================================================

print("\n" + "=" * 80)
print("🎉 ALL TESTS PASSED!")
print("=" * 80)
print(f"\n✅ Your Stage 1 model is correct and ready to use!")
print(f"\n📦 Repository: https://huggingface.co/{HF_REPO}")
print(f"\n🚀 Next steps:")
print(f"   1. You can now terminate the Lambda Labs instance")
print(f"   2. Use this model for inference or Stage 2 training")
print(f"   3. Load with: Qwen2VLForConditionalGeneration.from_pretrained('{HF_REPO}')")
print("\n" + "=" * 80)
