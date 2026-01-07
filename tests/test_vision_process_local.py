#!/usr/bin/env python3
"""
Test the locally modified vision_process.py before installing it.
"""

import sys
import os

# Add the current directory to sys.path so we can import the local vision_process
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("="*70)
print("🧪 Testing Local vision_process.py")
print("="*70)

# Import the local version
try:
    # Import your local modified version
    from vision_process import process_vision_info, extract_vision_info, fetch_audio
    print("✅ Successfully imported local vision_process functions")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Also need the dataset sample
from datasets import load_dataset
import io
import numpy as np

print("\n" + "="*70)
print("📊 Test 1: Load a sample from the dataset")
print("="*70)

# Load one sample
dataset = load_dataset("speechbrain/LargeScaleASR", "small", streaming=True)
sample = next(iter(dataset['train']))

print(f"Sample keys: {sample.keys()}")
print(f"Transcript: {sample['text'][:50]}...")

print("\n" + "="*70)
print("📊 Test 2: Test fetch_audio directly")
print("="*70)

# Create a simple element dict like the one in your formatted data
test_element = {
    "type": "audio",
    "audio": sample['wav']['bytes']  # Raw bytes from dataset
}

try:
    audio_array = fetch_audio(test_element, sr=16000)
    print(f"✅ fetch_audio succeeded!")
    print(f"   Audio shape: {audio_array.shape}")
    print(f"   Audio dtype: {audio_array.dtype}")
    print(f"   Duration: {len(audio_array)/16000:.2f} seconds")
    print(f"   Sample values: {audio_array[:5]}")
except Exception as e:
    print(f"❌ fetch_audio failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("📊 Test 3: Format data and test extract_vision_info")
print("="*70)

# Use your format_data function
def format_data(sample):
    """Maps a raw SpeechBrain sample to the Qwen2-VL conversation format."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "audio": sample['wav']['bytes']
                },
                {
                    "type": "text",
                    "text": "Transcribe the audio."
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample['text']
                }
            ]
        }
    ]

formatted_sample = format_data(sample)
print(f"Formatted sample roles: {[msg['role'] for msg in formatted_sample]}")

# Extract vision info from the user message
user_content = formatted_sample[0]['content']
print(f"User content items: {len(user_content)}")
print(f"Content types: {[item.get('type') for item in user_content]}")

try:
    vision_infos = extract_vision_info(formatted_sample)
    print(f"✅ extract_vision_info succeeded!")
    print(f"   Extracted {len(vision_infos)} vision elements")
    for i, info in enumerate(vision_infos):
        print(f"   Element {i}: type={info.get('type')}, keys={list(info.keys())}")
except Exception as e:
    print(f"❌ extract_vision_info failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("📊 Test 4: Full process_vision_info pipeline")
print("="*70)

try:
    # ✅ CORRECT: Pass the full conversation, not vision_infos
    result = process_vision_info(formatted_sample)

    if len(result) == 3:
        image_inputs, video_inputs, audio_inputs = result
        video_kwargs = None
    elif len(result) == 4:
        image_inputs, video_inputs, audio_inputs, video_kwargs = result
    else:
        raise ValueError(f"Unexpected number of return values: {len(result)}")

    print(f"✅ process_vision_info succeeded!")
    print(f"\nResults:")
    print(f"   image_inputs: {type(image_inputs)} - {image_inputs}")
    print(f"   video_inputs: {type(video_inputs)} - {video_inputs}")
    print(f"   audio_inputs: {type(audio_inputs)}")

    if audio_inputs is not None:
        print(f"      Number of audio inputs: {len(audio_inputs)}")
        print(f"      First audio shape: {audio_inputs[0].shape}")
        print(f"      First audio duration: {len(audio_inputs[0])/16000:.2f}s")

    if video_kwargs is not None:
        print(f"   video_kwargs: {video_kwargs}")

except Exception as e:
    print(f"❌ process_vision_info failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("📊 Test 5: Verify audio content")
print("="*70)

if audio_inputs is not None and len(audio_inputs) > 0:
    audio = audio_inputs[0]

    # Basic checks
    checks = [
        (isinstance(audio, np.ndarray), "Is numpy array"),
        (audio.dtype == np.float32, "Correct dtype (float32)"),
        (len(audio) > 0, "Has samples"),
        (len(audio) / 16000 > 0, "Non-zero duration"),
        (-1.0 <= audio.min() <= 1.0, "Values in valid range (min)"),
        (-1.0 <= audio.max() <= 1.0, "Values in valid range (max)"),
    ]

    all_passed = True
    for passed, description in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {description}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Your vision_process.py is working correctly!")
    else:
        print("\n⚠️  Some checks failed. Review the output above.")
else:
    print("❌ No audio inputs returned!")

print("\n" + "="*70)
print("🏁 Testing Complete")
print("="*70)
