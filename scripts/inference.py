
#!/usr/bin/env python3
"""
Quick inference script for custom audio files.
Usage: python scripts/inference.py --audio path/to/file.wav
"""

import sys
import torch
import librosa
import argparse
import os

# -----------------------------------------------------------------------------
# 🔧 CRITICAL: Force usage of local transformers fork
# -----------------------------------------------------------------------------
# We must insert this at position 0 to override any installed transformers version
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fork_path = os.path.join(project_root, "transformers_fork", "src")
sys.path.insert(0, fork_path)

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    WhisperFeatureExtractor
)

def transcribe(audio_path, model_path="kulsoom-abdullah/Qwen2-Audio-7B-Transcription"):
    print(f"⏳ Loading model from: {model_path}...")
    
    # 1. Load Model & Processor
    #    Note: We rely on the local fork logic, so we don't strictly need trust_remote_code=True 
    #    if we are loading the weights into our local class, but it's safer to keep it.
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")

    # 2. Process Audio
    print(f"🎧 Processing audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(model.device).to(torch.bfloat16)

    # 3. Build Prompt (Manually injecting audio tokens)
    AUDIO_TOKEN_ID, NUM_AUDIO_TOKENS = 151657, 1500
    audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
    input_ids_audio = torch.tensor([audio_tokens], device=model.device)

    p1 = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
    p2 = tokenizer.encode("<|audio_eos|>\nTranscribe this audio.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt").to(model.device)

    input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)

    # 4. Generate
    print("🤖 Generating transcription...")
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids, 
            input_features=input_features,
            attention_mask=torch.ones_like(input_ids), 
            max_new_tokens=128, 
            do_sample=False
        )

    # 5. Decode
    transcription = tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio file using Qwen2-Audio-Graft")
    parser.add_argument("--audio", required=True, help="Path to audio file (.wav, .mp3, .m4a)")
    parser.add_argument("--model", default="kulsoom-abdullah/Qwen2-Audio-7B-Transcription", help="HuggingFace model path")
    args = parser.parse_args()

    try:
        result = transcribe(args.audio, args.model)
        print("\n" + "="*50)
        print(f"📝 TRANSCRIPTION RESULT:")
        print("="*50)
        print(result)
        print("="*50 + "\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")