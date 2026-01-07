"""
00_graft_architecture.py
========================
Performs the "surgery" to graft Whisper-Large-v3-Turbo onto Qwen2-VL-7B.
Run this ONCE to create the initialized base model.
"""

import os
import sys
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig, AutoProcessor

# Setup Fork
FORK_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../transformers_fork/src"))
if os.path.exists(FORK_PATH):
    sys.path.insert(0, FORK_PATH)

OUTPUT_DIR = "./model_base_grafted"

def graft_model():
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️  Grafted model already exists at {OUTPUT_DIR}. Skipping.")
        return

    print("="*70)
    print("🎧 GRAFTING SURGERY: Qwen2-VL-7B + Whisper")
    print("="*70)

    # 1. Load Config & Enable Audio
    print("\n1. Loading 7B config with audio parameters...")
    config = Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    config.audio_hidden_size = 1280  # Whisper Turbo dimension
    config.use_audio_encoder = True

    # 2. Load Model (Triggers Custom Fork Logic)
    print("2. Loading Qwen2-VL-7B and grafting Whisper encoder...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        config=config,
        torch_dtype=torch.bfloat16
    )

    # 3. Load Processor
    print("3. Loading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # ------------------------------------------------------------------
    # 3.5. Tokenizer Surgery (The "Empty Seats" Logic)
    # ------------------------------------------------------------------
    print("3.5 Adding special audio tokens...")
    new_tokens = ["<|audio_pad|>", "<|audio_start|>", "<|audio_end|>"]
    
    # Add tokens to tokenizer
    num_added = processor.tokenizer.add_tokens(new_tokens, special_tokens=True)
    print(f"   Added {num_added} tokens: {new_tokens}")

    # Check against physical embedding size
    vocab_size = len(processor.tokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    
    print(f"   - New Vocab Size: {vocab_size}")
    print(f"   - Physical Embeddings: {embedding_size}")

    if vocab_size > embedding_size:
        print("   ⚠️  Resizing model embeddings to fit new tokens...")
        model.resize_token_embeddings(vocab_size)
    else:
        print("   ✅  No resize needed! Tokens fit in pre-allocated 'empty seats'.")

    # ------------------------------------------------------------------

    # 4. Save
    print(f"\n4. Saving grafted model to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("✅ Grafting Complete.")

if __name__ == "__main__":
    graft_model()