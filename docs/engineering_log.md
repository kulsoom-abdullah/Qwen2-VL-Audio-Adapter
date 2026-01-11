# 🛠️ Engineering Log: Challenges & Solutions

> **Summary**: This document chronicles the real-world engineering challenges encountered while building the Qwen2-VL Audio Adapter model.

## 🐛 The "Label Noise" Discovery
**Problem**: The model struggled to converge below 0.8 loss in early Stage 2 trials.
**Investigation**: I inspected the SpeechBrain dataset and found inconsistencies in the ground truth text (e.g., "Commission's" vs "Commissions").
**Solution**: Implemented a **Robust Collator** in `scripts/04_train_stage2.py`.
* **Standardizes formatting**: Converts Windows line endings (`\r\n`) to Unix (`\n`) and trims padding spaces to ensure identical words map to identical token IDs. Without this, `"polyp"` and `" polyp"` would be tokenized differently, causing incorrect loss calculations even when the model predicts the correct word.
* **Result**: Evaluating with `wer` (Word Error Rate) instead of exact string matching proved the model was actually performing well (3.6% WER).

## 🛑 Architectural Boundary Verification: The 30s Limit
**Purpose**: Verify the hard architectural limit imposed by Whisper's fixed positional embeddings (1,500 tokens = 30 seconds of audio).

**Verification Test**:
* Intentionally passed `truncation=False` to the processor with >30s audio.
* **Result**: Immediate crash (RuntimeError due to dimension mismatch), confirming the boundary.

**Architecture Insight**:
* The **Whisper Processor** (CPU preprocessing) can handle arbitrary-length audio.
* The **Whisper Encoder** (neural network) has a fixed matrix size for exactly 1,500 positional embeddings.
* **Why the flag exists**: The processor needs to know whether to truncate long audio to fit the encoder's fixed architecture, or to reject it entirely.

**Engineering Decision**:
* We enforced `truncation=True` in the data pipeline to guarantee stable tensor shapes `[Batch, 1500, 1280]` for the linear projector.
* **Alternative approach**: For >30s audio, a sliding window (chunking) strategy would be required—processing 0-30s, then 30-60s, and stitching embeddings together. This version explicitly supports only the first 30 seconds.

## 🔧 LoRA Targeting
**Challenge**: In Stage 2, we needed to fine-tune the LLM without breaking the frozen Audio Encoder.
**Mistake**: Using generic target names like `["q_proj", "v_proj"]` accidentally targeted the Whisper Encoder layers too (adding 584 unwanted parameters).
**Fix**: Switched to **Regex Targeting** in `peft_config`:
```python
target_modules=r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"