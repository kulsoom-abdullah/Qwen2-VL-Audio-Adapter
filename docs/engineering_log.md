# 🛠️ Engineering Log: Challenges & Solutions

> **Summary**: This document chronicles the real-world engineering challenges encountered while building the Audio-Grafted Qwen2-VL model.

## 🐛 The "Label Noise" Discovery
**Problem**: The model struggled to converge below 0.8 loss in early Stage 2 trials.
**Investigation**: I inspected the SpeechBrain dataset and found inconsistencies in the ground truth text (e.g., "Commission's" vs "Commissions").
**Solution**: Implemented a **Robust Collator** in `scripts/04_train_stage2.py`.
* Normalizes whitespace.
* Handles inconsistent newline characters (`\n` vs `\\n`).
* Result: Evaluating with `wer` (Word Error Rate) instead of exact string matching proved the model was actually performing well (3.6% WER).

## 🛑 The "Truncation=False" Verification Experiment
**Hypothesis**: Since the Whisper Encoder has hardcoded positional embeddings for 1,500 tokens (30s), disabling truncation should cause a dimension mismatch crash.

**Experiment**:
* We intentionally passed `truncation=False` to the processor with >30s audio.
* **Result**: Immediate crash (RuntimeError due to dimension mismatch).

**Architecture Insight**:
* The **Whisper Processor** allows arbitrary lengths (preprocessing layer).
* The **Whisper Encoder** enforces a strict 1,500-token limit (model layer).
* **Conclusion**: This confirmed that our "Grafting" strategy must strictly enforce `truncation=True` (or implement custom chunking) to align with the frozen encoder's tensor shape `[Batch, 1500, 1280]`.

**Decision**: We enforced `truncation=True` in the data pipeline to guarantee stable tensor shapes for the linear projector.

## 🔧 LoRA Targeting
**Challenge**: In Stage 2, we needed to fine-tune the LLM without breaking the frozen Audio Encoder.
**Mistake**: Using generic target names like `["q_proj", "v_proj"]` accidentally targeted the Whisper Encoder layers too (adding 584 unwanted parameters).
**Fix**: Switched to **Regex Targeting** in `peft_config`:
```python
target_modules=r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"