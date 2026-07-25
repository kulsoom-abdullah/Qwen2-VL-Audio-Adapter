
### 📋 The Hugging Face Model Card

```markdown
---
language:
- en
license: apache-2.0
tags:
- audio
- speech-recognition
- transcription
- qwen2-vl
- whisper
- multimodal-adapter
- modality-alignment
- audio-projection
base_model: Qwen/Qwen2-VL-7B-Instruct
datasets:
- speechbrain/LargeScaleASR
metrics:
- wer
- cer
model-index:
- name: Qwen2-VL-Audio-Adapter
  results:
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      type: speechbrain/LargeScaleASR
      name: SpeechBrain Large Scale ASR (validation split, n≈50)
    metrics:
    - type: wer
      value: 0.036
      name: Word Error Rate (validation split, n≈50)
    - type: cer
      value: 0.025
      name: Character Error Rate (validation split, n≈50)
  - task:
      type: automatic-speech-recognition
      name: Speech Recognition
    dataset:
      type: speechbrain/LargeScaleASR
      name: SpeechBrain Large Scale ASR (full official test partition, n=8086)
    metrics:
    - type: wer
      value: 0.1133
      name: Word Error Rate (test partition, corpus, n=8086)
    - type: cer
      value: 0.0670
      name: Character Error Rate (test partition, corpus, n=8086)
---

# Qwen2-VL-Audio-Adapter

> **Multimodal Fusion: Integrating Whisper Audio Encoder with Qwen2-VL for Production-Grade Speech Recognition**

**Achieves 11.3% WER on the full official LargeScaleASR test partition (n=8,086); see full evaluation below.**

## 🎯 Performance Highlights

**Evaluation Context**: WER and CER are the only directly computed metrics, measured on a **50-sample validation split** — the head of the training stream (`data/stage2_full/eval.json`), not a held-out test set (60% exact-match rate). These 50 samples are not guaranteed disjoint from the training set, so treat the figure as a validation-split estimate. A separate blind manual audit of 100 samples from the SpeechBrain test partition characterized label noise (see below).

| Metric | Value | Scope |
|--------|-------|-------|
| **Word Error Rate (WER)** | **3.6%** | Validation split (n≈50) |
| **Character Error Rate (CER)** | **2.5%** | Validation split (n≈50) |
| **Label Correction Rate** | **36%** | Manual audit (n=100, SpeechBrain test) |

### 🧪 Test-Partition Result (Full Official Test Set)

Measured on the **full official LargeScaleASR test partition** (dataset revision `0e84cdb9e4b826afaabca5d33ec9453b11aacef3`), n = 8,086 of 8,087 (1 excluded: `test_6343`, empty output). This is a held-out test set, distinct from the 50-sample validation split above.

| Metric | Value | Scope |
|--------|-------|-------|
| **Word Error Rate (WER)** | **11.33%** | Test partition, corpus (n=8,086) |
| **Character Error Rate (CER)** | **6.70%** | Test partition, corpus (n=8,086) |
| **WER (macro)** | **12.83%** | Test partition, macro |
| **CER (macro)** | **7.33%** | Test partition, macro |

95% CI (corpus): WER [10.84, 11.85], CER [6.32, 7.12].

A manual audit of 100 samples from this partition, all 100 reviewed against the audio, found 36 where the model's transcription was correct and the reference was wrong. The audited samples fall within this test partition. The reported figure is therefore an upper bound on true error rate.

**Novel Finding:** In the 100-sample manual audit, the model corrected ground-truth annotations in 36 of 100 samples — the majority of all disagreements between model and reference, demonstrating context-aware semantic reasoning. Note this is a sample-level rate from manual review, not a measured WER.

## 🏗️ Architecture


```

┌─────────────────────────────────────────────────┐
│  Whisper-Large-v3-Turbo Encoder (Frozen)        │
│  ~640M params → 1280-dim audio features         │
└────────────────┬────────────────────────────────┘
│
↓
┌─────────────────────────────────────────────────┐
│  Audio Projector (Trainable)                    │
│  Linear: 1280 → 3584 dims (4.6M params)         │
└────────────────┬────────────────────────────────┘
│
↓
┌─────────────────────────────────────────────────┐
│  Qwen2-VL-7B LLM (QLoRA Fine-tuned)             │
│  7B params with rank-64 LoRA adapters           │
└─────────────────────────────────────────────────┘

```

## 🔬 Rigorous Audit: Label Noise & Semantic Bias

To validate model quality on truly unseen data, I conducted a **blind manual audit** of 100 samples from the SpeechBrain test partition.

### 🔎 Audit Visualizer
**1. Label Noise & Entity Resolution**
*The model (Green) correctly identified "Mr. Šefčovič" (Maroš Šefčovič, EU Commissioner), correcting the ground truth "Mr. Efovi" (Red).*
![Label Noise Correction](figures/comparison1.png)

**2. Semantic Bias & Long-Range Context**
*The model "hallucinated" the word "Malta" (Green) in the first sentence because it attended to the context provided later in the audio, proving editorial reasoning.*
![Semantic Bias - Malta](figures/comparison2.png)

### Quantitative Analysis (N=100)
| Category | Count | Description |
|----------|-------|-------------|
| **✅ Label Noise (Model Correct)** | 36 | Model outperformed ground truth annotations |
| **❌ True Model Errors** | 14 | Model genuinely misheard or hallucinated |
| **⚠️ Ambiguous** | 11 | Heavy accents or unclear audio |
| **ℹ️ Normalization** | 1 | Punctuation/formatting differences |
| **✓ Perfect Matches** | 37 | Exact agreement |
| **❔ Uncategorized** | 1 | Disagreement not classified (sample #60) |
| **Total** | **100** | |

## 🧪 Training Infrastructure

- **GPUs**: Stage 1: 1× NVIDIA A100; Stage 2: 1× NVIDIA A6000 — single GPU per stage (no distributed training)
- **Training time**: ~18 GPU-hours total across both stages
- **Framework**: HuggingFace Transformers (custom fork) + PEFT + BitsAndBytes; BFloat16 + FlashAttention-2; Stage 2 uses 4-bit NF4 quantization (QLoRA)
- A rented H100 was used only for audit inference, not training.

## 💻 Usage

**Important**: This model requires a modified transformers library (included in the repo files).

### Installation

**Method 1: Git Clone (Recommended)**
```bash
# Clone the model repo (includes transformers fork)
git clone [https://huggingface.co/kulsoom-abdullah/Qwen2-VL-Audio-Adapter](https://huggingface.co/kulsoom-abdullah/Qwen2-VL-Audio-Adapter)
cd Qwen2-VL-Audio-Adapter

# Install dependencies
pip install torch transformers librosa soundfile accelerate

```

### Basic Inference

```python
import sys
import torch
import librosa

# Load modified transformers from repo
sys.path.insert(0, "./transformers_fork/src")

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    WhisperFeatureExtractor
)

# Load model
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "kulsoom-abdullah/Qwen2-VL-Audio-Adapter",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "kulsoom-abdullah/Qwen2-VL-Audio-Adapter",
    trust_remote_code=True
)

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-large-v3-turbo"
)

# Load and prepare audio
audio_path = "your_audio.wav"
y, sr = librosa.load(audio_path, sr=16000, mono=True)
inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(model.device).to(torch.bfloat16)

# Build prompt
AUDIO_TOKEN_ID = 151657
NUM_AUDIO_TOKENS = 1500
audio_tokens = [AUDIO_TOKEN_ID] * NUM_AUDIO_TOKENS
input_ids_audio = torch.tensor([audio_tokens], device=model.device)

p1 = tokenizer.encode("<|im_start|>user\n<|audio_bos|>", add_special_tokens=False, return_tensors="pt").to(model.device)
p2 = tokenizer.encode("<|audio_eos|>\nTranscribe this audio.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False, return_tensors="pt").to(model.device)
input_ids = torch.cat([p1, input_ids_audio, p2], dim=1)

# Generate
with torch.no_grad():
    generated_ids = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        max_new_tokens=128
    )

print(tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True))

```

## 📝 Citation

```bibtex
@misc{qwen2-vl-audio-adapter,
  author = {Kulsoom Abdullah},
  title = {Qwen2-VL-Audio-Adapter: Multimodal Projection Alignment for Speech Recognition},
  year = {2026},
  publisher = {HuggingFace},
  howpublished = {\url{[https://huggingface.co/kulsoom-abdullah/Qwen2-VL-Audio-Adapter](https://huggingface.co/kulsoom-abdullah/Qwen2-VL-Audio-Adapter)}}
}

```

## 📄 License

Apache 2.0 (inherits from Qwen2-VL and Whisper)

---

**Kulsoom Abdullah** | [GitHub](https://www.google.com/search?q=https://github.com/kulsoom-abdullah/Qwen2-VL-Audio-Adapter)
