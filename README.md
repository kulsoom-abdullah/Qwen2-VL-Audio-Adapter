# Qwen2-VL-Audio-Graft

> **Architecture Grafting: Fusing Whisper Audio Encoder onto Qwen2-VL for Production-Grade Speech Recognition**

[![HuggingFace](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/kulsoom-abdullah/Qwen2-Audio-7B-Transcription)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![WER](https://img.shields.io/badge/WER-3.6%25-brightgreen)](https://huggingface.co/kulsoom-abdullah/Qwen2-Audio-7B-Transcription)

**Achieves commercial-grade ASR quality (WER 3.6%) with only 20K training samples** by grafting a Whisper-Large-v3-Turbo encoder onto Qwen2-VL-7B using a two-stage training pipeline.

---

## 🎯 Project Overview

This project demonstrates **architecture grafting** — a technique for extending vision-language models with new modalities by:

1. **Grafting** a frozen Whisper audio encoder onto Qwen2-VL-7B
2. **Stage 1**: Training a projector layer to align Whisper features (1280-dim) with Qwen embeddings (3584-dim)
3. **Stage 2**: QLoRA fine-tuning on instruction-following transcription data

### The "Why"

**Why graft instead of training from scratch?**
- **Leverage Pretrained Reasoning**: Inherits the multimodal capabilities of Qwen2-VL-7B.
- **Reuse Robust Audio Features**: Uses the frozen Whisper-Large-v3-Turbo encoder (1.5B params).
- **Compute Efficiency**: Achieves SOTA results with minimal compute (~18 hours total on single A100/RTX 6000).
- **Parameter Efficient**: Trains only 4.6M projector params + Rank-64 LoRA adapters.


---

## 📊 Performance Highlights

**Evaluation Context**: Tested on a held-out subset of 200 samples (English Parliamentary/Political speech).

| Metric | This Model | Comparative Baseline* |
|--------|------------|-----------------------|
| **Word Error Rate (WER)** | **3.6%** | ~5-8% (Typical Zero-Shot) |
| **Character Error Rate (CER)** | **2.5%** | 3-5% |
| **Training Samples** | **20,000** | Millions (for Foundation Models) |
| **Compute Efficiency** | **~18 GPU-hours** | Hundreds/Thousands |

*\*Baseline refers to typical performance of similarly sized models without domain-specific fine-tuning.*

**Qualitative Finding: Label Noise Robustness**
During error analysis, we observed that the model frequently **corrected ground-truth label errors** (e.g., fixing typos or missing articles present in the training transcripts). This suggests the model has learned robust phonetic mapping rather than just memorizing the dataset noise.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  Whisper-Large-v3-Turbo Encoder (FROZEN)        │
│  1.5B params → 1280-dim audio features          │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│  Audio Projector (Linear Layer)                 │
│  1280 → 3584 dims (4.6M trainable params)       │
└────────────────┬────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────┐
│  Qwen2-VL-7B LLM (QLoRA Fine-tuned)             │
│  7B params with rank-64 LoRA adapters           │
└─────────────────────────────────────────────────┘
```

![Architecture Diagram](figures/architecture_diagram.png)

## 📈 Training Dynamics
| **Stage 1: Projector Alignment** | **Stage 2: QLoRA Fine-Tuning** |
| :---: | :---: |
| **Train Loss** (Learning)<br>![Stage 1 Train](figures/stage1_train_loss.png) | **Train Loss** (Learning)<br>![Stage 2 Train](figures/stage2_train_loss.png) |
| **Eval Loss** (Generalization)<br>![Stage 1 Eval](figures/stage1_eval_loss.png) | **Eval Loss** (Generalization)<br>![Stage 2 Eval](figures/stage2_eval_loss.png) |
---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/kulsoom-abdullah/Qwen2-VL-Audio-Graft
cd Qwen2-VL-Audio-Graft

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install the custom Transformers fork
# (Required: Contains the modified Qwen2VL source code for Audio Grafting)
pip install -e transformers_fork/

```

### Inference

**Option A: Command Line (Fast)**
Transcribe an audio file using the custom script:

```bash
python scripts/inference.py --audio your_audio.wav

```

**Option B: Interactive Notebook (Exploratory)**
For deep analysis, visualization, and error checking:
**`notebooks/02_Test_Custom_Audio.ipynb`**


This notebook handles:

1. Loading the Base Model + Stage 2 Adapters.
2. Resampling your audio to 16kHz.
3. Running the generation pipeline with visualization.

**Command Line Evaluation (Test Set):**
To reproduce the WER/CER metrics on the full test dataset:

```bash
python scripts/05_evaluate.py

```

---

## 📁 Project Structure

```text
Qwen2-VL-Audio-Graft/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── scripts/                       # End-to-end training pipeline
│   ├── 00_graft_architecture.py   # Initial model grafting
│   ├── 01_prepare_data.py         # Dataset preparation
│   ├── 02_train_stage1.py         # Stage 1: Projector training
│   ├── 03_check_stage1.py         # Stage 1 validation
│   ├── 04_train_stage2.py         # Stage 2: QLoRA fine-tuning
│   ├── 05_evaluate.py             # Final evaluation script
│   ├── inference.py               # CLI inference script
│   ├── upload_to_hf.py            # HuggingFace upload utility
│   └── verify_hf_model.py         # Model verification utility
│
├── notebooks/                     # Interactive demos & analysis
│   ├── 00_Project_Walkthrough_and_Concepts.ipynb  # Deep dive into methodology
│   ├── 01_View_Results_Highlighted.ipynb          # Error analysis with audio playback
│   └── 02_Test_Custom_Audio.ipynb                 # Test with your own recordings
│
├── tests/                         # Validation & sanity checks
│   ├── test_audio_sensitivity.py  # Verify audio affects output
│   ├── test_generate_fix.py       # Validate .generate() works
│   ├── test_nano_inference.py     # Quick inference test
│   ├── test_processing_qwen2_vl.py # Verify processor logic
│   └── test_vision_process_local.py # Regression test for vision capabilities
│
├── docs/                          # Technical documentation
│   ├── architecture_deep_dive.md  # Theory behind the grafting
│   ├── engineering_log.md         # Daily dev log & decisions
│   ├── MODELING_QWEN2_VL_CHANGES.md      # Modifications to modeling_qwen2_vl.py
│   ├── PROCESSING_QWEN2_VL_CHANGES.md    # Modifications to processing_qwen2_vl.py
│   ├── VISION_PROCESS_CHANGES.md         # Modifications to image handling
│   └── QWEN_VERSION_COMPARISON.md        # Diff against upstream Qwen
│
├── figures/                       # Architecture diagrams & plots
├── transformers_fork/             # Modified HuggingFace transformers source
└── output/                        # Training artifacts & results
```
---
## Notebook Descriptions

1. **`00_Project_Walkthrough_and_Concepts.ipynb`**
   - End-to-end project narrative
   - Dataset EDA and statistics
   - Whisper architecture analysis (30s truncation proof)
   - Teacher forcing vs autoregressive training
   - Weight comparison between stages

2. **`01_View_Results_Highlighted.ipynb`**
   - Word-by-word diff analysis with audio playback
   - Error pattern analysis
   - Label noise discovery (model corrects dataset errors!)
   - Performance metrics visualization

3. **`02_Test_Custom_Audio.ipynb`**
   - Record audio on Mac (Voice Memos)
   - Audio preprocessing (16kHz resampling)
   - Full inference pipeline
   - Test with different instructions

---

## 🔬 Tech Stack

- **Framework**: PyTorch 2.4+
- **Transformers**: Custom fork of HuggingFace Transformers 4.45.2
  - Modified `Qwen2VLForConditionalGeneration` for audio grafting
  - Updated `prepare_inputs_for_generation()` for `.generate()` support
  - Custom processor for audio token handling
- **PEFT**: 4-bit QLoRA (rank-64, alpha-128) for memory-efficient fine-tuning
- **Audio**: Librosa + torchaudio + Whisper feature extraction
- **Monitoring**: Weights & Biases for experiment tracking
- **Hardware**: NVIDIA A100 40GB and RTX 6000 Ada (Lambda Labs / RunPod)

---


## 🧪 Training Details

### Hardware & Environment
- **Hardware**: NVIDIA RTX 6000 Ada / A100 40GB (Lambda Labs/RunPod)
- **Framework**: PyTorch 2.4+, Transformers (Custom Fork), PEFT
- **Precision**: BFloat16 with Flash Attention 2

### Stage 1: Audio Projector Alignment
**Objective**: Align Whisper audio features with Qwen2-VL embedding space.

| Parameter | Value |
|-----------|-------|
| **Trainable** | Audio Projector only (4.6M params) |
| **Frozen** | Audio Encoder + LLM (~8.5B params) |
| **Dataset** | ~20,000 samples (SpeechBrain ASR) |
| **Duration** | ~12 hours |
| **Final Loss** | ~0.30 (Converged) |
| **Optimizer** | AdamW (lr=1e-3, High LR for initialization) |

### Stage 2: QLoRA Fine-Tuning
**Objective**: Teach the model to follow transcription instructions.

| Parameter | Value |
|-----------|-------|
| **Trainable** | Projector + LLM Adapters (Rank-64) |
| **Frozen** | Audio Encoder |
| **Dataset** | ~20,000 samples |
| **Duration** | ~6 hours |
| **Final Loss** | ~0.06 (Highly accurate) |
| **Technique** | 4-bit QLoRA (BitsAndBytes) |



**Critical Implementation Details:**
- **Regex-based LoRA Targeting**: Specifically targets Qwen2 attention (`q_proj`, `v_proj`, etc.) and MLP layers while strictly avoiding the frozen Whisper encoder to prevent catastrophic forgetting.
  ```python
  target_modules_regex=r"model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"

```

* **Custom `.generate()` Support**: Patched `transformers` source code (3 specific edits) to pass audio features through the KV-cache generation loop.
* **Adapter Merging**: LoRA weights were merged into the base model before saving to ensure inference stability without complex adapter loading logic.

---

## 📈 Evaluation Results

Tested on a held-out subset of 50 samples from the SpeechBrain test set:

| Metric | Result |
| :--- | :--- |
| **Total Samples** | 50 |
| **Exact Matches** | 30 (60.0%) |
| **Partial Matches** | 4 (8.0%) |
| **Mismatches** | 16 (32.0%) |
| **WER** | **3.6%** |
| **CER** | **2.5%** |

**Qualitative Finding: Label Noise**
During error analysis, we observed that "mismatches" often represent **higher accuracy than the ground truth labels**. The model frequently corrects human transcription errors (e.g., fixing typos, adding missing articles, or correcting "inter american" to "interamerican" based on the audio).


---

## 🔧 Custom Transformers Fork

**Why a fork?** The standard Qwen2-VL architecture does not natively support audio embedding injection.

**Key Source Code Modifications** (see `docs/MODELING_QWEN2_VL_CHANGES.md`):

1. **`__init__()`**: Initialized the Whisper-Large-v3-Turbo encoder and the trainable Linear Projector (1280 → 3584 dim).
2. **`forward()`**: Modified the forward pass to intercept `input_features`, project them, and swap them into the token sequence at `<|audio_pad|>` indices.
3. **`.generate()`**: Patched `prepare_inputs_for_generation` to accept audio features and handle caching correctly during autoregressive decoding.

---
## 🎓 Key Learnings

1.  **Label Noise Detection**: At <4% WER, the model began outperforming the ground truth labels, frequently correcting typos and missing articles in the original SpeechBrain dataset.
2.  **Grafting Viability**: Proved that a simple Linear Projector is sufficient to bridge a 1.5B parameter Audio Encoder to a 7B LLM without retraining the backbones from scratch.
3.  **The Importance of LoRA Targeting**: Precise regex targeting was required to ensure LoRA adapters attached *only* to the LLM layers. Accidental training of the frozen audio encoder results in catastrophic feature collapse.

---

## 🚧 Known Limitations

-   **Context Window**: Audio input is hard-capped at ~30 seconds (1500 tokens).
-   **Domain Specificity**: High performance on formal/parliamentary speech; untested on noisy or conversational audio.
-   **Dependency**: Inference requires the custom `transformers_fork` included in this repository.

---

## 📝 Citation

```bibtex
@misc{qwen2-vl-audio-graft,
  author = {Kulsoom Abdullah},
  title = {Qwen2-VL-Audio-Graft: Architecture Grafting for Speech Understanding},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/kulsoom-abdullah/Qwen2-VL-Audio-Graft}}
}

```

---

## 🙏 Acknowledgments

* **Qwen Team** @ Alibaba Cloud for [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
* **OpenAI** for [Whisper-Large-v3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
* **SpeechBrain** for [LargeScaleASR dataset](https://huggingface.co/datasets/speechbrain/LargeScaleASR)
* **LLaVA** team for pioneering visual instruction tuning methodology

---

## 📄 License

Apache 2.0 (inherits from Qwen2-VL and Whisper)

---

## 🔗 Links

* **Trained Model**: [HuggingFace Hub](https://huggingface.co/kulsoom-abdullah/Qwen2-Audio-7B-Transcription)
* **Stage 1 Checkpoint**: [HuggingFace Hub](https://huggingface.co/kulsoom-abdullah/Qwen2-Audio-Stage1)
* **Grafted Checkpoint**: [HuggingFace Hub](https://huggingface.co/kulsoom-abdullah/qwen2-vl-audio-graft)
* **Project Report**: [LinkedIn Post](https://linkedin.com/in/kulsoom-abdullah) *(Coming Soon)*

---

**Built with 🧠 + ☕ + 🎧 by [Kulsoom Abdullah](https://linkedin.com/in/kulsoom-abdullah)

