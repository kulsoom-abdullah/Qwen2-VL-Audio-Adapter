# 🧠 Architecture Deep Dive: The Qwen2-Audio Graft

To enable audio understanding in Qwen2-VL, we modified the Hugging Face `transformers` library. This "Grafting" process involved three critical changes to the model architecture.

## 1. The Input Injection (`modeling_qwen2_vl.py`)
Standard Qwen2-VL has a vision tower for images. We bypassed this for audio.
* **Original**: `Inputs -> Vision Transformer -> Projector -> LLM`
* **Modified**: `Audio -> Whisper Encoder -> Linear Projector -> LLM`

We modified the `forward()` method to accept `input_features` (Audio) alongside `pixel_values` (Images). The audio features are passed through the frozen Whisper Encoder and then projected to match Qwen's hidden size (3584).

## 2. The Processor (`processing_qwen2_vl.py`)
Instead of modifying the tokenizer class directly (which can be fragile), we updated the **Processor** to handle audio token expansion dynamically.
* **Logic**:
    1.  Detects audio inputs.
    2.  Expands the `<|audio_pad|>` token into **1,500** fixed tokens (matching the Whisper encoder output).
    3.  Ensures the tokenizer sees the fully expanded sequence before encoding.

## 3. The Vision/Audio Process (`vision_process.py`)
We extended the utility functions to support audio loading.
* **New Function**: `fetch_audio(ele, sr=16000)`
* **Capability**: Handles Bytes (streaming), File Paths, and URLs.
* **Resampling**: Enforces 16kHz sample rate to match Whisper requirements.

## 🧬 Why Fork?
We used a local fork (`transformers_fork`) instead of monkey-patching to ensure:
1.  **Reproducibility**: The exact model code is versioned with the weights.
2.  **Stability**: Updates to the upstream `transformers` library won't break our custom architecture.
3.  **Exportability**: The model can be loaded with `trust_remote_code=True`.