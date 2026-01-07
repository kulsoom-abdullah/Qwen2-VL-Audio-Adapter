"""
Qwen2-Audio-Transcription Training Pipeline

This package contains the end-to-end training pipeline for grafting
Whisper audio encoder onto Qwen2-VL.

Scripts:
    00_graft_architecture.py - Initial model grafting
    01_prepare_data.py - Dataset preparation
    02_train_stage1.py - Stage 1: Projector alignment
    03_check_stage1.py - Stage 1 validation
    04_train_stage2.py - Stage 2: QLoRA fine-tuning
    05_evaluate.py - Final evaluation
    upload_to_hf.py - HuggingFace upload
    verify_hf_model.py - Model verification
"""

__version__ = "1.0.0"
__author__ = "Kulsoom Abdullah"
