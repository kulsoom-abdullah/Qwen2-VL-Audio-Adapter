# Vision Process Modifications for Audio Support

## 📋 Summary

Modified `qwen_vl_utils/vision_process.py` to add audio handling capability alongside existing image and video support.

**File Location**:
- Original: `/opt/homebrew/Caskroom/miniforge/base/envs/qwen_audio_project/lib/python3.10/site-packages/qwen_vl_utils/vision_process.py`
- Backup: `vision_process.py.backup`

**Date Modified**: December 2025

---

## 🔧 Changes Made

### 1. Added Import (Line ~8)

```python
import librosa  # For audio loading and resampling
```

**Why**: Need librosa to load audio from bytes/files/URLs and resample to 16kHz for Whisper.

---

### 2. Added `fetch_audio` Function (After `fetch_image`, ~Line 138)

```python
def fetch_audio(ele: Dict[str, Union[str, bytes]], sr: int = 16000) -> np.ndarray:
    """
    Loads audio from the dict element (path, url, or bytes) and returns raw array.

    Handles:
    - Bytes (from streaming datasets)
    - File paths (local .wav files)
    - URLs (remote audio)
    - Already loaded numpy arrays

    Returns:
        np.ndarray: Audio samples at specified sample rate (default 16kHz for Whisper)
    """
```

**Purpose**:
- Load audio from multiple sources (bytes, file, URL)
- Resample to 16kHz (Whisper requirement)
- Return numpy array for processing

**Key Design Decision**: Made `sr` a parameter for flexibility, though Whisper requires 16kHz.

---

### 3. Updated `extract_vision_info` Function (~Line 408)

**Before**:
```python
if (
    "image" in ele
    or "image_url" in ele
    or "video" in ele
    or ele.get("type", "text") in ("image", "image_url", "video")
):
```

**After**:
```python
if (
    "image" in ele
    or "image_url" in ele
    or "video" in ele
    or "audio" in ele         # ← Added
    or "audio_url" in ele     # ← Added
    or ele.get("type", "text") in ("image", "image_url", "video", "audio")  # ← Added "audio"
):
```

**Purpose**: Detect audio elements in conversation content.

---

### 4. Updated `process_vision_info` Function (~Line 427)

**Changes**:

**A. Initialize audio list**:
```python
image_inputs = []
video_inputs = []
audio_inputs = []  # ← Added
```

**B. Process audio elements**:
```python
elif "audio" in vision_info or "audio_url" in vision_info:
    audio_inputs.append(fetch_audio(vision_info, sr=16000))
```

**C. Handle empty case**:
```python
if len(audio_inputs) == 0: audio_inputs = None
```

**D. Updated return statements** (~Line 592-593):
```python
# Before:
return image_inputs, video_inputs, video_kwargs
return image_inputs, video_inputs

# After:
return image_inputs, video_inputs, audio_inputs, video_kwargs
return image_inputs, video_inputs, audio_inputs
```

**Purpose**:
- Process audio alongside images/videos
- Maintain consistent API: `(images, videos, audios, [kwargs])`

---

## 🧪 Testing

**Test Suite**: `test_vision_process_local.py`

**Tests Performed**:
1. ✅ Unit test: `fetch_audio` with bytes
2. ✅ Unit test: `extract_vision_info` detects audio
3. ✅ Integration test: `process_vision_info` full pipeline
4. ✅ Validation: Audio properties (dtype, range, duration)

**Sample Test Result**:
```
Audio shape: (273920,)
Audio dtype: float32
Duration: 17.12 seconds
Values in range: [-1.0, 1.0] ✅
```

---

## 🎯 API Changes

### Function Signature Change

**`process_vision_info`**:

**Before**:
```python
def process_vision_info(messages) -> Tuple[List, List]:
    # Returns: (image_inputs, video_inputs)
```

**After**:
```python
def process_vision_info(messages) -> Tuple[List, List, List]:
    # Returns: (image_inputs, video_inputs, audio_inputs)

    # Or with kwargs:
    # Returns: (image_inputs, video_inputs, audio_inputs, video_kwargs)
```

**Breaking Change**: Yes - adds a third return value.

**Impact**: Any code calling `process_vision_info` must be updated to handle 3 (or 4) return values.

---

## 🔗 Integration with Audio Grafting Pipeline

### Data Flow

```
Dataset Sample (bytes)
       ↓
format_data() → Conversation format
       ↓
extract_vision_info() → Detect audio element
       ↓
process_vision_info() → Call fetch_audio()
       ↓
fetch_audio() → Load with librosa @ 16kHz
       ↓
numpy array (N,) at 16kHz
       ↓
[Next: Whisper Processor converts to spectrogram]
```

---

## 📚 Dependencies

**New Dependency**: `librosa`
- Used for: Audio loading and resampling
- Install: `pip install librosa`

**Existing Dependencies** (unchanged):
- `numpy`, `requests`, `io`, `os`

---

## 🚨 Important Notes

### Sample Rate Requirement

**Critical**: Whisper requires **16kHz** audio. The `sr=16000` parameter in `fetch_audio` is **not optional**.

**What librosa does**:
- Input already 16kHz → Just loads
- Input ≠ 16kHz → **Resamples** automatically

### Return Order Logic

**Design Decision**: Data before metadata
```python
return image_inputs, video_inputs, audio_inputs, video_kwargs
       └─────────── Data ──────────┘            └─ Metadata ─┘
```

**Why**: Keeps tensors grouped together, metadata at the end.

---

## 🔍 Verification

### Quick Test in Notebook

```python
from qwen_vl_utils import process_vision_info

# Test with your formatted sample
formatted_sample = format_data(sample)
image_inputs, video_inputs, audio_inputs = process_vision_info(formatted_sample)

print(f"Audio inputs: {len(audio_inputs)}")
print(f"Audio shape: {audio_inputs[0].shape}")
print(f"Duration: {len(audio_inputs[0])/16000:.2f}s")
```

**Expected Output**:
```
Audio inputs: 1
Audio shape: (273920,)
Duration: 17.12s
```

---

## 🎓 Key Learnings

1. **Separation of Concerns**: Each function has one job
   - `fetch_audio` = Load audio
   - `extract_vision_info` = Detect media
   - `process_vision_info` = Orchestrate

2. **Resampling**: Always specify `sr=16000` for Whisper compatibility

3. **BytesIO Trick**: Lets librosa read bytes as if they were files

4. **API Consistency**: Return order matters for clean integration

5. **Testing Strategy**: Bottom-up (unit → integration → validation)

---

**Status**: ✅ Complete and tested
**Next Module**: `transformers/src/transformers/models/qwen2_vl/processing_qwen2_vl.py`
