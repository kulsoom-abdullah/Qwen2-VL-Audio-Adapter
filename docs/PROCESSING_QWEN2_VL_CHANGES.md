# Processing Qwen2-VL Modifications for Audio Support

## 📋 Summary

Modified `transformers/src/transformers/models/qwen2_vl/processing_qwen2_vl.py` to add audio handling capability alongside existing image and video support.

**File Location**: `/transformers_fork/src/transformers/models/qwen2_vl/processing_qwen2_vl.py`

**Date Modified**: December 2025

---

## 🔧 Changes Made

### 1. Added Audio Processor to Attributes (Line 58-62)

**Before**:
```python
attributes = ["image_processor", "tokenizer"]
valid_kwargs = ["chat_template"]
image_processor_class = "Qwen2VLImageProcessor"
tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
```

**After**:
```python
attributes = ["image_processor", "tokenizer", "audio_processor"]
valid_kwargs = ["chat_template"]
image_processor_class = "Qwen2VLImageProcessor"
tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")
audio_processor_class = "WhisperProcessor" # ADDED
```

**Why**:
- Register audio_processor as a processor attribute for proper serialization
- Specify which class to use for audio processing (WhisperProcessor)

---

### 2. Updated `__init__` Method (Lines 63-66)

**Before**:
```python
def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
    self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
    self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
    super().__init__(image_processor, tokenizer, chat_template=chat_template)
```

**After**:
```python
def __init__(self, image_processor=None, tokenizer=None, audio_processor=None, chat_template=None, **kwargs):
    self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
    self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
    self.audio_token = "<|audio_pad|>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token #ADDED
    super().__init__(image_processor, tokenizer, audio_processor, chat_template=chat_template)
```

**Changes**:
- Added `audio_processor` parameter
- Added `self.audio_token` initialization
- Pass `audio_processor` to parent `__init__`

---

### 3. Updated `__call__` Signature (Lines 68-73)

**Before**:
```python
def __call__(
    self,
    images: ImageInput = None,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    videos: VideoInput = None,
    **kwargs: Unpack[Qwen2VLProcessorKwargs],
) -> BatchFeature:
```

**After**:
```python
def __call__(
    self,
    images: ImageInput = None,
    text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
    videos: VideoInput = None,
    audios = None,  # np.ndarray or List[np.ndarray] ADDED
    **kwargs: Unpack[Qwen2VLProcessorKwargs],
) -> BatchFeature:
```

**Why**: Accept audio inputs alongside images/videos.

---

### 4. Added Audio Processing Section (After line 128)

**Insert after the video processing block**:

```python
# Process audios with WhisperProcessor if provided
if audios is not None and self.audio_processor is not None:
    if not isinstance(audios, list):
        audios = [audios]

    # Process each audio with Whisper feature extractor
    audio_inputs = self.audio_processor(
        audios,
        sampling_rate=16000,
        return_tensors="pt"
    )
else:
    audio_inputs = {}
```

**Purpose**:
- Convert numpy audio arrays to Whisper input features (spectrograms)
- Handles both single audio and list of audios
- Uses 16kHz sampling rate (Whisper requirement)

---

### 5. Added Audio Token Expansion (After line 153, before tokenization)

**Insert after video token expansion block**:

```python
# Expand audio tokens (fixed 1500 tokens per audio)
if audios is not None:
    FIXED_AUDIO_LENGTH = 1500  # Whisper encoder output is always 1500 tokens
    audio_index = 0
    for i in range(len(text)):
        while self.audio_token in text[i]:
            # Replace single <|audio_pad|> with 1500 <|audio_pad|> tokens
            text[i] = text[i].replace(
                self.audio_token,
                "<|placeholder|>" * FIXED_AUDIO_LENGTH,
                1  # Replace only first occurrence
            )
            audio_index += 1
        text[i] = text[i].replace("<|placeholder|>", self.audio_token)
```

**Key Design Decision**:
- **Fixed length**: Always 1500 tokens (no calculation needed, unlike image/video grid_thw)
- **Placeholder trick**: Prevents double-replacement if text contains multiple audio tokens
- **Index tracking**: Ensures each audio placeholder maps to correct audio input

---

### 6. Updated Return Statement (Line 157)

**Before**:
```python
return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})
```

**After**:
```python
return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs, **audio_inputs})
```

**Why**: Include audio features in the output batch.

---

## 🎯 How Token Expansion Works

### Image/Video (Dynamic)
```python
# Calculate tokens based on grid dimensions
num_tokens = grid_thw.prod() // merge_length
text = text.replace("<|image_pad|>", "<|image_pad|>" * num_tokens)
```

### Audio (Fixed)
```python
# Always 1500 tokens
FIXED_AUDIO_LENGTH = 1500
text = text.replace("<|audio_pad|>", "<|audio_pad|>" * FIXED_AUDIO_LENGTH)
```

**Why the difference?**
- **Images/Videos**: Variable resolution → variable token count
- **Audio**: Fixed 30s max → fixed 1500 tokens (Whisper constraint)

---

## 🔗 Data Flow

```
1. User provides: images, text, videos, audios
   ↓
2. Process images → pixel_values, image_grid_thw
   ↓
3. Process videos → pixel_values_videos, video_grid_thw
   ↓
4. Process audios → input_features (Whisper spectrograms)
   ↓
5. Expand image tokens in text (dynamic count)
   ↓
6. Expand video tokens in text (dynamic count)
   ↓
7. Expand audio tokens in text (fixed 1500)
   ↓
8. Tokenize text → input_ids, attention_mask
   ↓
9. Return BatchFeature with all modalities
```

---

## 📚 Dependencies

**New Dependency**: WhisperProcessor
- Used for: Converting audio → spectrogram features
- Import: `from transformers import WhisperProcessor`
- Note: Must be passed to `Qwen2VLProcessor.__init__`

**Existing Dependencies** (unchanged):
- Qwen2VLImageProcessor
- Qwen2TokenizerFast

---

## 🧪 Testing Strategy

### Unit Tests to Create:

1. **Test audio token initialization**:
   ```python
   processor = Qwen2VLProcessor(image_processor, tokenizer, audio_processor)
   assert processor.audio_token == "<|audio_pad|>"
   ```

2. **Test audio processing**:
   ```python
   audio = np.random.randn(80000).astype(np.float32)  # 5s audio
   result = processor(audios=[audio], text="<|audio_pad|> Transcribe.")
   assert "input_features" in result  # Whisper output
   ```

3. **Test token expansion**:
   ```python
   text = "Audio: <|audio_pad|> End."
   result = processor(audios=[audio], text=text)
   # Count tokens - should have 1500 audio_pad tokens
   token_ids = result["input_ids"][0]
   audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
   audio_token_count = (token_ids == audio_pad_id).sum()
   assert audio_token_count == 1500
   ```

4. **Test multi-modal**:
   ```python
   result = processor(
       images=image,
       videos=video,
       audios=audio,
       text="<|image_pad|> <|video_pad|> <|audio_pad|> Describe."
   )
   assert all(k in result for k in ["pixel_values", "pixel_values_videos", "input_features"])
   ```

---

## 🚨 Important Notes

### Token Expansion Order

**Critical**: Audio expansion must happen AFTER image/video expansion but BEFORE tokenization:

```python
# 1. Process modalities
image_inputs = process_images(images)
video_inputs = process_videos(videos)
audio_inputs = process_audios(audios)  # ← New

# 2. Expand tokens IN ORDER
expand_image_tokens(text, image_grid_thw)
expand_video_tokens(text, video_grid_thw)
expand_audio_tokens(text)  # ← New (always 1500)

# 3. Tokenize (AFTER all expansions)
text_inputs = tokenizer(text)
```

**Why**: Tokenizer needs to see the fully expanded text with all placeholder tokens.


---

## 🎓 Key Learnings

1. **Consistency**: Follow the same pattern as image/video processing
2. **Fixed vs Dynamic**: Audio is simpler (no grid calculations)
3. **Placeholder Trick**: Elegant solution to prevent double-replacement
4. **Order Matters**: Process → Expand → Tokenize (strict sequence)
5. **Separation of Concerns**:
   - Processor = Preprocessing (token expansion)
   - Model = Inference (audio encoder + projection)

---

**Status**: Ready for implementation
**Testing**: Create test script before installation (same as vision_process.py)
