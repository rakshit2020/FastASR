## Faster Whisper transcription with CTranslate2

Faster-whisper is a reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models.

This implementation is up to 4 times faster than openai/whisper for the same accuracy while using less memory. The efficiency can be further improved with 8-bit quantization on both CPU and GPU.
## Inference Script

```bash
pip install faster-whisper
```
```python
from faster_whisper import WhisperModel

class FasterWhisperInference:
    def __init__(self, model_size="large-v3", device="cpu", compute_type="float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        # or run on GPU with INT8
        # self.model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
        # or run on CPU with INT8
        # self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        
    def run_inference(self, audio_path, beam_size=5):
        segments, info = self.model.transcribe(audio_path, beam_size=beam_size)
        
        print(f"Detected language '{info.language}' with probability {info.language_probability:.6f}")
        
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        return segments, info

# Example usage
if __name__ == "__main__":
    audio_path = "audio.mp3"
    asr_inference = FasterWhisperInference(device="cpu", compute_type="float16")  # You can change compute_type and device as needed
    asr_inference.run_inference(audio_path)

```

### Faster Distil-Whisper

```python
from faster_whisper import WhisperModel

model_size = "distil-large-v3"

model = WhisperModel(model_size, device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.mp3", beam_size=5, language="en", condition_on_previous_text=False)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```