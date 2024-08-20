## FP16 + SDPA + Chunking + Speculative Decoding 

Integrating the aspect of chunking  introduces an additional layer of efficiency. Instead of processing the audio data as a continuous stream, the input is divided into smaller segments or batches.   

## Inference Script

### Usage
To use this optimization:

Install the required packages:

``` bash 
pip install torch transformers colorama flash-attn
```
We recommend using `Flash-Attention 2` if your GPU supports it.

Ensure you have a CUDA-compatible GPU for optimal performance.

Here's a Python script that demonstrates how to implement FP16 + SDPA optimization for Whisper ASR:


```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoModelForCausalLM, AutoProcessor, pipeline
import time
from colorama import Fore

class SpeechRecognitionInference:
    def __init__(self, speech_model_id, assistant_model_id, device=None, torch_dtype=None):
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype if torch_dtype else (torch.float16 if torch.cuda.is_available() else torch.float32)

        print(f"Using device: {self.device}")

        self.assistant_model = AutoModelForCausalLM.from_pretrained(
            assistant_model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
        )
        self.assistant_model.to(self.device)

        self.speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            speech_model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
        )
        self.speech_model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(speech_model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.speech_model,
            tokenizer=self.processor.tokenizer,
            generate_kwargs={"assistant_model": self.assistant_model},
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def infer(self, audio_file_path, task="translate"):
        t1 = time.time()
        result = self.pipe(audio_file_path, generate_kwargs={"task": task},batch_size=8)
        t2 = time.time()
        print(Fore.LIGHTMAGENTA_EX + f"INFERENCE TIME - {round((t2 - t1), 2)} seconds")
        return result["text"]

if __name__ == "__main__":
    assistant_model_id = "distil-whisper/distil-large-v3"
    speech_model_id = "openai/whisper-large-v3"
    sample_audio_file = r"path/to/your/audio"

    inference_engine = SpeechRecognitionInference(speech_model_id, assistant_model_id)
    transcript = inference_engine.infer(sample_audio_file)
    print(transcript)

```