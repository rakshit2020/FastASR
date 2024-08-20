## Distil- whisper + FP16 + SDPA + Chunking 

Distil-Whisper represents a streamlined variant of the original Whisper model, designed to enhance performance without sacrificing accuracy. It achieves a remarkable speed increase, operating six times faster than its predecessor. Additionally, Distil-Whisper boasts a significant reduction in size, being 49% smaller, which contributes to its efficiency. Despite these optimizations, it maintains a high level of accuracy, performing with a word error rate (WER) that is within just 1% of the original model. 

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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
from colorama import Fore

class ASRInference:
    def __init__(self, model_id="distil-whisper/distil-large-v3", device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True, 
            attn_implementation="sdpa"
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def run_inference(self, audio_path, task="translate", batch_size=8):
        start_time = time.time()
        result = self.pipe(audio_path, generate_kwargs={"task": task}, batch_size=batch_size)
        end_time = time.time()

        print(Fore.LIGHTMAGENTA_EX + f"INFERENCE TIME - {round((end_time - start_time), 2)} seconds")
        print(result["text"])
        return result["text"]

# Example usage
if __name__ == "__main__":
    audio_path = r"path/to/audio"
    asr_inference = ASRInference(device='cuda:0')  # You can change the device as needed
    asr_inference.run_inference(audio_path)

```