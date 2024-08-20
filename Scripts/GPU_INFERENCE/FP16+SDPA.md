# FP16 + SDPA Optimization for Whisper ASR

This document explains the FP16 + SDPA (Scaled Dot Product Attention) optimization technique for Whisper ASR models, including its benefits and implementation details.

## Overview

The FP16 + SDPA optimization combines two powerful techniques to enhance the performance of Whisper ASR models:

1. **FP16 (Half-Precision)**: This reduces memory usage and computational requirements by using 16-bit floating-point numbers instead of 32-bit.
2. **SDPA (Scaled Dot Product Attention)**: Also known as Flash Attention, this is an optimized implementation of the attention mechanism.

## What is SDPA (Flash Attention)?

Scaled Dot Product Attention (SDPA), also referred to as Flash Attention 2 (FA2), is a faster and more efficient implementation of the standard attention mechanism. It can significantly speed up inference by:

1. Parallelizing the attention computation over sequence length.
2. Partitioning the work between GPU threads to reduce communication and shared memory reads/writes between them.

SDPA optimizes memory access patterns and reduces the overall computational complexity of the attention mechanism, leading to faster processing times, especially for longer sequences.

## Implementation Details

To implement FP16 + SDPA optimization:

1. Use `torch.float16` dtype when loading the model.
2. Set `attn_implementation="sdpa"` when initializing the model.
3. We recommend using `Flash-Attention 2` if your GPU supports it.
3. Ensure your PyTorch version supports SDPA (generally available in recent versions).
4. Use a CUDA-enabled GPU for maximum performance gains.

## Inference Script

### Usage
To use this optimization:

Install the required packages:

``` bash 
pip install torch transformers colorama flash-attn
```


Ensure you have a CUDA-compatible GPU for optimal performance.

Here's a Python script that demonstrates how to implement FP16 + SDPA optimization for Whisper ASR:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from colorama import Fore
import time

class ASRInference:
    def __init__(self, model_id="openai/whisper-large-v3", device=None):
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

    def run_inference(self, audio_path, task="translate"):
        start_time = time.time()
        result = self.pipe(audio_path, generate_kwargs={"task": task})
        end_time = time.time()
        print(Fore.LIGHTMAGENTA_EX + f"INFERENCE TIME - {round((end_time - start_time), 2)} seconds")
        print(result["text"])
        return result["text"]

# Example usage
if __name__ == "__main__":
    audio_path = r"/path/to/your/audio/file.m4a"
    asr_inference = ASRInference(device='cuda:0')  # Use 'cpu' if CUDA is not available
    asr_inference.run_inference(audio_path) 

```