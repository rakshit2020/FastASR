# ASR Inference Optimization Hub

This repository serves as a comprehensive collection of inference scripts for Automatic Speech Recognition (ASR) models, optimized for both GPU and CPU environments. It aims to provide developers with a centralized resource to explore and implement various ASR inference solutions tailored to their specific use cases.



## Overview

This project offers a range of inference scripts for popular ASR models, each optimized for different hardware configurations and performance requirements. Whether you're looking for high-speed CPU inference or GPU-accelerated processing, you'll find implementations suited to your needs.

## Features

- Optimized inference scripts for both GPU and CPU
- Support for multiple ASR models and architectures
- Easy-to-use interfaces for quick integration
- Comprehensive documentation and usage examples
- Performance benchmarks to help you choose the right solution

## Supported Models

- OpenAI Whisper (Faster for GPU)
- Distil-Whisper (Faster for GPU)
- whisper.cpp models (Faster for CPU)

### About Whisper

Whisper is a state-of-the-art model for automatic speech recognition (ASR) and speech translation, proposed in the paper **Robust Speech Recognition via Large-Scale Weak Supervision** by Alec Radford et al. from OpenAI. Trained on >5M hours of labeled data, Whisper demonstrates a strong ability to generalise to many datasets and domains in a zero-shot setting.

In our scripts, we primarily use openai/whisper-large-v3 and distil-whisper large. You can modify the scripts to use small and medium models for faster inference speed.

## Optimized Inference Methods

### We focus on five optimized methods for GPU inference:

1. [fp16 + SDPA](Scripts/GPU_INFERENCE/FP16+SDPA.md)
2. [fp16 + SDPA + Speculative Decoding](Scripts/GPU_INFERENCE/fp16+SDPA+SpeculativeDecoding.md)
3. [Distil-whisper + fp16 + SDPA + Chunking](Scripts/GPU_INFERENCE/Distil-whisper+fp16+SDPA+Chunking.md)
4. [fp16 + SDPA + Chunking + Speculative Decoding](Scripts/GPU_INFERENCE/fp16+SDPA+Chunking+SpeculativeDecoding.md)
5. [Whisper Medusa](Scripts/GPU_INFERENCE/Whisper_Medusa.md)

Each method is designed to balance speed and accuracy for different use cases.

### We focus on two optimized methods for CPU inference:

1. [Faster-Whisper](Scripts/CPU_INFERENCE/Faster-Whisper.md)
2. [Whisper.cpp](Scripts/CPU_INFERENCE/WhisperCPP.md)

## Support and Contributions

If you find this project helpful or interesting, please consider giving it a star ⭐️ on GitHub. Your support helps make this resource more visible to other developers who might benefit from it.

We're always looking to improve and expand our collection of ASR inference optimization techniques. If you have experience with other optimized methods for ASR inference or have developed your own optimizations, we'd love to hear from you! Feel free to open an issue to discuss new ideas or submit a pull request with your contributions. Whether it's a new optimization technique, an improvement to existing scripts, or documentation enhancements, your input is valuable to the community.

Together, we can make this repository an even more comprehensive resource for ASR developers. Thank you for your interest and support!
