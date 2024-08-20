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

We focus on five optimized methods for GPU inference:

1. [fp16 + SDPA](Scripts/GPU_INFERENCE/FP16 + SDPA.md)
2. fp16 + SDPA + Speculative Decoding
3. Distil-whisper + fp16 + SDPA + Chunking
4. fp16 + SDPA + Chunking + Speculative Decoding
5. Whisper Medusa

Each method is designed to balance speed and accuracy for different use cases.

