# Mental Health Chatbot Fine-Tuning with QLoRA

## Overview

This project demonstrates how to fine-tune the **Meta LLaMA 3.2-3B-Instruct** model for building a mental health support chatbot using QLoRA (Quantized Low-Rank Adaptation). The notebook handles model loading, quantization, tokenizer setup, dataset preparation, and prompt formatting.

The goal is to create a conversational assistant that can respond with empathy and psychological relevance in mental health contexts.

The fine-tuning pipeline uses:
- **Transformers** from Hugging Face for model architecture,
- **BitsAndBytes** for 4-bit quantization,
- **PEFT** for parameter-efficient training,
- **Datasets** with Parquet loading for conversational data,
- **Accelerate** and `trl` for scalable training utilities.

---

## Features

- 🤖 **LLaMA 3.2-3B Model** – Uses a state-of-the-art causal language model
- 💡 **QLoRA (4-bit) Training** – Efficient and memory-friendly fine-tuning
- 🧠 **Mental Health Prompt Design** – System prompt aligned with therapeutic support
- 🧵 **Formatted Conversational Dataset** – Preprocessing using special instruction tokens
- ⚙️ **PEFT + LoRA** – Low-rank adaptation configuration for lightweight training

---

## Prerequisites

- Python 3.8+
- Access to Hugging Face Hub
- GPU with support for 4-bit quantized models (recommended)
- Installed libraries:
  - `transformers`, `datasets`, `peft`, `bitsandbytes`, `accelerate`, `trl`, `ipywidgets`

---
