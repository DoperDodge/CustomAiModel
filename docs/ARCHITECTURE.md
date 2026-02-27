# Custom Multimodal AI System — Architecture Design Document

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Module Design: Text-to-Text](#3-module-design-text-to-text)
4. [Module Design: Text-to-Speech](#4-module-design-text-to-speech)
5. [Module Design: Speech-to-Speech](#5-module-design-speech-to-speech)
6. [Module Design: Image Generation](#6-module-design-image-generation)
7. [Training Setup & Hardware](#7-training-setup--hardware)
8. [Datasets](#8-datasets)
9. [Customization & Fine-Tuning](#9-customization--fine-tuning)
10. [Integration Architecture](#10-integration-architecture)
11. [Resource Efficiency](#11-resource-efficiency)
12. [Ethical & Safety Guidance](#12-ethical--safety-guidance)
13. [Implementation Roadmap](#13-implementation-roadmap)

---

## 1. Executive Summary

This document describes the architecture for a **modular multimodal AI system** that
supports four core capabilities:

| Capability         | Input  | Output | Core Technique                    |
|--------------------|--------|--------|-----------------------------------|
| Text-to-Text       | Text   | Text   | Decoder-only Transformer (LLM)   |
| Text-to-Speech     | Text   | Audio  | VITS / SoundStorm vocoder         |
| Speech-to-Speech   | Audio  | Audio  | ASR → LLM → TTS pipeline + streaming |
| Image Generation   | Text   | Image  | Latent Diffusion (Stable Diffusion) |

**Design philosophy:** Start modular, unify later. Each capability is implemented
as a standalone module behind a shared API. A central **Orchestrator** routes
requests to the right module. This lets you build, test, and improve each piece
independently before (optionally) merging weights into a unified multimodal backbone.

### Why Modular First?

- **Simpler debugging** — isolate failures to one module.
- **Incremental training** — train each module on its own data/GPU budget.
- **Swap-friendly** — replace any module with a better one without retraining everything.
- **Lower compute floor** — load only the modules you need at inference time.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Unified REST / gRPC API                  │
│                        (FastAPI Gateway)                     │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
        ┌──────▼──────┐ ┌────▼────┐  ┌──────▼──────┐
        │ Orchestrator │ │  Auth   │  │  Rate Limit │
        │  (Router)    │ │ Module  │  │  & Queue    │
        └──────┬───────┘ └─────────┘  └─────────────┘
               │
   ┌───────────┼────────────┬──────────────┐
   │           │            │              │
┌──▼──┐   ┌───▼───┐   ┌────▼───┐   ┌──────▼──────┐
│ T2T │   │  TTS  │   │  S2S   │   │  Image Gen  │
│ LLM │   │ VITS  │   │Pipeline│   │  Diffusion  │
└─────┘   └───────┘   └────────┘   └─────────────┘
   │           │            │              │
   └───────────┴────────────┴──────────────┘
               │
        ┌──────▼──────┐
        │ Shared Utils │
        │ (tokenizers, │
        │  caching,    │
        │  safety)     │
        └─────────────┘
```

### Component Summary

| Component       | Technology                  | Parameters (target) | Notes                          |
|-----------------|-----------------------------|---------------------|--------------------------------|
| Text-to-Text    | LLaMA-style transformer     | 1.3B → 7B           | Start small, scale up          |
| Text-to-Speech  | VITS2 / Piper               | ~80M                | Single-stage end-to-end        |
| Speech-to-Speech| Whisper + LLM + TTS         | Composite            | Pipeline; upgrade to streaming |
| Image Generation| Stable Diffusion (LDM)      | ~860M UNet + CLIP   | Use existing SD checkpoints    |
| Orchestrator    | Python + FastAPI            | —                   | Thin routing layer             |

---

## 3. Module Design: Text-to-Text

### 3.1 Architecture Choice: Decoder-Only Transformer

A **decoder-only transformer** (GPT / LLaMA family) is the proven architecture for
conversational and generative text. Reasons:

- Autoregressive generation produces fluent, coherent text.
- Massive open-source ecosystem (LLaMA, Mistral, Pythia, GPT-NeoX).
- Can be fine-tuned for chat, summarization, Q&A with the same base weights.

### 3.2 Model Specification

```
Model: LLaMA-style decoder-only transformer
Sizes:  Phase 1 → 1.3B params (fits on a single 24GB GPU)
        Phase 2 → 7B params   (needs multi-GPU or quantization)

Hidden dim:       2048 (1.3B) / 4096 (7B)
Layers:           24   (1.3B) / 32   (7B)
Attention heads:  16   (1.3B) / 32   (7B)
Context length:   4096 tokens (expandable via RoPE scaling)
Vocabulary:       32,000 (SentencePiece / BPE)
Positional enc:   Rotary Position Embeddings (RoPE)
Normalization:    RMSNorm (pre-norm)
Activation:       SiLU (SwiGLU FFN)
```

### 3.3 Why These Choices?

- **RoPE** allows extending context length after training without retraining.
- **RMSNorm** is cheaper than LayerNorm and stabilizes training at scale.
- **SwiGLU FFN** gives ~1-2% quality improvement over standard ReLU FFN.
- **Grouped-Query Attention (GQA)** at 7B reduces KV-cache memory by ~4x during inference.

### 3.4 Training Strategy

| Phase         | Data                          | Method               | Compute       |
|---------------|-------------------------------|----------------------|---------------|
| Pre-training  | RedPajama / SlimPajama        | Next-token prediction| 1-4 GPUs, weeks|
| SFT           | OpenAssistant, Dolly, Alpaca  | Supervised fine-tune | 1 GPU, hours  |
| RLHF / DPO    | UltraFeedback, HH-RLHF       | DPO (simpler)        | 1 GPU, hours  |

**Recommendation:** For Phase 1, start from a pre-trained 1.3B checkpoint (e.g., Pythia-1.4B
or TinyLlama-1.1B) and fine-tune. Pre-training from scratch is educational but expensive.

---

## 4. Module Design: Text-to-Speech

### 4.1 Architecture Choice: VITS2 (End-to-End)

**VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech)
is the best open-source single-stage TTS system:

- Single model: text → mel-spectrogram → waveform (no separate vocoder needed).
- Near human-quality on single-speaker; good multi-speaker support.
- Fast inference (~real-time on CPU, >10x real-time on GPU).

### 4.2 Model Specification

```
Model: VITS2
Components:
  - Text Encoder:      Transformer (6 layers, 192 hidden)
  - Posterior Encoder:  WaveNet-style (16 layers)
  - Decoder/Vocoder:   HiFi-GAN generator
  - Duration Predictor: Stochastic duration predictor
  - Flow:              Normalizing flow (4 coupling layers)

Parameters: ~80M total
Sample rate: 22050 Hz
Mel channels: 80
```

### 4.3 Alternative: Piper TTS

If you want an even simpler deployment path, **Piper** wraps VITS into a
production-ready package with pre-trained voices in 30+ languages. You can
fine-tune Piper voices with as little as 30 minutes of recorded audio.

### 4.4 Training Strategy

| Phase         | Data                           | Duration    |
|---------------|--------------------------------|-------------|
| Pre-train     | LJSpeech (single speaker)      | 12-24h, 1 GPU |
| Multi-speaker | VCTK (109 speakers)            | 24-48h, 1 GPU |
| Fine-tune     | Your own voice recordings      | 2-6h, 1 GPU   |

---

## 5. Module Design: Speech-to-Speech

### 5.1 Architecture: Pipeline (ASR → LLM → TTS)

Real-time S2S is the hardest modality. The practical approach is a **three-stage
pipeline** with streaming optimizations:

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Whisper  │───▶│  LLM     │───▶│  VITS    │
│  (ASR)    │    │  (T2T)   │    │  (TTS)   │
└──────────┘    └──────────┘    └──────────┘
    Audio           Text            Audio
    Input        Processing        Output

Streaming: Each stage starts output before the previous finishes.
```

### 5.2 Streaming Strategy

To achieve low latency (<2s end-to-end):

1. **Whisper streaming:** Use `faster-whisper` with voice-activity detection (VAD).
   Process audio in 1-second chunks. Emit partial transcripts.
2. **LLM streaming:** Generate tokens one at a time. Begin TTS on the first sentence
   boundary (period, question mark, newline).
3. **TTS streaming:** Feed partial text to VITS. Generate audio in chunks of ~0.5s
   and stream to the client via WebSocket.

### 5.3 Future Upgrade: Unified Speech-Language Model

Once the pipeline works, consider upgrading to a **direct speech-to-speech model**
like Mini-Omni or SpeechGPT, which encodes speech tokens directly into the LLM
vocabulary. This eliminates the ASR/TTS stages and reduces latency dramatically.

---

## 6. Module Design: Image Generation

### 6.1 Architecture Choice: Latent Diffusion Model (LDM)

**Stable Diffusion** is the open-source standard for text-to-image:

- Operates in compressed latent space (64x64) rather than pixel space (512x512).
- Uses a frozen CLIP text encoder for conditioning.
- UNet with cross-attention performs the denoising.
- VAE decodes latents back to pixels.

### 6.2 Model Specification

```
Model: Stable Diffusion v2.1 / SDXL (or train your own LDM)

Components:
  - Text Encoder: CLIP ViT-L/14 (frozen, ~400M params)
  - UNet:         ~860M params (cross-attention conditioned)
  - VAE:          ~80M params (encoder + decoder)
  - Scheduler:    DDPM / DDIM / DPM-Solver++

Image resolution:  512×512 (SD 2.1) or 1024×1024 (SDXL)
Latent resolution:  64×64 (8x compression)
Diffusion steps:    20-50 (inference), 1000 (training)
```

### 6.3 Training Strategy

| Phase              | Data                        | Compute              |
|--------------------|-----------------------------|----------------------|
| Use pre-trained    | — (start from SD checkpoint)| 0 (just download)    |
| Fine-tune (style)  | DreamBooth / LoRA on ~20 images | 1 GPU, 30 min   |
| Fine-tune (domain) | LAION-Art / your own pairs  | 1-4 GPUs, days       |
| Train from scratch | LAION-5B subset             | 8+ GPUs, weeks       |

**Recommendation:** Start from a pre-trained Stable Diffusion checkpoint. Fine-tune
with LoRA for domain-specific style. Training from scratch is only needed if you
want a fundamentally different architecture.

---

## 7. Training Setup & Hardware

### 7.1 Hardware Recommendations

| Budget Tier   | GPU                     | VRAM   | What You Can Train              |
|---------------|-------------------------|--------|---------------------------------|
| Minimum       | RTX 3090 / 4090         | 24GB   | Fine-tune all modules, train TTS |
| Recommended   | 2× A6000 or 2× RTX 4090| 48GB+  | Train 1.3B LLM, all modules    |
| Ideal         | 4× A100 80GB            | 320GB  | Train 7B LLM from scratch      |
| Cloud         | Lambda Labs / Vast.ai   | varies | Rent A100s hourly ($1-2/GPU/hr) |

### 7.2 Software Stack

```
Framework:        PyTorch 2.x (primary)
Distributed:      DeepSpeed ZeRO Stage 2/3 (multi-GPU)
Mixed precision:  bfloat16 (Ampere+) or float16
Experiment track: Weights & Biases (wandb)
Data loading:     HuggingFace Datasets (streaming mode)
Tokenizer:        SentencePiece / HuggingFace Tokenizers
Serving:          vLLM (LLM), Triton Inference Server (all)
```

### 7.3 Framework Justification

**Why PyTorch over TensorFlow?**

- Dominant in research; most open-source models are PyTorch-first.
- `torch.compile()` in PyTorch 2.x closes the performance gap with TF/XLA.
- HuggingFace ecosystem is PyTorch-native.
- DeepSpeed and FSDP provide excellent multi-GPU scaling.

---

## 8. Datasets

### 8.1 Text-to-Text

| Dataset               | Size        | Use Case                | License      |
|-----------------------|-------------|-------------------------|--------------|
| RedPajama-v2          | 30T tokens  | Pre-training            | Apache 2.0   |
| SlimPajama            | 627B tokens | Pre-training (curated)  | Apache 2.0   |
| The Pile              | 825 GB      | Pre-training            | MIT          |
| OpenAssistant (OASST) | 160K convos | Chat fine-tuning        | Apache 2.0   |
| Dolly-15K             | 15K pairs   | Instruction tuning      | CC-BY-SA     |
| UltraFeedback         | 64K pairs   | Preference/DPO          | MIT          |
| HH-RLHF              | 170K pairs  | Safety alignment        | MIT          |

### 8.2 Text-to-Speech

| Dataset       | Hours  | Speakers | Use Case           | License   |
|---------------|--------|----------|--------------------|-----------|
| LJSpeech      | 24h    | 1        | Single-speaker TTS | Public    |
| VCTK          | 44h    | 109      | Multi-speaker TTS  | CC-BY 4.0 |
| LibriTTS      | 585h   | 2,456    | Large-scale TTS    | CC-BY 4.0 |
| Common Voice  | 19K+ h | Many     | Multilingual       | CC-0      |

### 8.3 Image Generation

| Dataset         | Size       | Use Case              | License     |
|-----------------|------------|-----------------------|-------------|
| LAION-2B-en     | 2B pairs   | Pre-training LDM      | Open        |
| LAION-Aesthetics| 600M pairs | Quality-filtered      | Open        |
| COCO Captions   | 330K images| Evaluation / fine-tune| CC-BY 4.0   |
| Your own data   | 20+ images | DreamBooth / LoRA     | —           |

---

## 9. Customization & Fine-Tuning

### 9.1 Parameter-Efficient Fine-Tuning (PEFT)

For all modules, prefer **LoRA** (Low-Rank Adaptation) over full fine-tuning:

- Trains only 0.1-1% of parameters.
- Keeps base weights frozen (easy to swap adapters).
- Fits on a single consumer GPU.

```
LoRA config (typical for LLM):
  rank (r):       16-64
  alpha:          32-128
  target_modules: [q_proj, v_proj, k_proj, o_proj]
  dropout:        0.05

LoRA config (typical for Stable Diffusion):
  rank (r):       4-16
  target_modules: [to_q, to_v, to_k]  (cross-attention)
```

### 9.2 Fine-Tuning Methods by Module

| Module     | Method             | Data Needed         | Time      |
|------------|--------------------|---------------------|-----------|
| T2T (chat) | LoRA + SFT         | 1K-50K conversations| 1-4h      |
| T2T (DPO)  | LoRA + DPO         | 5K-20K preference pairs | 1-2h |
| TTS        | Full fine-tune     | 30min-2h of audio   | 2-6h      |
| TTS        | Speaker adaptation | 5-10 min of audio   | 30min     |
| Image Gen  | LoRA               | 20-200 images       | 30min-2h  |
| Image Gen  | DreamBooth         | 5-20 images         | 15-30min  |

### 9.3 Custom Data Preparation

**Text:** Format as JSONL with instruction/response pairs:
```json
{"instruction": "Summarize this article:", "input": "...", "output": "..."}
```

**Speech:** Record WAV files at 22050 Hz mono, with matching transcript files.

**Images:** Collect images + text captions as pairs. Minimum 20 for LoRA, 5 for DreamBooth.

---

## 10. Integration Architecture

### 10.1 Unified API Design

All modules are exposed through a **single FastAPI gateway**:

```
POST /v1/chat/completions      → Text-to-Text
POST /v1/audio/speech           → Text-to-Speech
POST /v1/audio/transcriptions   → Speech-to-Text (Whisper)
WS   /v1/audio/conversation     → Speech-to-Speech (WebSocket)
POST /v1/images/generations     → Image Generation
```

This mirrors the OpenAI API format for easy integration with existing tooling.

### 10.2 Orchestrator Design

The orchestrator handles:

1. **Intent detection** — Classify incoming requests by modality.
2. **Pipeline composition** — Chain modules for complex requests (e.g., "draw what I described" = ASR → Image Gen).
3. **Resource management** — Load/unload models based on demand (GPU memory).
4. **Streaming** — Manage WebSocket connections for real-time speech.

### 10.3 Multimodal Chains

```
User says: "Tell me a story and draw a picture of it"
  1. Orchestrator routes to T2T → generates story text
  2. Orchestrator routes story to Image Gen → generates illustration
  3. (Optional) Routes story to TTS → generates audiobook narration
  4. Returns: { text, image_url, audio_url }
```

---

## 11. Resource Efficiency

### 11.1 Quantization

| Technique    | Size Reduction | Quality Loss | Use Case              |
|-------------|----------------|--------------|------------------------|
| FP16/BF16   | 2×             | None         | Training + inference   |
| INT8 (LLM.int8()) | 4×      | Minimal      | Inference              |
| INT4 (GPTQ/AWQ)   | 8×      | Small        | Local deployment       |
| GGUF (llama.cpp)   | 4-8×   | Small        | CPU inference          |

**Recommendation:** Deploy LLM with AWQ 4-bit quantization (via vLLM). This lets
a 7B model run in ~4GB VRAM with <1% quality loss.

### 11.2 Model Distillation

Train a smaller "student" model to mimic the larger "teacher":

1. Generate outputs from your best (largest) model on a diverse prompt set.
2. Train a smaller model (e.g., 400M params) to match those outputs.
3. Achieves 70-90% of the teacher's quality at 5-10× faster inference.

### 11.3 Inference Optimizations

| Optimization         | Speedup | Applies To        |
|---------------------|---------|-------------------|
| KV-cache             | 2-5×    | LLM               |
| Flash Attention 2    | 2-3×    | LLM               |
| Speculative decoding | 2-3×    | LLM               |
| torch.compile()      | 1.3-2×  | All models         |
| ONNX Runtime         | 1.5-3×  | TTS, Image Gen     |
| TensorRT             | 2-5×    | All (NVIDIA GPUs)  |

### 11.4 Memory Management for Multi-Model Serving

Since loading all models simultaneously requires significant VRAM:

- **Lazy loading:** Only load a model when its endpoint is hit.
- **LRU eviction:** Unload least-recently-used models when VRAM is full.
- **CPU offloading:** Keep idle models in RAM, move to GPU on demand (~1s swap).
- **Shared backbone:** If you later unify the LLM, TTS encoder, and CLIP, share weights.

---

## 12. Ethical & Safety Guidance

### 12.1 Bias Mitigation

- **Data auditing:** Use tools like `cleanlab` to identify label noise and bias in datasets.
- **Balanced sampling:** Ensure training data covers diverse demographics, topics, cultures.
- **Evaluation:** Run bias benchmarks (BBQ, WinoBias, StereoSet) on every checkpoint.
- **Red-teaming:** Before deployment, systematically probe for harmful outputs.

### 12.2 Content Safety

```
Safety Pipeline:
  Input  → [Input Filter] → [Model] → [Output Filter] → Output
            (block harmful      ↑          (block harmful
             prompts)       (RLHF/DPO       generations)
                            alignment)
```

Implement a three-layer safety system:

1. **Input filtering:** Keyword + classifier-based blocking of harmful prompts.
2. **Model alignment:** DPO/RLHF training to make the model refuse harmful requests.
3. **Output filtering:** Post-generation classifier to catch anything the model misses.

For image generation specifically:
- Use a NSFW classifier (e.g., `safety_checker` from Stable Diffusion).
- Maintain a blocklist of concepts that should not be generated.

### 12.3 Privacy

- **No PII in training data:** Scrub names, emails, phone numbers, addresses before training.
- **Differential privacy:** Consider DP-SGD for fine-tuning on sensitive data.
- **Local deployment:** Keep inference on-premises when handling private data.
- **Data retention:** Don't log user prompts/responses in production unless needed + consented.

### 12.4 Transparency

- Document model capabilities and limitations in a **model card**.
- Publish training data sources and methodology.
- Disclose when output is AI-generated (especially for speech and images).

---

## 13. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Goal:** Get a working text chatbot.

- [ ] Set up development environment (PyTorch, CUDA, dependencies)
- [ ] Download and fine-tune a pre-trained 1.3B LLM (TinyLlama or Pythia)
- [ ] Implement chat interface with basic prompt template
- [ ] Add SFT with OpenAssistant dataset
- [ ] Basic safety filtering (input/output keyword filters)
- [ ] Deploy locally with simple FastAPI endpoint

### Phase 2: Voice (Weeks 5-8)

**Goal:** Add TTS and basic speech conversation.

- [ ] Train or fine-tune VITS on LJSpeech
- [ ] Implement TTS API endpoint
- [ ] Integrate Whisper for speech-to-text
- [ ] Build S2S pipeline (Whisper → LLM → VITS)
- [ ] Add WebSocket streaming for real-time conversation
- [ ] Test end-to-end latency, optimize bottlenecks

### Phase 3: Vision (Weeks 9-12)

**Goal:** Add image generation.

- [ ] Load pre-trained Stable Diffusion checkpoint
- [ ] Implement image generation API endpoint
- [ ] Fine-tune with LoRA for custom style (optional)
- [ ] Add NSFW safety classifier
- [ ] Integrate with orchestrator for multimodal chains

### Phase 4: Polish & Scale (Weeks 13-16)

**Goal:** Production-quality system.

- [ ] Quantize all models (AWQ for LLM, ONNX for TTS)
- [ ] Implement model memory management (lazy load, eviction)
- [ ] Add authentication and rate limiting to API
- [ ] DPO alignment training for the LLM
- [ ] Comprehensive safety evaluation and red-teaming
- [ ] Write model cards and documentation
- [ ] Dockerize the entire system

### Phase 5: Advanced (Ongoing)

**Goal:** Scale up and unify.

- [ ] Scale LLM to 7B parameters
- [ ] Explore unified multimodal architecture
- [ ] Add multi-language support
- [ ] Implement speculative decoding for faster inference
- [ ] Community feedback and iteration
