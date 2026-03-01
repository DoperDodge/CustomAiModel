# Custom Multimodal AI Model

A modular, open-source multimodal AI system supporting **Text, Speech, and Image** generation.

## Capabilities

| Module | Input → Output | Architecture |
|--------|---------------|--------------|
| **Text-to-Text** | Text → Text | LLaMA-style Transformer (1.3B–7B) |
| **Text-to-Speech** | Text → Audio | VITS2 / Piper |
| **Speech-to-Speech** | Audio → Audio | Whisper + LLM + TTS pipeline |
| **Image Generation** | Text → Image | Latent Diffusion (Stable Diffusion) |

## Development Roadmap

> **Legend:** ✅ Complete & Tested | 📝 Code Written (not yet tested) | 🔧 In Progress | ⬜ Planned

### Phase 1 — Core Foundation
| Status | Milestone | Description |
|:------:|-----------|-------------|
| ✅ | Project Architecture | Design system architecture, define modules, create project structure |
| ✅ | Text-to-Text (LLM) | Phi-3 Mini 3.8B with float16 + automatic GPU/CPU split |
| ✅ | Text-to-Speech | Multi-engine TTS: Edge TTS (neural voices), Piper, espeak-ng, pyttsx3 with auto-detection |
| ✅ | Speech-to-Speech | Whisper ASR → LLM → TTS streaming pipeline with WebSocket conversation endpoint and mic UI |
| ✅ | Image Generation | Stable Diffusion 2.1 with LoRA/DreamBooth, auto device detection, `/image` command in chat UI |
| ✅ | Unified API Gateway | FastAPI server with OpenAI-compatible endpoints |
| ✅ | Web Chat UI | Browser-based chat interface |
| ✅ | Safety & Content Filtering | Input/output filters, PII redaction, NSFW image detection — integrated into all endpoints, 26 tests passing |
| ✅ | LoRA Fine-Tuning | Train and load custom LoRA adapters for the chat model |

### Phase 2 — Tool Use
| Status | Milestone | Description |
|:------:|-----------|-------------|
| ✅ | Calculator Tool | AST-based safe math evaluator — arithmetic, trig, logs, constants, with security limits (47 tests) |
| ✅ | Code Execution | Subprocess sandbox with import whitelist, AST pre-scan, 5s timeout, output truncation (41 tests) |
| ✅ | Tool Dispatch Framework | Prompt-engineered `[TOOL: name(args)]` tag detection, execution, and inline result replacement |

### Phase 3 — RAG (Retrieval-Augmented Generation)
| Status | Milestone | Description |
|:------:|-----------|-------------|
| ✅ | Vector Database | ChromaDB with all-MiniLM-L6-v2 embeddings, cosine similarity, dedup, metadata filtering (26 tests) |
| ✅ | Document Ingestion | Loader (txt/md/json/csv/pdf/html) + recursive chunker + ingestion pipeline (32 tests) |
| ✅ | Context Injection | ContextInjector retrieves & scores chunks, augments system prompt, integrated into /v1/chat/completions (20 tests) |
| ⬜ | Web Search Integration | Query live search APIs for real-time information retrieval |

### Phase 4 — Model Upgrade
| Status | Milestone | Description |
|:------:|-----------|-------------|
| ✅ | Upgrade Base Model | Upgraded from TinyLlama 1.1B to Phi-3 Mini 3.8B (microsoft/Phi-3-mini-4k-instruct) |
| ✅ | Quantization (4-bit / 8-bit) | 4-bit NF4 quantization via bitsandbytes — fits in 8GB VRAM (RTX 3070 Ti) |
| ⬜ | Re-fine-tune on Reasoning Data | Train on GSM8K, OpenOrca, and other reasoning-focused datasets |

### Phase 5 — Deep Thinking Mode
| Status | Milestone | Description |
|:------:|-----------|-------------|
| ⬜ | Chain-of-Thought Prompting | System prompt mode that forces step-by-step reasoning |
| ⬜ | Self-Reflection | Model reviews and corrects its own answers before responding |
| ⬜ | Multi-Step Problem Solving | Break complex queries into sub-tasks and solve sequentially |

### Phase 6 — UI & Mode Toggles
| Status | Milestone | Description |
|:------:|-----------|-------------|
| ⬜ | Quick Answer Mode | Default fast-response mode for simple queries |
| ⬜ | Deep Thinking Toggle | UI switch to enable chain-of-thought reasoning |
| ⬜ | Web Search Toggle | UI switch to enable live search-augmented answers |
| ⬜ | Image Generation UI | Dedicated panel for text-to-image with parameter controls |
| ⬜ | Voice Selector | UI control to switch the AI's TTS voice (engine and voice selection) |
| ⬜ | Live Voice Conversation UI | Fullscreen conversational voice mode — animated waveform/orb visualizer, push-to-talk or auto-detect, live transcript overlay, seamless turn-taking (similar to ChatGPT / Character.AI voice mode) |

## Quick Start

```bash
# 1. Setup environment
bash scripts/setup.sh

# 2. Activate
source .venv/bin/activate

# 3. Start the API
uvicorn src.api.server:app --host 0.0.0.0 --port 8000

# 4. Test it
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## Fine-Tuning

```bash
# Fine-tune the LLM with LoRA
python -m src.text_to_text.train \
    --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --dataset tatsu-lab/alpaca \
    --output_dir ./checkpoints/my-chat-model \
    --epochs 3

# Fine-tune image generation with LoRA
# See docs/ARCHITECTURE.md § Image Generation
```

## Project Structure

```
CustomAiModel/
├── docs/
│   └── ARCHITECTURE.md          # Full design document
├── src/
│   ├── text_to_text/
│   │   ├── model.py             # LLM architecture + loading
│   │   └── train.py             # LoRA fine-tuning script
│   ├── text_to_speech/
│   │   └── tts_engine.py        # VITS / Piper TTS
│   ├── speech_to_speech/
│   │   └── pipeline.py          # ASR → LLM → TTS streaming
│   ├── image_generation/
│   │   └── diffusion.py         # Stable Diffusion + LoRA
│   ├── api/
│   │   └── server.py            # Unified FastAPI gateway
│   └── utils/
│       └── safety.py            # Content filtering & PII redaction
├── configs/
│   └── model_config.yaml        # Master configuration
├── scripts/
│   └── setup.sh                 # Environment setup
├── requirements.txt
├── Dockerfile
└── README.md
```

## Documentation

See **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** for the complete design document covering:

- Architecture decisions and trade-offs
- Training setup, hardware requirements, and datasets
- Customization and fine-tuning guides
- Resource efficiency strategies (quantization, distillation)
- Ethical and safety best practices
- Step-by-step implementation roadmap

## Hardware Requirements

| Tier | GPU | What You Can Do |
|------|-----|-----------------|
| Minimum | RTX 3090/4090 (24 GB) | Fine-tune all modules |
| Recommended | 2× A6000 (48 GB) | Train 1.3B LLM |
| Ideal | 4× A100 (320 GB) | Train 7B LLM from scratch |

## License

This project is open-source. Individual model weights may have their own licenses — check each model's page on HuggingFace.
