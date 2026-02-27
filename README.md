# Custom Multimodal AI Model

A modular, open-source multimodal AI system supporting **Text, Speech, and Image** generation.

## Capabilities

| Module | Input → Output | Architecture |
|--------|---------------|--------------|
| **Text-to-Text** | Text → Text | LLaMA-style Transformer (1.3B–7B) |
| **Text-to-Speech** | Text → Audio | VITS2 / Piper |
| **Speech-to-Speech** | Audio → Audio | Whisper + LLM + TTS pipeline |
| **Image Generation** | Text → Image | Latent Diffusion (Stable Diffusion) |

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
