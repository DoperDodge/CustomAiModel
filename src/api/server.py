"""
Unified API Gateway — FastAPI Server

Exposes all modalities through a single REST/WebSocket API.
Mirrors the OpenAI API format for compatibility with existing tooling.

Endpoints:
    POST /v1/chat/completions       → Text-to-Text (streaming supported)
    POST /v1/audio/speech            → Text-to-Speech
    POST /v1/audio/transcriptions    → Speech-to-Text
    WS   /v1/audio/conversation      → Speech-to-Speech (real-time)
    POST /v1/images/generations      → Image Generation
    GET  /v1/models                  → List loaded models
    GET  /health                     → Health check

Run:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import io
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Request / Response Schemas
# ──────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "custom-llm"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]


class TTSRequest(BaseModel):
    model: str = "custom-tts"
    input: str
    voice: str = "default"
    response_format: str = "wav"  # wav, mp3
    speed: float = 1.0


class ImageRequest(BaseModel):
    model: str = "custom-diffusion"
    prompt: str
    negative_prompt: str = "blurry, bad quality, distorted"
    n: int = 1
    size: str = "512x512"
    guidance_scale: float = 7.5
    num_steps: int = 30
    seed: int | None = None


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None


class ImageResponse(BaseModel):
    created: int
    data: list[ImageData]


# ──────────────────────────────────────────────
# Model Manager — Lazy Loading & Memory Management
# ──────────────────────────────────────────────

class ModelManager:
    """Manages model lifecycle — lazy loading, caching, and eviction.

    Only loads a model when its endpoint is first called.
    Tracks last-used time for LRU eviction when VRAM is constrained.
    """

    def __init__(self):
        self._models: dict[str, object] = {}
        self._last_used: dict[str, float] = {}

    def get_llm(self):
        if "llm" not in self._models:
            print("[ModelManager] Loading LLM...")
            from src.text_to_text.model import load_pretrained_llm
            lora_path = Path("checkpoints/t2t-chat")
            # Only load LoRA if adapter_config.json exists and matches the base model
            use_lora = None
            if lora_path.exists() and (lora_path / "adapter_config.json").exists():
                import json
                adapter_cfg = json.loads((lora_path / "adapter_config.json").read_text())
                base_model = adapter_cfg.get("base_model_name_or_path", "")
                # Only use LoRA weights if they were trained on the current base model
                if "Phi-3" in base_model or "phi-3" in base_model:
                    use_lora = str(lora_path)
                else:
                    print(f"[ModelManager] Skipping LoRA — trained on {base_model}, not compatible with current model")
            model, tokenizer = load_pretrained_llm(
                lora_path=use_lora,
                quantization="int4",
            )
            self._models["llm"] = (model, tokenizer)
        self._last_used["llm"] = time.time()
        return self._models["llm"]

    def get_tts(self):
        if "tts" not in self._models:
            print("[ModelManager] Loading TTS...")
            from src.text_to_speech.tts_engine import PiperTTS
            self._models["tts"] = PiperTTS()
        self._last_used["tts"] = time.time()
        return self._models["tts"]

    def get_image_gen(self):
        if "image_gen" not in self._models:
            print("[ModelManager] Loading Image Generator...")
            from src.image_generation.diffusion import StableDiffusionGenerator
            self._models["image_gen"] = StableDiffusionGenerator()
        self._last_used["image_gen"] = time.time()
        return self._models["image_gen"]

    def get_s2s_pipeline(self):
        if "s2s" not in self._models:
            print("[ModelManager] Loading S2S Pipeline...")
            from src.speech_to_speech.pipeline import SpeechToSpeechPipeline, S2SConfig
            self._models["s2s"] = SpeechToSpeechPipeline(S2SConfig())
        self._last_used["s2s"] = time.time()
        return self._models["s2s"]

    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def unload(self, name: str) -> None:
        if name in self._models:
            model = self._models.pop(name)
            if hasattr(model, "unload"):
                model.unload()
            del model
            self._last_used.pop(name, None)
            import torch
            torch.cuda.empty_cache()

    def evict_lru(self) -> None:
        """Unload the least recently used model to free VRAM."""
        if not self._last_used:
            return
        oldest = min(self._last_used, key=self._last_used.get)
        print(f"[ModelManager] Evicting LRU model: {oldest}")
        self.unload(oldest)


# ──────────────────────────────────────────────
# Application
# ──────────────────────────────────────────────

models = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Custom AI Model API starting...")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Custom AI Model API",
    description="Multimodal AI system: Text, Speech, and Image generation",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    html_path = Path(__file__).parent / "chat_ui.html"
    return html_path.read_text()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "loaded_models": models.loaded_models(),
    }


@app.get("/v1/models")
async def list_models():
    available = [
        {"id": "custom-llm", "object": "model", "owned_by": "custom"},
        {"id": "custom-tts", "object": "model", "owned_by": "custom"},
        {"id": "custom-diffusion", "object": "model", "owned_by": "custom"},
        {"id": "custom-s2s", "object": "model", "owned_by": "custom"},
    ]
    return {"object": "list", "data": available}


# ──────────────────────────────────────────────
# Text-to-Text
# ──────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        model, tokenizer = models.get_llm()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM not available: {e}")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    if request.stream:
        return StreamingResponse(
            _stream_chat(model, tokenizer, inputs, request),
            media_type="text/event-stream",
        )

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
        )

    response_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=response_text))],
    )


async def _stream_chat(model, tokenizer, inputs, request: ChatRequest) -> AsyncGenerator[str, None]:
    """Server-Sent Events stream for chat completions."""
    import threading
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": request.max_tokens,
        "temperature": request.temperature,
        "do_sample": True,
        "streamer": streamer,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    for text in streamer:
        chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"
    thread.join()


# ──────────────────────────────────────────────
# Text-to-Speech
# ──────────────────────────────────────────────

@app.post("/v1/audio/speech")
async def text_to_speech(request: TTSRequest):
    try:
        tts = models.get_tts()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"TTS not available: {e}")

    audio = tts.synthesize(request.input)

    # Convert to WAV bytes
    wav_buffer = io.BytesIO()
    import wave
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(audio.tobytes())

    wav_buffer.seek(0)
    return StreamingResponse(wav_buffer, media_type="audio/wav")


# ──────────────────────────────────────────────
# Speech-to-Speech (WebSocket)
# ──────────────────────────────────────────────

@app.websocket("/v1/audio/conversation")
async def speech_to_speech(websocket: WebSocket):
    """Real-time speech conversation over WebSocket.

    Protocol:
        Client sends: binary audio frames (int16, 16kHz, mono)
        Client sends: JSON {"type": "end_of_speech"} to signal turn end
        Server sends: binary audio frames (int16, 22050Hz, mono)
        Server sends: JSON {"type": "end_of_response"}
    """
    await websocket.accept()
    pipeline = models.get_s2s_pipeline()
    audio_buffer = []

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                # Accumulate audio
                chunk = np.frombuffer(data["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(chunk)

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "end_of_speech" and audio_buffer:
                    # Process accumulated audio
                    full_audio = np.concatenate(audio_buffer)
                    audio_buffer.clear()

                    async for audio_chunk in pipeline.process_streaming(full_audio):
                        await websocket.send_bytes(audio_chunk.tobytes())

                    await websocket.send_json({"type": "end_of_response"})

    except WebSocketDisconnect:
        pass


# ──────────────────────────────────────────────
# Image Generation
# ──────────────────────────────────────────────

@app.post("/v1/images/generations")
async def generate_image(request: ImageRequest):
    try:
        gen = models.get_image_gen()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Image generator not available: {e}")

    width, height = map(int, request.size.split("x"))

    images = gen.generate(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=width,
        height=height,
        num_steps=request.num_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        num_images=request.n,
    )

    if not isinstance(images, list):
        images = [images]

    data = []
    for img in images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        data.append(ImageData(b64_json=b64))

    return ImageResponse(created=int(time.time()), data=data)
