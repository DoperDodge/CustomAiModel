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


_BASE_SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant. "
    "Keep your responses concise and conversational."
)

# SYSTEM_PROMPT is built after tool_dispatcher is created (see below)

# ──────────────────────────────────────────────
# Safety Pipeline
# ──────────────────────────────────────────────

from src.utils.safety import SafetyPipeline, SafetyConfig

safety = SafetyPipeline(SafetyConfig())

# ──────────────────────────────────────────────
# Tool Dispatch
# ──────────────────────────────────────────────

from src.tools.dispatch import create_default_dispatcher

tool_dispatcher = create_default_dispatcher()

SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT + tool_dispatcher.registry.system_prompt_section()

# ──────────────────────────────────────────────
# RAG — Context Injection (optional)
# ──────────────────────────────────────────────
# Imports are deferred so the server starts even without chromadb installed.

_rag_injector = None
_rag_init_attempted = False

def get_rag_injector():
    """Lazy-init the RAG context injector from config (returns None if disabled)."""
    global _rag_injector, _rag_init_attempted
    if _rag_init_attempted:
        return _rag_injector
    _rag_init_attempted = True

    try:
        from src.rag.context import ContextInjector
        from src.rag.vector_store import VectorStore
        import yaml
    except ImportError:
        # chromadb or pyyaml not installed — RAG disabled
        return None

    try:
        config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "model_config.yaml"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        rag_cfg = cfg.get("rag", {})
        vs_cfg = rag_cfg.get("vector_store", {})
        search_cfg = rag_cfg.get("search", {})
        persist_path = vs_cfg.get("persist_path")
        if not persist_path:
            return None

        store = VectorStore.persistent(
            path=persist_path,
            collection_name=vs_cfg.get("collection_name", "documents"),
        )
        if store.count() == 0:
            return None

        _rag_injector = ContextInjector(
            store=store,
            n_results=search_cfg.get("n_results", 5),
            min_score=search_cfg.get("min_score", 0.3),
        )
        return _rag_injector
    except Exception:
        return None

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
            from src.text_to_speech.tts_engine import create_tts_engine
            self._models["tts"] = create_tts_engine()
        self._last_used["tts"] = time.time()
        return self._models["tts"]

    def get_image_gen(self):
        if "image_gen" not in self._models:
            print("[ModelManager] Loading Image Generator...")
            from src.image_generation.diffusion import StableDiffusionGenerator, ImageGenConfig
            config = ImageGenConfig(device="auto")
            self._models["image_gen"] = StableDiffusionGenerator(config)
        self._last_used["image_gen"] = time.time()
        return self._models["image_gen"]

    def get_s2s_pipeline(self):
        if "s2s" not in self._models:
            print("[ModelManager] Loading S2S Pipeline...")
            from src.speech_to_speech.pipeline import SpeechToSpeechPipeline, S2SConfig
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = S2SConfig(whisper_device=device)
            model, tokenizer = self.get_llm()
            tts = self.get_tts()
            self._models["s2s"] = SpeechToSpeechPipeline(config, model, tokenizer, tts)
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
    return html_path.read_text(encoding="utf-8")


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
    # Safety: check user input
    user_text = " ".join(m.content for m in request.messages if m.role == "user")
    is_safe, reason = safety.check_input(user_text)
    if not is_safe:
        raise HTTPException(status_code=400, detail=reason)

    try:
        model, tokenizer = models.get_llm()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM not available: {e}")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # RAG: inject relevant context into the system prompt
    rag = get_rag_injector()
    if rag is not None:
        messages = rag.augment_messages(messages, SYSTEM_PROMPT)
    elif not any(m.role == "system" for m in request.messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
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

    # Tools: detect and execute any tool calls in the response
    response_text, tool_results = tool_dispatcher.process(response_text)
    if tool_results:
        print(f"[Tools] Executed: {[r.call.tool_name for r in tool_results]}")

    # Safety: sanitize output (truncate if too long)
    response_text = safety.output_filter.sanitize(response_text)

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
    _DONE = object()
    streamer_iter = iter(streamer)
    loop = asyncio.get_event_loop()
    total_len = 0
    max_len = safety.config.max_response_length

    # Buffer to accumulate text for tool-call detection across chunk boundaries
    buffer = ""

    def _make_chunk(content: str) -> str:
        return f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"

    # Read tokens off the streamer in a thread so we don't block the event loop
    while True:
        text = await loop.run_in_executor(
            None, lambda: next(streamer_iter, _DONE)
        )
        if text is _DONE:
            break

        # Safety: enforce max response length on streaming output
        total_len += len(text)
        if total_len > max_len:
            break

        buffer += text

        # If we see an opening tag but no closing tag, keep buffering
        if "[TOOL:" in buffer and "]" not in buffer.split("[TOOL:")[-1]:
            continue
        if "[CODE]" in buffer and "[/CODE]" not in buffer:
            continue
        if "[CODE" in buffer and "]" not in buffer.split("[CODE")[-1]:
            continue

        # Tools: process any complete tool calls in the buffer
        processed, results = tool_dispatcher.process(buffer)
        if results:
            print(f"[Tools] Stream executed: {[r.call.tool_name for r in results]}")
        yield _make_chunk(processed)
        buffer = ""

    # Flush any remaining buffer
    if buffer:
        processed, _ = tool_dispatcher.process(buffer)
        yield _make_chunk(processed)

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

    # Run in thread pool so async engines (Edge TTS) can use asyncio.run()
    audio_bytes, media_type = await asyncio.to_thread(
        tts.synthesize_to_bytes, request.input
    )

    return StreamingResponse(io.BytesIO(audio_bytes), media_type=media_type)


# ──────────────────────────────────────────────
# Speech-to-Speech (WebSocket)
# ──────────────────────────────────────────────

@app.websocket("/v1/audio/conversation")
async def speech_to_speech(websocket: WebSocket):
    """Real-time speech conversation over WebSocket.

    Protocol:
        Client sends: binary audio frames (int16, 16kHz, mono)
        Client sends: JSON {"type": "end_of_speech"} to signal turn end
        Server sends: JSON {"type": "transcript", "text": "..."} (what user said)
        Server sends: JSON {"type": "audio", "media_type": "audio/..."} + binary audio
        Server sends: JSON {"type": "response_text", "text": "..."} (full LLM response)
        Server sends: JSON {"type": "end_of_response"}
    """
    await websocket.accept()
    pipeline = models.get_s2s_pipeline()
    audio_buffer = []

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                chunk = np.frombuffer(data["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(chunk)

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "end_of_speech" and audio_buffer:
                    full_audio = np.concatenate(audio_buffer)
                    audio_buffer.clear()

                    async for event in pipeline.process_streaming(full_audio):
                        if event["type"] == "transcript":
                            await websocket.send_json(event)
                        elif event["type"] == "audio":
                            await websocket.send_json({
                                "type": "audio",
                                "media_type": event["media_type"],
                            })
                            await websocket.send_bytes(event["data"])
                        elif event["type"] == "response_text":
                            await websocket.send_json(event)

                    await websocket.send_json({"type": "end_of_response"})

    except WebSocketDisconnect:
        pass
    except RuntimeError as e:
        if "disconnect" in str(e).lower():
            pass  # Client disconnected mid-receive, safe to ignore
        else:
            raise


# ──────────────────────────────────────────────
# Image Generation
# ──────────────────────────────────────────────

@app.post("/v1/images/generations")
async def generate_image(request: ImageRequest):
    # Safety: check the image prompt
    is_safe, reason = safety.check_input(request.prompt)
    if not is_safe:
        raise HTTPException(status_code=400, detail=reason)

    try:
        gen = models.get_image_gen()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Image generator not available: {e}")

    width, height = map(int, request.size.split("x"))

    try:
        # Run in thread pool — diffusion is CPU/GPU-heavy and blocks the event loop
        images = await asyncio.to_thread(
            gen.generate,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=width,
            height=height,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            num_images=request.n,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    if not isinstance(images, list):
        images = [images]

    # Safety: NSFW check on generated images
    safe_images = []
    for img in images:
        is_safe, score = await asyncio.to_thread(safety.check_image, img)
        if is_safe:
            safe_images.append(img)
        else:
            print(f"[Safety] NSFW image blocked (score={score:.2f})")

    if not safe_images:
        raise HTTPException(
            status_code=400,
            detail="Generated image was blocked by the safety filter. Try a different prompt.",
        )

    data = []
    for img in safe_images:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        data.append(ImageData(b64_json=b64))

    return ImageResponse(created=int(time.time()), data=data)
