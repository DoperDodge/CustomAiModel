"""
Speech-to-Speech Module — Real-Time Conversational Pipeline

Architecture:
    Audio In → [Whisper ASR] → Text → [LLM] → Text → [TTS] → Audio Out

Streaming strategy:
    1. Whisper processes audio with VAD
    2. LLM generates tokens streamingly
    3. TTS begins synthesis on first sentence boundary
    4. Audio chunks are streamed to client via WebSocket
"""

import asyncio
import threading
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class S2SConfig:
    """Configuration for the Speech-to-Speech pipeline."""

    # ASR (Whisper)
    whisper_model: str = "base"  # tiny, base, small, medium, large-v3
    whisper_language: str = "en"
    whisper_device: str = "cuda"

    # Pipeline
    vad_threshold: float = 0.5
    sentence_delimiters: str = ".!?\n"
    audio_chunk_duration: float = 0.5  # seconds per streamed audio chunk


class WhisperASR:
    """Speech-to-Text using faster-whisper for low-latency streaming.

    faster-whisper uses CTranslate2 for 4x faster inference than
    the original Whisper implementation.
    """

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        self.model_size = model_size
        self.device = device
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError("Install faster-whisper: pip install faster-whisper")

            compute_type = "float16" if self.device == "cuda" else "int8"
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio waveform as float32 numpy array.
            sample_rate: Sample rate of the audio (Whisper expects 16kHz).

        Returns:
            Transcribed text.
        """
        model = self._load_model()
        segments, _info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )
        return " ".join(seg.text.strip() for seg in segments)


class SpeechToSpeechPipeline:
    """Complete Speech-to-Speech pipeline with streaming.

    Reuses the server's already-loaded LLM and TTS engines to avoid
    loading duplicate models into memory.

    Usage:
        pipeline = SpeechToSpeechPipeline(config, model, tokenizer, tts)
        async for audio_chunk, media_type in pipeline.process_streaming(audio):
            websocket.send(audio_chunk)
    """

    def __init__(self, config: S2SConfig, model, tokenizer, tts):
        self.config = config
        self.asr = WhisperASR(config.whisper_model, config.whisper_device)
        self.model = model
        self.tokenizer = tokenizer
        self.tts = tts

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> tuple[bytes, str]:
        """Process audio input and return audio response (non-streaming).

        Returns:
            (audio_bytes, media_type) from the TTS engine.
        """
        transcript = self.asr.transcribe(audio, sample_rate)
        print(f"[S2S] ASR: {transcript}")

        response_text = self._generate_response(transcript)
        print(f"[S2S] LLM: {response_text}")

        audio_bytes, media_type = self.tts.synthesize_to_bytes(response_text)
        return audio_bytes, media_type

    async def process_streaming(self, audio: np.ndarray, sample_rate: int = 16000):
        """Process audio and yield (audio_bytes, media_type, transcript, response_text) as ready.

        First yields the transcript, then audio chunks on sentence boundaries.
        """
        from transformers import TextIteratorStreamer

        # Step 1: ASR
        loop = asyncio.get_event_loop()
        transcript = await loop.run_in_executor(
            None, self.asr.transcribe, audio, sample_rate
        )
        print(f"[S2S] ASR: {transcript}")

        if not transcript.strip():
            return

        # Yield transcript as a text event
        yield {"type": "transcript", "text": transcript}

        # Step 2+3: Stream LLM → TTS on sentence boundaries
        messages = [
            {"role": "user", "content": transcript},
        ]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        sentence_buffer = ""
        full_response = ""
        delimiters = set(self.config.sentence_delimiters)
        _DONE = object()
        streamer_iter = iter(streamer)

        while True:
            text_chunk = await loop.run_in_executor(
                None, lambda: next(streamer_iter, _DONE)
            )
            if text_chunk is _DONE:
                break

            sentence_buffer += text_chunk
            full_response += text_chunk

            # Check for sentence boundary
            if any(c in delimiters for c in text_chunk):
                sentence = sentence_buffer.strip()
                if sentence:
                    audio_bytes, media_type = await loop.run_in_executor(
                        None, self.tts.synthesize_to_bytes, sentence
                    )
                    yield {"type": "audio", "data": audio_bytes, "media_type": media_type}
                    sentence_buffer = ""

        # Synthesize any remaining text
        if sentence_buffer.strip():
            audio_bytes, media_type = await loop.run_in_executor(
                None, self.tts.synthesize_to_bytes, sentence_buffer.strip()
            )
            yield {"type": "audio", "data": audio_bytes, "media_type": media_type}

        thread.join()
        yield {"type": "response_text", "text": full_response}

    def _generate_response(self, user_message: str) -> str:
        """Generate a full LLM response (non-streaming)."""
        messages = [{"role": "user", "content": user_message}]
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=256, temperature=0.7, do_sample=True,
            )

        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
