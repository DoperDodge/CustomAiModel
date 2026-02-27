"""
Speech-to-Speech Module — Real-Time Conversational Pipeline

Architecture:
    Audio In → [Whisper ASR] → Text → [LLM] → Text → [TTS] → Audio Out

Streaming strategy:
    1. Whisper processes audio in 1-second chunks with VAD
    2. LLM generates tokens streamingly
    3. TTS begins synthesis on first sentence boundary
    4. Audio chunks are streamed to client via WebSocket
"""

import asyncio
import queue
import threading
import time
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

    # LLM
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7

    # TTS
    tts_voice: str = "en_US-lessac-medium"

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

    def transcribe_streaming(self, audio_chunks: list[np.ndarray]) -> str:
        """Transcribe a stream of audio chunks.

        Concatenates chunks and transcribes when VAD detects speech ended.
        """
        combined = np.concatenate(audio_chunks)
        return self.transcribe(combined)


class StreamingLLM:
    """Streaming text generation for real-time conversation."""

    def __init__(self, model_name: str, max_tokens: int = 256, temperature: float = 0.7):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        return self._model, self._tokenizer

    def generate_stream(self, user_message: str, system_prompt: str = "You are a helpful assistant."):
        """Generate a response token by token.

        Yields text chunks as they are generated. This allows the TTS
        to begin synthesis before the full response is complete.
        """
        from transformers import TextIteratorStreamer

        model, tokenizer = self._load_model()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generation_kwargs = {
            **inputs,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": True,
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for text_chunk in streamer:
            yield text_chunk

        thread.join()


class SpeechToSpeechPipeline:
    """Complete Speech-to-Speech pipeline with streaming.

    Usage:
        config = S2SConfig()
        pipeline = SpeechToSpeechPipeline(config)

        # Process a single audio input (non-streaming)
        response_audio = pipeline.process(input_audio)

        # Stream responses (for WebSocket integration)
        async for audio_chunk in pipeline.process_streaming(input_audio):
            websocket.send(audio_chunk)
    """

    def __init__(self, config: S2SConfig):
        self.config = config
        self.asr = WhisperASR(config.whisper_model, config.whisper_device)
        self.llm = StreamingLLM(config.llm_model, config.llm_max_tokens, config.llm_temperature)
        self.tts = None  # Lazy load

    def _get_tts(self):
        if self.tts is None:
            from src.text_to_speech.tts_engine import PiperTTS
            self.tts = PiperTTS(voice=self.config.tts_voice)
        return self.tts

    def process(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Process audio input and return audio response (non-streaming).

        Args:
            audio: Input audio waveform (float32, 16kHz).
            sample_rate: Sample rate of input audio.

        Returns:
            Response audio waveform.
        """
        # Step 1: ASR
        transcript = self.asr.transcribe(audio, sample_rate)
        print(f"[ASR] User said: {transcript}")

        # Step 2: LLM
        response_text = ""
        for chunk in self.llm.generate_stream(transcript):
            response_text += chunk
        print(f"[LLM] Response: {response_text}")

        # Step 3: TTS
        tts = self._get_tts()
        response_audio = tts.synthesize(response_text)
        print(f"[TTS] Generated {len(response_audio) / 22050:.1f}s of audio")

        return response_audio

    async def process_streaming(self, audio: np.ndarray, sample_rate: int = 16000):
        """Process audio input and yield audio chunks as they're ready.

        This is the streaming version for real-time conversation.
        Yields audio chunks as soon as each sentence is synthesized.
        """
        # Step 1: ASR (fast, non-streaming)
        transcript = self.asr.transcribe(audio, sample_rate)
        print(f"[ASR] User said: {transcript}")

        # Step 2+3: LLM streaming → TTS on sentence boundaries
        tts = self._get_tts()
        sentence_buffer = ""
        delimiters = set(self.config.sentence_delimiters)

        for text_chunk in self.llm.generate_stream(transcript):
            sentence_buffer += text_chunk

            # Check for sentence boundary
            if any(c in delimiters for c in text_chunk):
                # Synthesize the completed sentence
                sentence = sentence_buffer.strip()
                if sentence:
                    audio_chunk = tts.synthesize(sentence)
                    yield audio_chunk
                    sentence_buffer = ""

        # Synthesize any remaining text
        if sentence_buffer.strip():
            audio_chunk = tts.synthesize(sentence_buffer.strip())
            yield audio_chunk

    def measure_latency(self, audio: np.ndarray) -> dict[str, float]:
        """Measure latency of each pipeline stage.

        Returns timing for ASR, LLM (first token), and TTS stages.
        """
        timings = {}

        # ASR latency
        start = time.time()
        transcript = self.asr.transcribe(audio)
        timings["asr_ms"] = (time.time() - start) * 1000

        # LLM first-token latency
        start = time.time()
        first_token = True
        response_text = ""
        for chunk in self.llm.generate_stream(transcript):
            if first_token:
                timings["llm_first_token_ms"] = (time.time() - start) * 1000
                first_token = False
            response_text += chunk
        timings["llm_total_ms"] = (time.time() - start) * 1000

        # TTS latency
        tts = self._get_tts()
        start = time.time()
        tts.synthesize(response_text[:100])  # First sentence approx
        timings["tts_first_chunk_ms"] = (time.time() - start) * 1000

        timings["total_first_audio_ms"] = (
            timings["asr_ms"] + timings["llm_first_token_ms"] + timings["tts_first_chunk_ms"]
        )

        return timings
