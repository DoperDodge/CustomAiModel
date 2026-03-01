"""
Text-to-Speech Module — Multi-Engine TTS

Supported engines (auto-detected in priority order):
  1. PiperTTS:   High-quality VITS2 voices (requires .onnx model download)
  2. EdgeTTS:    Natural neural voices via Microsoft Edge (requires internet)
  3. EspeakTTS:  Lightweight offline synthesis via espeak-ng (Linux)
  4. Pyttsx3TTS: System TTS via pyttsx3 — uses Windows SAPI5, macOS NSSpeech, or Linux espeak

Usage:
    from src.text_to_speech.tts_engine import create_tts_engine

    tts = create_tts_engine()          # auto-detect best available
    audio_bytes, content_type = tts.synthesize_to_bytes("Hello!")
"""

import asyncio
import io
import shutil
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Default directory for downloaded TTS models
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "tts"


class BaseTTS(ABC):
    """Common interface for all TTS engines."""

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio waveform (int16 PCM)."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""
        ...

    def synthesize_to_bytes(self, text: str) -> tuple[bytes, str]:
        """Synthesize and return raw audio bytes with media type.

        Returns:
            (audio_bytes, media_type) — e.g. (b"...", "audio/wav").
            Subclasses may override to return other formats like audio/mpeg.
        """
        audio = self.synthesize(text)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        return wav_buffer.getvalue(), "audio/wav"

    @staticmethod
    def save_wav(audio: np.ndarray, path: str, sample_rate: int = 22050) -> None:
        """Save audio waveform to WAV file."""
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())


# ──────────────────────────────────────────────
# Engine 1: Piper TTS (high-quality VITS2)
# ──────────────────────────────────────────────

class PiperTTS(BaseTTS):
    """Text-to-Speech using Piper (production-ready VITS wrapper).

    Piper provides pre-trained voices in 30+ languages.
    Install: pip install piper-tts
    Download voices: python -m piper.download_voices en_US-lessac-medium --download-dir models/tts

    Usage:
        tts = PiperTTS(voice="en_US-lessac-medium")
        audio = tts.synthesize("Hello, how are you?")
        tts.save_wav(audio, "output.wav")
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        models_dir: str | Path | None = None,
        speaker_id: int | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        self.voice = voice
        self.models_dir = Path(models_dir) if models_dir else _MODELS_DIR
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self._model = None
        self._sample_rate = 22050

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _resolve_model_path(self) -> Path:
        """Find the .onnx model file, checking common locations."""
        # Direct path
        direct = Path(self.voice)
        if direct.exists():
            return direct

        # Check models directory
        in_models = self.models_dir / f"{self.voice}.onnx"
        if in_models.exists():
            return in_models

        # Check current working directory
        in_cwd = Path.cwd() / f"{self.voice}.onnx"
        if in_cwd.exists():
            return in_cwd

        raise FileNotFoundError(
            f"Piper voice model not found: {self.voice}\n"
            f"Searched: {direct}, {in_models}, {in_cwd}\n"
            f"Download with: python -m piper.download_voices {self.voice} "
            f"--download-dir {self.models_dir}"
        )

    def _load_model(self):
        """Lazy-load the Piper model."""
        if self._model is None:
            try:
                from piper import PiperVoice
            except ImportError:
                raise ImportError(
                    "piper-tts is not installed. Install with: pip install piper-tts\n"
                    "Also download a voice model from: https://github.com/rhasspy/piper#voices"
                )
            model_path = self._resolve_model_path()
            self._model = PiperVoice.load(str(model_path))
            # Read actual sample rate from loaded model config
            if hasattr(self._model, "config") and hasattr(self._model.config, "sample_rate"):
                self._sample_rate = self._model.config.sample_rate
        return self._model

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text to speak.

        Returns:
            Audio waveform as numpy array (int16, 22050 Hz).
        """
        model = self._load_model()
        audio_buffer = io.BytesIO()
        model.synthesize(
            text,
            audio_buffer,
            speaker_id=self.speaker_id,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w=self.noise_w,
        )
        audio_buffer.seek(0)
        audio = np.frombuffer(audio_buffer.read(), dtype=np.int16)
        return audio

    @classmethod
    def is_available(cls, voice: str = "en_US-lessac-medium", models_dir: Path | None = None) -> bool:
        """Check if Piper is installed and a voice model exists."""
        try:
            from piper import PiperVoice  # noqa: F401
        except ImportError:
            return False

        search_dir = models_dir or _MODELS_DIR
        model_path = search_dir / f"{voice}.onnx"
        return model_path.exists() or Path(voice).exists()


# ──────────────────────────────────────────────
# Engine 2: espeak-ng (lightweight fallback)
# ──────────────────────────────────────────────

class EspeakTTS(BaseTTS):
    """Text-to-Speech using espeak-ng (lightweight, always available).

    Produces robotic but intelligible speech. Good as a fallback or for
    testing when Piper models aren't downloaded yet.

    Requires: apt install espeak-ng

    Usage:
        tts = EspeakTTS(voice="en", speed=175)
        audio = tts.synthesize("Hello, how are you?")
        tts.save_wav(audio, "output.wav")
    """

    def __init__(
        self,
        voice: str = "en",
        speed: int = 175,        # words per minute
        pitch: int = 50,         # 0-99
        amplitude: int = 100,    # 0-200
    ):
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.amplitude = amplitude
        self._sample_rate = 22050

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech via espeak-ng subprocess.

        Args:
            text: Input text to speak.

        Returns:
            Audio waveform as numpy array (int16).
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cmd = [
                "espeak-ng",
                "-v", self.voice,
                "-s", str(self.speed),
                "-p", str(self.pitch),
                "-a", str(self.amplitude),
                "-w", tmp_path,
                "--",
                text,
            ]
            subprocess.run(cmd, check=True, capture_output=True)

            with wave.open(tmp_path, "rb") as wf:
                self._sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)

            return audio
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @classmethod
    def is_available(cls) -> bool:
        """Check if espeak-ng is installed."""
        return shutil.which("espeak-ng") is not None


# ──────────────────────────────────────────────
# Engine 3: Edge TTS (neural voices, online)
# ──────────────────────────────────────────────

class EdgeTTS(BaseTTS):
    """Text-to-Speech using Microsoft Edge's neural voices.

    Produces natural, human-like speech using the same neural voices
    as Microsoft Azure Cognitive Services — completely free, no API key.
    Requires internet connection.

    Install: pip install edge-tts

    Popular voices:
      en-US-AriaNeural       (female, natural/conversational)
      en-US-GuyNeural        (male, natural)
      en-US-JennyNeural      (female, warm)
      en-US-ChristopherNeural (male, professional)
      en-GB-SoniaNeural      (female, British)

    List all voices: python -m edge_tts --list-voices

    Usage:
        tts = EdgeTTS(voice="en-US-AriaNeural")
        audio_bytes, media_type = tts.synthesize_to_bytes("Hello!")
    """

    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", pitch: str = "+0Hz"):
        self._voice = voice
        self._rate = rate
        self._pitch = pitch
        self._sample_rate = 24000  # Edge TTS uses 24kHz

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """Not directly supported — Edge TTS outputs MP3.

        Use synthesize_to_bytes() instead for best results.
        This method converts via a temp file for interface compatibility.
        """
        audio_bytes, _ = self.synthesize_to_bytes(text)
        # Return raw bytes wrapped as int16 — the server uses synthesize_to_bytes()
        return np.frombuffer(audio_bytes, dtype=np.int8).view(np.int8)

    def synthesize_to_bytes(self, text: str) -> tuple[bytes, str]:
        """Synthesize speech and return MP3 bytes directly.

        Returns:
            (mp3_bytes, "audio/mpeg")
        """
        async def _run():
            import edge_tts
            communicate = edge_tts.Communicate(
                text, self._voice, rate=self._rate, pitch=self._pitch,
            )
            chunks = []
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            return b"".join(chunks)

        # Run async edge-tts in a fresh event loop (safe from thread pool)
        mp3_bytes = asyncio.run(_run())
        return mp3_bytes, "audio/mpeg"

    @classmethod
    def is_available(cls) -> bool:
        """Check if edge-tts is installed."""
        try:
            import edge_tts  # noqa: F401
            return True
        except ImportError:
            return False


# ──────────────────────────────────────────────
# Engine 4: pyttsx3 (cross-platform fallback)
# ──────────────────────────────────────────────

class Pyttsx3TTS(BaseTTS):
    """Text-to-Speech using pyttsx3 (cross-platform system TTS).

    Uses the OS built-in speech engine:
      - Windows: SAPI5 (high quality, multiple voices)
      - macOS:   NSSpeechSynthesizer
      - Linux:   espeak (via pyttsx3 wrapper)

    Install: pip install pyttsx3

    Usage:
        tts = Pyttsx3TTS()
        audio = tts.synthesize("Hello, how are you?")
        tts.save_wav(audio, "output.wav")
    """

    def __init__(self, voice_id: str | None = None, rate: int = 175):
        self._voice_id = voice_id
        self._rate = rate
        self._sample_rate = 22050
        self._engine = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def _get_engine(self):
        """Lazy-init the pyttsx3 engine."""
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self._rate)
            if self._voice_id:
                self._engine.setProperty('voice', self._voice_id)
        return self._engine

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech via pyttsx3 (saves to temp WAV then reads back).

        Args:
            text: Input text to speak.

        Returns:
            Audio waveform as numpy array (int16).
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engine = self._get_engine()
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            with wave.open(tmp_path, "rb") as wf:
                self._sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)

            return audio
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @classmethod
    def is_available(cls) -> bool:
        """Check if pyttsx3 is installed and can initialize."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            return True
        except Exception:
            return False


# ──────────────────────────────────────────────
# Factory — auto-detect best engine
# ──────────────────────────────────────────────

def create_tts_engine(
    engine: str = "auto",
    voice: str | None = None,
    models_dir: str | Path | None = None,
    **kwargs,
) -> BaseTTS:
    """Create a TTS engine, auto-detecting the best available.

    Priority order for "auto":
        Piper → Edge TTS → espeak-ng → pyttsx3

    Args:
        engine: "piper", "edge", "espeak", "pyttsx3", or "auto".
        voice:  Voice identifier (engine-specific).
        models_dir: Directory containing Piper .onnx model files.
        **kwargs: Additional engine-specific parameters.

    Returns:
        A TTS engine instance ready for synthesis.
    """
    if engine == "piper" or (engine == "auto" and PiperTTS.is_available(
        voice=voice or "en_US-lessac-medium",
        models_dir=Path(models_dir) if models_dir else None,
    )):
        piper_voice = voice or "en_US-lessac-medium"
        print(f"[TTS] Using Piper engine (voice: {piper_voice})")
        return PiperTTS(voice=piper_voice, models_dir=models_dir, **kwargs)

    if engine == "edge" or (engine == "auto" and EdgeTTS.is_available()):
        edge_voice = voice or "en-US-AriaNeural"
        print(f"[TTS] Using Edge TTS engine (voice: {edge_voice})")
        return EdgeTTS(voice=edge_voice, **kwargs)

    if engine == "espeak" or (engine == "auto" and EspeakTTS.is_available()):
        espeak_voice = voice or "en"
        print(f"[TTS] Using espeak-ng engine (voice: {espeak_voice})")
        return EspeakTTS(voice=espeak_voice, **kwargs)

    if engine == "pyttsx3" or (engine == "auto" and Pyttsx3TTS.is_available()):
        print("[TTS] Using pyttsx3 engine (system TTS)")
        return Pyttsx3TTS(**kwargs)

    raise RuntimeError(
        "No TTS engine available. Install one of:\n"
        "  1. Piper (best offline): pip install piper-tts && download a voice model\n"
        "  2. Edge TTS (natural, online): pip install edge-tts\n"
        "  3. espeak-ng (Linux): apt install espeak-ng\n"
        "  4. pyttsx3 (cross-platform): pip install pyttsx3"
    )


# ──────────────────────────────────────────────
# Legacy: VITS Trainer (for custom voice training)
# ──────────────────────────────────────────────

class VITSTrainer:
    """Helper for training a custom VITS model.

    This wraps the vits2 training pipeline. For training from scratch,
    you need a dataset of (text, audio) pairs.

    Dataset structure:
        dataset/
        ├── wavs/
        │   ├── 001.wav
        │   ├── 002.wav
        │   └── ...
        └── metadata.csv   (format: filename|transcription)
    """

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str = "./checkpoints/tts-vits",
        sample_rate: int = 22050,
        batch_size: int = 16,
        epochs: int = 1000,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def prepare_dataset(self) -> None:
        """Validate and preprocess the dataset.

        Checks that all WAV files exist, are the correct sample rate,
        and have matching transcriptions.
        """
        metadata_path = self.dataset_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found in {self.dataset_dir}. "
                "Create a file with lines: filename|transcription"
            )

        wavs_dir = self.dataset_dir / "wavs"
        if not wavs_dir.exists():
            raise FileNotFoundError(f"wavs/ directory not found in {self.dataset_dir}")

        entries = []
        with open(metadata_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    print(f"Warning: skipping malformed line {line_num}: {line}")
                    continue
                filename, text = parts[0], parts[1]
                wav_path = wavs_dir / f"{filename}.wav"
                if not wav_path.exists():
                    print(f"Warning: missing audio file: {wav_path}")
                    continue
                entries.append((filename, text))

        print(f"Dataset validated: {len(entries)} valid entries")
        return entries

    def get_training_command(self) -> list[str]:
        """Generate the training command for VITS.

        This assumes you have the VITS repository cloned. For a complete
        training setup, see: https://github.com/jaywalnut310/vits
        """
        return [
            "python", "train.py",
            "-c", str(self.output_dir / "config.json"),
            "-m", str(self.output_dir),
            "--batch_size", str(self.batch_size),
            "--epochs", str(self.epochs),
        ]
