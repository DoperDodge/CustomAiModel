"""
Text-to-Speech Module — Multi-Engine TTS

Supported engines (auto-detected in priority order):
  1. PiperTTS:  High-quality VITS2 voices (requires .onnx model download)
  2. EspeakTTS: Lightweight offline synthesis via espeak-ng (always available)

Usage:
    from src.text_to_speech.tts_engine import create_tts_engine

    tts = create_tts_engine()          # auto-detect best available
    audio = tts.synthesize("Hello!")
    tts.save_wav(audio, "output.wav")
"""

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
# Factory — auto-detect best engine
# ──────────────────────────────────────────────

def create_tts_engine(
    engine: str = "auto",
    voice: str | None = None,
    models_dir: str | Path | None = None,
    **kwargs,
) -> BaseTTS:
    """Create a TTS engine, auto-detecting the best available.

    Args:
        engine: "piper", "espeak", or "auto" (try piper first, fall back to espeak).
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

    if engine == "espeak" or (engine == "auto" and EspeakTTS.is_available()):
        espeak_voice = voice or "en"
        print(f"[TTS] Using espeak-ng engine (voice: {espeak_voice})")
        return EspeakTTS(voice=espeak_voice, **kwargs)

    raise RuntimeError(
        "No TTS engine available. Install one of:\n"
        "  1. Piper: pip install piper-tts && python -m piper.download_voices en_US-lessac-medium --download-dir models/tts\n"
        "  2. espeak-ng: apt install espeak-ng"
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
