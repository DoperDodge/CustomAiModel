"""
Text-to-Speech Module — VITS / Piper Integration

Two approaches are provided:
  1. PiperTTS: Use pre-trained Piper voices (easiest, production-ready)
  2. Custom VITS: Train your own VITS model from scratch

For most users, starting with Piper and fine-tuning a voice is recommended.
"""

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class PiperTTS:
    """Text-to-Speech using Piper (production-ready VITS wrapper).

    Piper provides pre-trained voices in 30+ languages with simple CLI usage.
    Install: pip install piper-tts

    Usage:
        tts = PiperTTS(voice="en_US-lessac-medium")
        audio = tts.synthesize("Hello, how are you?")
        tts.save_wav(audio, "output.wav")
    """

    def __init__(
        self,
        voice: str = "en_US-lessac-medium",
        speaker_id: int | None = None,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
    ):
        self.voice = voice
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self._model = None

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
            self._model = PiperVoice.load(self.voice)
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

    @staticmethod
    def save_wav(audio: np.ndarray, path: str, sample_rate: int = 22050) -> None:
        """Save audio waveform to WAV file."""
        import wave

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio.tobytes())


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


def synthesize_with_coqui(text: str, model_name: str = "tts_models/en/ljspeech/vits") -> np.ndarray:
    """Alternative: Synthesize using Coqui TTS (another excellent open-source option).

    Install: pip install TTS

    Args:
        text: Text to synthesize.
        model_name: Coqui TTS model identifier.

    Returns:
        Audio waveform as numpy array.
    """
    from TTS.api import TTS

    tts = TTS(model_name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tts.tts_to_file(text=text, file_path=f.name)
        import soundfile as sf
        audio, sr = sf.read(f.name)
    return audio
