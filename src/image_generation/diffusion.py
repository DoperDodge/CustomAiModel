"""
Image Generation Module — Latent Diffusion (Stable Diffusion)

Two approaches:
  1. Use a pre-trained Stable Diffusion checkpoint (recommended start)
  2. Fine-tune with LoRA or DreamBooth for custom styles

This module wraps the diffusers library for a clean interface.

Setup:
  Many Stable Diffusion models on HuggingFace are now gated — you must:
    1. Create a free account at https://huggingface.co
    2. Accept the model license at the model page
    3. Run: huggingface-cli login  (or set HF_TOKEN env var)
"""

import os
from pathlib import Path

import torch
from dataclasses import dataclass, field

# Models to try in order of preference (some may be gated/removed)
FALLBACK_MODELS = [
    "stabilityai/stable-diffusion-2-1",
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
]


@dataclass
class ImageGenConfig:
    """Configuration for image generation."""

    model_id: str = "stabilityai/stable-diffusion-2-1"
    device: str = "auto"  # auto, cuda, cpu
    dtype: str = "float16"  # float16, bfloat16, float32
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    safety_checker: bool = True
    lora_weights: str | None = None  # Path to LoRA weights
    hf_token: str | None = None  # HuggingFace token (or set HF_TOKEN env var)

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CPU doesn't support float16
        if self.device == "cpu" and self.dtype == "float16":
            self.dtype = "float32"
        # Resolve token from env if not provided
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN")


class StableDiffusionGenerator:
    """Text-to-Image generation using Stable Diffusion.

    Usage:
        gen = StableDiffusionGenerator()
        image = gen.generate("A sunset over mountains, oil painting style")
        image.save("output.png")

        # With negative prompt
        image = gen.generate(
            prompt="A beautiful landscape",
            negative_prompt="blurry, low quality, distorted",
        )
    """

    def __init__(self, config: ImageGenConfig | None = None):
        self.config = config or ImageGenConfig()
        self._pipeline = None

    def _try_load_model(self, model_id: str, torch_dtype):
        """Attempt to load a single model, returning the pipeline or None."""
        from diffusers import StableDiffusionPipeline

        token = self.config.hf_token
        try:
            print(f"[ImageGen] Trying model: {model_id}")
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                token=token,
            )
            print(f"[ImageGen] Loaded: {model_id}")
            return pipeline
        except OSError as e:
            error_msg = str(e)
            if "401" in error_msg or "403" in error_msg:
                print(f"[ImageGen] Auth required for {model_id}. "
                      f"Run 'huggingface-cli login' or set HF_TOKEN env var.")
            elif "404" in error_msg or "Repository Not Found" in error_msg:
                print(f"[ImageGen] Model {model_id} not found on HuggingFace, skipping.")
            else:
                print(f"[ImageGen] Failed to load {model_id}: {e}")
            return None
        except Exception as e:
            print(f"[ImageGen] Failed to load {model_id}: {e}")
            return None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        from diffusers import DPMSolverMultistepScheduler

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map[self.config.dtype]

        print(f"[ImageGen] Loading Stable Diffusion ({self.config.device}, {self.config.dtype})")

        # Try the configured model first, then fallbacks
        models_to_try = [self.config.model_id]
        for m in FALLBACK_MODELS:
            if m != self.config.model_id:
                models_to_try.append(m)

        pipeline = None
        for model_id in models_to_try:
            pipeline = self._try_load_model(model_id, torch_dtype)
            if pipeline is not None:
                break

        if pipeline is None:
            raise RuntimeError(
                "Could not load any Stable Diffusion model. Please:\n"
                "  1. Run: huggingface-cli login\n"
                "  2. Accept the model license at https://huggingface.co/stabilityai/stable-diffusion-2-1\n"
                "  3. Or set HF_TOKEN=<your-token> environment variable\n"
                f"  Tried: {', '.join(models_to_try)}"
            )

        self._pipeline = pipeline

        # Use DPM-Solver++ for faster inference (20-30 steps instead of 50)
        self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipeline.scheduler.config
        )

        # Optional: disable safety checker for unrestricted generation
        if not self.config.safety_checker:
            self._pipeline.safety_checker = None

        # Load LoRA weights if specified
        if self.config.lora_weights:
            lora_path = Path(self.config.lora_weights)
            if lora_path.exists():
                print(f"[ImageGen] Loading LoRA weights: {self.config.lora_weights}")
                self._pipeline.load_lora_weights(self.config.lora_weights)

        self._pipeline.to(self.config.device)

        # Enable memory optimizations
        self._pipeline.enable_attention_slicing()
        if self.config.device == "cuda":
            try:
                self._pipeline.enable_xformers_memory_efficient_attention()
                print("[ImageGen] xformers memory-efficient attention enabled")
            except Exception:
                pass  # xformers not installed, slicing is sufficient

        return self._pipeline

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "blurry, bad quality, distorted, deformed",
        width: int | None = None,
        height: int | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        num_images: int = 1,
    ):
        """Generate images from a text prompt.

        Args:
            prompt: Text description of the desired image.
            negative_prompt: What to avoid in the generated image.
            width: Image width (must be divisible by 8).
            height: Image height (must be divisible by 8).
            num_steps: Number of denoising steps (more = better quality, slower).
            guidance_scale: How closely to follow the prompt (higher = more literal).
            seed: Random seed for reproducibility.
            num_images: Number of images to generate.

        Returns:
            PIL Image (or list of PIL Images if num_images > 1).
        """
        pipeline = self._load_pipeline()

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.config.device).manual_seed(seed)

        result = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width or self.config.width,
            height=height or self.config.height,
            num_inference_steps=num_steps or self.config.num_inference_steps,
            guidance_scale=guidance_scale or self.config.guidance_scale,
            generator=generator,
            num_images_per_prompt=num_images,
        )

        images = result.images
        return images[0] if num_images == 1 else images

    def generate_variations(self, image, prompt: str, strength: float = 0.75, **kwargs):
        """Generate variations of an existing image (img2img).

        Args:
            image: PIL Image to use as the starting point.
            prompt: Text description guiding the variation.
            strength: How much to change the image (0.0 = identical, 1.0 = complete redo).
        """
        from diffusers import StableDiffusionImg2ImgPipeline

        # Load img2img pipeline (shares weights with txt2img)
        pipeline = StableDiffusionImg2ImgPipeline(**self._load_pipeline().components)
        pipeline.to(self.config.device)

        result = pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            **kwargs,
        )
        return result.images[0]

    def unload(self) -> None:
        """Free GPU memory by unloading the model."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            torch.cuda.empty_cache()


class LoRAFineTuner:
    """Fine-tune Stable Diffusion with LoRA for custom styles.

    Usage:
        tuner = LoRAFineTuner(
            base_model="stabilityai/stable-diffusion-2-1",
            dataset_dir="./my_images",
            output_dir="./checkpoints/sd-lora",
        )
        tuner.train()

    Dataset structure (for LoRA):
        dataset_dir/
        ├── image1.png
        ├── image1.txt       (caption for image1)
        ├── image2.png
        ├── image2.txt
        └── ...

    For DreamBooth (subject-specific):
        dataset_dir/
        ├── 01.png
        ├── 02.png
        └── ...   (5-20 images of the subject, no captions needed)
    """

    def __init__(
        self,
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        dataset_dir: str = "./data/images",
        output_dir: str = "./checkpoints/sd-lora",
        resolution: int = 512,
        train_batch_size: int = 1,
        max_train_steps: int = 1000,
        learning_rate: float = 1e-4,
        lora_rank: int = 8,
    ):
        self.base_model = base_model
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.resolution = resolution
        self.train_batch_size = train_batch_size
        self.max_train_steps = max_train_steps
        self.learning_rate = learning_rate
        self.lora_rank = lora_rank

    def get_training_command(self) -> list[str]:
        """Generate the training command using diffusers' training script.

        This uses the official LoRA training script from the diffusers library.
        """
        return [
            "accelerate", "launch",
            "diffusers/examples/text_to_image/train_text_to_image_lora.py",
            f"--pretrained_model_name_or_path={self.base_model}",
            f"--train_data_dir={self.dataset_dir}",
            f"--output_dir={self.output_dir}",
            f"--resolution={self.resolution}",
            f"--train_batch_size={self.train_batch_size}",
            f"--max_train_steps={self.max_train_steps}",
            f"--learning_rate={self.learning_rate}",
            f"--rank={self.lora_rank}",
            "--mixed_precision=fp16",
            "--gradient_accumulation_steps=4",
            "--lr_scheduler=cosine",
            "--lr_warmup_steps=100",
            "--checkpointing_steps=500",
            "--validation_prompt=a test image in the trained style",
        ]

    def get_dreambooth_command(self, instance_prompt: str, class_prompt: str = "") -> list[str]:
        """Generate the DreamBooth training command.

        DreamBooth learns to associate a unique token with your subject.
        Needs only 5-20 images, no captions required.

        Args:
            instance_prompt: Prompt with unique token, e.g., "a photo of sks dog"
            class_prompt: Class description, e.g., "a photo of a dog"
        """
        cmd = [
            "accelerate", "launch",
            "diffusers/examples/dreambooth/train_dreambooth_lora.py",
            f"--pretrained_model_name_or_path={self.base_model}",
            f"--instance_data_dir={self.dataset_dir}",
            f"--output_dir={self.output_dir}",
            f"--instance_prompt={instance_prompt}",
            f"--resolution={self.resolution}",
            f"--train_batch_size={self.train_batch_size}",
            f"--max_train_steps={self.max_train_steps}",
            f"--learning_rate={self.learning_rate}",
            "--mixed_precision=fp16",
            "--gradient_accumulation_steps=1",
            "--lr_scheduler=constant",
            "--lr_warmup_steps=0",
        ]
        if class_prompt:
            cmd.extend([
                f"--class_prompt={class_prompt}",
                "--with_prior_preservation",
                "--prior_loss_weight=1.0",
            ])
        return cmd
