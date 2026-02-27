"""
Safety & Content Filtering Utilities

Implements a three-layer safety system:
  1. Input filtering  — Block harmful prompts before they reach the model
  2. Model alignment  — (Handled by DPO/RLHF during training)
  3. Output filtering — Block harmful content in model responses

This is a starting point. For production, use a dedicated content
moderation model (e.g., OpenAI's moderation API, Llama Guard, etc.).
"""

import re
from dataclasses import dataclass, field


@dataclass
class SafetyConfig:
    """Configuration for the safety system."""

    # Enable/disable safety layers
    input_filter_enabled: bool = True
    output_filter_enabled: bool = True
    nsfw_image_filter_enabled: bool = True

    # Blocked keyword categories (customize for your use case)
    blocked_patterns: list[str] = field(default_factory=lambda: [
        # These are example patterns — customize based on your needs.
        # In production, use a classifier instead of keyword matching.
        r"\b(make|create|build)\b.*\b(bomb|weapon|explosive)\b",
        r"\b(hack|breach|exploit)\b.*\b(system|server|account)\b",
        r"\bhow to (harm|hurt|injure|kill)\b",
    ])

    # Maximum response length (prevents infinite generation)
    max_response_length: int = 4096

    # PII patterns to redact from training data
    pii_patterns: dict[str, str] = field(default_factory=lambda: {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone_us": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    })


class InputFilter:
    """Filter harmful input prompts before they reach the model."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.blocked_patterns
        ]

    def check(self, text: str) -> tuple[bool, str]:
        """Check if input text is safe.

        Returns:
            (is_safe, reason) — True if safe, False with explanation if blocked.
        """
        if not self.config.input_filter_enabled:
            return True, ""

        for pattern in self._compiled_patterns:
            if pattern.search(text):
                return False, f"Input blocked by safety filter (pattern: {pattern.pattern})"

        return True, ""


class OutputFilter:
    """Filter harmful content from model outputs."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    def check(self, text: str) -> tuple[bool, str]:
        """Check if output text is safe.

        Returns:
            (is_safe, reason) — True if safe, False with explanation if blocked.
        """
        if not self.config.output_filter_enabled:
            return True, ""

        # Length check
        if len(text) > self.config.max_response_length:
            return False, "Response exceeded maximum length"

        return True, ""

    def sanitize(self, text: str) -> str:
        """Remove or redact potentially harmful content from output."""
        # Truncate if too long
        if len(text) > self.config.max_response_length:
            text = text[:self.config.max_response_length] + "... [truncated]"
        return text


class PIIRedactor:
    """Redact personally identifiable information from text.

    Use this when preparing training data to remove PII.
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self._compiled = {
            name: re.compile(pattern) for name, pattern in self.config.pii_patterns.items()
        }

    def redact(self, text: str) -> str:
        """Replace PII with placeholder tokens.

        Example:
            "Email me at john@example.com" → "Email me at [EMAIL_REDACTED]"
        """
        for name, pattern in self._compiled.items():
            placeholder = f"[{name.upper()}_REDACTED]"
            text = pattern.sub(placeholder, text)
        return text

    def contains_pii(self, text: str) -> dict[str, int]:
        """Check how many PII instances of each type are in the text."""
        counts = {}
        for name, pattern in self._compiled.items():
            matches = pattern.findall(text)
            if matches:
                counts[name] = len(matches)
        return counts


class NSFWImageFilter:
    """Filter NSFW images from the image generation pipeline.

    Uses a lightweight classifier to detect inappropriate content.
    """

    def __init__(self):
        self._classifier = None

    def _load_classifier(self):
        if self._classifier is None:
            try:
                from transformers import pipeline
                self._classifier = pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection",
                    device="cpu",  # Runs on CPU to save GPU for generation
                )
            except ImportError:
                print("Warning: transformers not installed, NSFW filter disabled")
                return None
        return self._classifier

    def is_safe(self, image) -> tuple[bool, float]:
        """Check if an image is safe (not NSFW).

        Args:
            image: PIL Image to check.

        Returns:
            (is_safe, nsfw_score) — True if safe, with the NSFW confidence score.
        """
        classifier = self._load_classifier()
        if classifier is None:
            return True, 0.0

        results = classifier(image)
        nsfw_score = 0.0
        for result in results:
            if result["label"].lower() == "nsfw":
                nsfw_score = result["score"]

        return nsfw_score < 0.5, nsfw_score


class SafetyPipeline:
    """Combined safety pipeline for all modalities.

    Usage:
        safety = SafetyPipeline()

        # Check text input
        is_safe, reason = safety.check_input("user message here")

        # Check text output
        is_safe, reason = safety.check_output("model response here")

        # Check image
        is_safe, score = safety.check_image(pil_image)

        # Redact PII from training data
        clean_text = safety.redact_pii("Contact john@example.com")
    """

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self.input_filter = InputFilter(self.config)
        self.output_filter = OutputFilter(self.config)
        self.pii_redactor = PIIRedactor(self.config)
        self.nsfw_filter = NSFWImageFilter()

    def check_input(self, text: str) -> tuple[bool, str]:
        return self.input_filter.check(text)

    def check_output(self, text: str) -> tuple[bool, str]:
        return self.output_filter.check(text)

    def check_image(self, image) -> tuple[bool, float]:
        return self.nsfw_filter.is_safe(image)

    def redact_pii(self, text: str) -> str:
        return self.pii_redactor.redact(text)
