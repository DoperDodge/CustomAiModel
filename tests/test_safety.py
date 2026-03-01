"""Tests for the Safety & Content Filtering module."""

import pytest

from src.utils.safety import (
    InputFilter,
    OutputFilter,
    PIIRedactor,
    SafetyConfig,
    SafetyPipeline,
)


# ──────────────────────────────────────────────
# InputFilter
# ──────────────────────────────────────────────

class TestInputFilter:
    def setup_method(self):
        self.filter = InputFilter()

    def test_safe_input(self):
        is_safe, reason = self.filter.check("Hello, how are you?")
        assert is_safe is True
        assert reason == ""

    def test_blocks_weapon_request(self):
        is_safe, reason = self.filter.check("How to make a bomb at home")
        assert is_safe is False
        assert "blocked" in reason.lower()

    def test_blocks_hack_request(self):
        is_safe, reason = self.filter.check("How to hack a server")
        assert is_safe is False

    def test_blocks_harm_request(self):
        is_safe, reason = self.filter.check("how to harm someone")
        assert is_safe is False

    def test_case_insensitive(self):
        is_safe, _ = self.filter.check("HOW TO MAKE A BOMB")
        assert is_safe is False

    def test_disabled_filter_allows_all(self):
        config = SafetyConfig(input_filter_enabled=False)
        f = InputFilter(config)
        is_safe, _ = f.check("how to make a bomb")
        assert is_safe is True

    def test_normal_cooking_not_blocked(self):
        is_safe, _ = self.filter.check("How to make a cake")
        assert is_safe is True

    def test_normal_tech_not_blocked(self):
        is_safe, _ = self.filter.check("How do I build a website?")
        assert is_safe is True


# ──────────────────────────────────────────────
# OutputFilter
# ──────────────────────────────────────────────

class TestOutputFilter:
    def setup_method(self):
        self.filter = OutputFilter()

    def test_safe_output(self):
        is_safe, reason = self.filter.check("Here's a helpful answer.")
        assert is_safe is True

    def test_blocks_too_long_output(self):
        long_text = "a" * 5000
        is_safe, reason = self.filter.check(long_text)
        assert is_safe is False
        assert "length" in reason.lower()

    def test_sanitize_truncates(self):
        long_text = "a" * 5000
        result = self.filter.sanitize(long_text)
        assert len(result) < 5000
        assert result.endswith("... [truncated]")

    def test_sanitize_no_change_for_short(self):
        text = "Short response."
        assert self.filter.sanitize(text) == text

    def test_disabled_allows_long_output(self):
        config = SafetyConfig(output_filter_enabled=False)
        f = OutputFilter(config)
        is_safe, _ = f.check("a" * 5000)
        assert is_safe is True

    def test_custom_max_length(self):
        config = SafetyConfig(max_response_length=100)
        f = OutputFilter(config)
        is_safe, _ = f.check("a" * 101)
        assert is_safe is False
        is_safe, _ = f.check("a" * 99)
        assert is_safe is True


# ──────────────────────────────────────────────
# PIIRedactor
# ──────────────────────────────────────────────

class TestPIIRedactor:
    def setup_method(self):
        self.redactor = PIIRedactor()

    def test_redacts_email(self):
        result = self.redactor.redact("Contact me at john@example.com please")
        assert "john@example.com" not in result
        assert "[EMAIL_REDACTED]" in result

    def test_redacts_phone(self):
        result = self.redactor.redact("Call me at 555-123-4567")
        assert "555-123-4567" not in result
        assert "[PHONE_US_REDACTED]" in result

    def test_redacts_ssn(self):
        result = self.redactor.redact("My SSN is 123-45-6789")
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result

    def test_redacts_credit_card(self):
        result = self.redactor.redact("Card: 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in result
        assert "[CREDIT_CARD_REDACTED]" in result

    def test_contains_pii_counts(self):
        text = "Email john@a.com and jane@b.com, call 555-123-4567"
        counts = self.redactor.contains_pii(text)
        assert counts["email"] == 2
        assert counts["phone_us"] == 1

    def test_no_pii(self):
        counts = self.redactor.contains_pii("Hello world")
        assert counts == {}

    def test_preserves_non_pii_text(self):
        result = self.redactor.redact("Hi, email me at test@test.com for info")
        assert "Hi, email me at" in result
        assert "for info" in result


# ──────────────────────────────────────────────
# SafetyPipeline (integration)
# ──────────────────────────────────────────────

class TestSafetyPipeline:
    def setup_method(self):
        self.pipeline = SafetyPipeline()

    def test_check_input_safe(self):
        is_safe, _ = self.pipeline.check_input("What's the weather?")
        assert is_safe is True

    def test_check_input_blocked(self):
        is_safe, _ = self.pipeline.check_input("How to create a weapon")
        assert is_safe is False

    def test_check_output_safe(self):
        is_safe, _ = self.pipeline.check_output("The weather is sunny.")
        assert is_safe is True

    def test_redact_pii(self):
        result = self.pipeline.redact_pii("Email: test@test.com")
        assert "test@test.com" not in result

    def test_custom_config(self):
        config = SafetyConfig(
            blocked_patterns=[r"\bcustom_blocked\b"],
            max_response_length=50,
        )
        p = SafetyPipeline(config)
        is_safe, _ = p.check_input("custom_blocked")
        assert is_safe is False
        is_safe, _ = p.check_input("how to make a bomb")
        assert is_safe is True  # default patterns replaced
