"""Tests for the Chain-of-Thought prompting module (Phase 5, Milestone 1)."""

import pytest

from src.text_to_text.chain_of_thought import (
    ThinkingConfig,
    ThinkingDepth,
    ThinkingMode,
    ThinkingParser,
    ThinkingResult,
)


# ──────────────────────────────────────────────
# ThinkingParser — Basic Parsing
# ──────────────────────────────────────────────

class TestThinkingParserBasic:
    def setup_method(self):
        self.parser = ThinkingParser()

    def test_parse_with_thinking_tags(self):
        text = "<think>\nStep 1: Add 2+3=5\nStep 2: Multiply 5*4=20\n</think>\n\nThe answer is 20."
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "Step 1" in result.thinking
        assert "Step 2" in result.thinking
        assert result.answer == "The answer is 20."

    def test_parse_without_thinking_tags(self):
        text = "The answer is simply 42."
        result = self.parser.parse(text)
        assert not result.has_thinking
        assert result.thinking == ""
        assert result.answer == "The answer is simply 42."

    def test_parse_empty_thinking(self):
        text = "<think>\n\n</think>\n\nJust the answer."
        result = self.parser.parse(text)
        assert result.has_thinking
        assert result.thinking == ""
        assert result.answer == "Just the answer."

    def test_parse_preserves_raw(self):
        text = "<think>reasoning</think>\n\nfinal answer"
        result = self.parser.parse(text)
        assert result.raw == text

    def test_parse_multiline_thinking(self):
        text = (
            "<think>\n"
            "First, I need to understand the problem.\n"
            "Then, I'll break it down:\n"
            "- Part A: Calculate X\n"
            "- Part B: Calculate Y\n"
            "- Part C: Combine results\n"
            "X = 10, Y = 20\n"
            "Combined: X + Y = 30\n"
            "</think>\n\n"
            "The result is 30."
        )
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "Part A" in result.thinking
        assert "Part B" in result.thinking
        assert "Part C" in result.thinking
        assert "X + Y = 30" in result.thinking
        assert result.answer == "The result is 30."


# ──────────────────────────────────────────────
# ThinkingParser — Edge Cases
# ──────────────────────────────────────────────

class TestThinkingParserEdgeCases:
    def setup_method(self):
        self.parser = ThinkingParser()

    def test_empty_string(self):
        result = self.parser.parse("")
        assert not result.has_thinking
        assert result.answer == ""

    def test_only_thinking_no_answer(self):
        text = "<think>Some reasoning here</think>"
        result = self.parser.parse(text)
        assert result.has_thinking
        assert result.thinking == "Some reasoning here"
        assert result.answer == ""

    def test_text_before_thinking(self):
        text = "Let me think about this.\n\n<think>reasoning</think>\n\nFinal answer."
        result = self.parser.parse(text)
        assert result.has_thinking
        assert result.thinking == "reasoning"
        assert "Let me think about this." in result.answer
        assert "Final answer." in result.answer

    def test_thinking_with_special_chars(self):
        text = "<think>\n2^3 = 8\n√16 = 4\n3 < 5 && 5 > 3\n</think>\n\nThe math checks out."
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "2^3 = 8" in result.thinking
        assert result.answer == "The math checks out."

    def test_thinking_with_code_blocks(self):
        text = (
            "<think>\n"
            "Let me trace through the code:\n"
            "```python\n"
            "x = 5\n"
            "y = x * 2  # y = 10\n"
            "```\n"
            "So y equals 10.\n"
            "</think>\n\n"
            "The variable y is 10."
        )
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "x = 5" in result.thinking
        assert result.answer == "The variable y is 10."

    def test_thinking_with_newlines_in_answer(self):
        text = "<think>quick check</think>\n\nLine 1\nLine 2\nLine 3"
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "Line 1\nLine 2\nLine 3" in result.answer

    def test_whitespace_handling(self):
        text = "<think>   padded reasoning   </think>   padded answer   "
        result = self.parser.parse(text)
        assert result.has_thinking
        assert result.thinking == "padded reasoning"
        assert result.answer == "padded answer"

    def test_no_newline_between_think_and_answer(self):
        text = "<think>reasoning</think>answer immediately"
        result = self.parser.parse(text)
        assert result.has_thinking
        assert result.thinking == "reasoning"
        assert result.answer == "answer immediately"


# ──────────────────────────────────────────────
# ThinkingResult — to_dict
# ──────────────────────────────────────────────

class TestThinkingResultDict:
    def test_to_dict_with_thinking_shown(self):
        result = ThinkingResult(
            thinking="Step 1: ...",
            answer="42",
            raw="<think>Step 1: ...</think>\n\n42",
            has_thinking=True,
        )
        d = result.to_dict(include_thinking=True)
        assert d["answer"] == "42"
        assert d["thinking"] == "Step 1: ..."

    def test_to_dict_with_thinking_hidden(self):
        result = ThinkingResult(
            thinking="Step 1: ...",
            answer="42",
            raw="<think>Step 1: ...</think>\n\n42",
            has_thinking=True,
        )
        d = result.to_dict(include_thinking=False)
        assert d["answer"] == "42"
        assert "thinking" not in d

    def test_to_dict_no_thinking(self):
        result = ThinkingResult(
            thinking="",
            answer="42",
            raw="42",
            has_thinking=False,
        )
        d = result.to_dict(include_thinking=True)
        assert d["answer"] == "42"
        assert "thinking" not in d


# ──────────────────────────────────────────────
# ThinkingDepth — Enum
# ──────────────────────────────────────────────

class TestThinkingDepth:
    def test_brief_value(self):
        assert ThinkingDepth.BRIEF == "brief"

    def test_standard_value(self):
        assert ThinkingDepth.STANDARD == "standard"

    def test_thorough_value(self):
        assert ThinkingDepth.THOROUGH == "thorough"

    def test_from_string(self):
        assert ThinkingDepth("brief") == ThinkingDepth.BRIEF
        assert ThinkingDepth("standard") == ThinkingDepth.STANDARD
        assert ThinkingDepth("thorough") == ThinkingDepth.THOROUGH

    def test_invalid_depth_raises(self):
        with pytest.raises(ValueError):
            ThinkingDepth("invalid")


# ──────────────────────────────────────────────
# ThinkingConfig
# ──────────────────────────────────────────────

class TestThinkingConfig:
    def test_defaults(self):
        config = ThinkingConfig()
        assert config.depth == ThinkingDepth.STANDARD
        assert config.show_thinking is True
        assert config.max_thinking_tokens == 1024
        assert config.max_answer_tokens == 512

    def test_custom_config(self):
        config = ThinkingConfig(
            depth=ThinkingDepth.THOROUGH,
            show_thinking=False,
            max_thinking_tokens=2048,
            max_answer_tokens=1024,
        )
        assert config.depth == ThinkingDepth.THOROUGH
        assert config.show_thinking is False
        assert config.max_thinking_tokens == 2048
        assert config.max_answer_tokens == 1024


# ──────────────────────────────────────────────
# ThinkingMode — System Prompts
# ──────────────────────────────────────────────

class TestThinkingModePrompts:
    def test_brief_prompt_mentions_brief(self):
        mode = ThinkingMode(ThinkingConfig(depth=ThinkingDepth.BRIEF))
        prompt = mode.system_prompt
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "1-2 sentences" in prompt.lower() or "brief" in prompt.lower()

    def test_standard_prompt_has_steps(self):
        mode = ThinkingMode(ThinkingConfig(depth=ThinkingDepth.STANDARD))
        prompt = mode.system_prompt
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "Step 1" in prompt

    def test_thorough_prompt_mentions_verify(self):
        mode = ThinkingMode(ThinkingConfig(depth=ThinkingDepth.THOROUGH))
        prompt = mode.system_prompt
        assert "<think>" in prompt
        assert "</think>" in prompt
        assert "verify" in prompt.lower() or "Verify" in prompt

    def test_all_depths_have_unique_prompts(self):
        prompts = set()
        for depth in ThinkingDepth:
            mode = ThinkingMode(ThinkingConfig(depth=depth))
            prompts.add(mode.system_prompt)
        assert len(prompts) == 3


# ──────────────────────────────────────────────
# ThinkingMode — prepare_messages
# ──────────────────────────────────────────────

class TestThinkingModePrepareMessages:
    def setup_method(self):
        self.mode = ThinkingMode()

    def test_prepends_system_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = self.mode.prepare_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "<think>" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

    def test_replaces_existing_system_message(self):
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve x+2=5"},
        ]
        result = self.mode.prepare_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        # CoT prompt should be there
        assert "<think>" in result[0]["content"]
        # Original system content should be preserved
        assert "math tutor" in result[0]["content"]

    def test_appends_extra_system_content(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = self.mode.prepare_messages(messages, extra_system_content="## Tools\nUse [TOOL: calc(expr)]")
        assert len(result) == 2
        assert "## Tools" in result[0]["content"]
        assert "[TOOL: calc(expr)]" in result[0]["content"]

    def test_preserves_conversation_order(self):
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        result = self.mode.prepare_messages(messages)
        assert len(result) == 4  # system + 3 messages
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "What is 2+2?"
        assert result[2]["content"] == "4"
        assert result[3]["content"] == "And 3+3?"

    def test_multiple_system_messages_merged(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        result = self.mode.prepare_messages(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "Be helpful." in result[0]["content"]
        assert "Be concise." in result[0]["content"]

    def test_empty_messages(self):
        result = self.mode.prepare_messages([])
        assert len(result) == 1
        assert result[0]["role"] == "system"

    def test_extra_system_content_without_existing_system(self):
        messages = [{"role": "user", "content": "Hi"}]
        result = self.mode.prepare_messages(messages, extra_system_content="Extra context")
        assert "Extra context" in result[0]["content"]
        assert "<think>" in result[0]["content"]


# ──────────────────────────────────────────────
# ThinkingMode — parse_response
# ──────────────────────────────────────────────

class TestThinkingModeParseResponse:
    def setup_method(self):
        self.mode = ThinkingMode()

    def test_parse_valid_cot_response(self):
        text = "<think>\n2+2=4\n</think>\n\nThe answer is 4."
        result = self.mode.parse_response(text)
        assert result.has_thinking
        assert "2+2=4" in result.thinking
        assert "The answer is 4." in result.answer

    def test_parse_plain_response(self):
        text = "The answer is 4."
        result = self.mode.parse_response(text)
        assert not result.has_thinking
        assert result.answer == "The answer is 4."


# ──────────────────────────────────────────────
# ThinkingMode — total_max_tokens
# ──────────────────────────────────────────────

class TestThinkingModeTokens:
    def test_default_total(self):
        mode = ThinkingMode()
        assert mode.total_max_tokens() == 1024 + 512

    def test_custom_total(self):
        config = ThinkingConfig(max_thinking_tokens=500, max_answer_tokens=200)
        mode = ThinkingMode(config)
        assert mode.total_max_tokens() == 700


# ──────────────────────────────────────────────
# ThinkingMode — format_for_display
# ──────────────────────────────────────────────

class TestThinkingModeDisplay:
    def test_display_with_thinking_shown(self):
        mode = ThinkingMode(ThinkingConfig(show_thinking=True))
        result = ThinkingResult(
            thinking="Step 1: 2+2=4",
            answer="The answer is 4.",
            raw="...",
            has_thinking=True,
        )
        display = mode.format_for_display(result)
        assert "**Thinking:**" in display
        assert "Step 1: 2+2=4" in display
        assert "---" in display
        assert "The answer is 4." in display

    def test_display_with_thinking_hidden(self):
        mode = ThinkingMode(ThinkingConfig(show_thinking=False))
        result = ThinkingResult(
            thinking="Step 1: 2+2=4",
            answer="The answer is 4.",
            raw="...",
            has_thinking=True,
        )
        display = mode.format_for_display(result)
        assert "**Thinking:**" not in display
        assert display == "The answer is 4."

    def test_display_no_thinking(self):
        mode = ThinkingMode()
        result = ThinkingResult(
            thinking="",
            answer="Simple answer.",
            raw="Simple answer.",
            has_thinking=False,
        )
        display = mode.format_for_display(result)
        assert display == "Simple answer."


# ──────────────────────────────────────────────
# Integration: Realistic Model Outputs
# ──────────────────────────────────────────────

class TestRealisticOutputParsing:
    def setup_method(self):
        self.parser = ThinkingParser()

    def test_math_problem(self):
        text = (
            "<think>\n"
            "Step 1: I need to find 15% of 280.\n"
            "Step 2: 15% means 15/100 = 0.15\n"
            "Step 3: 0.15 × 280 = 42\n"
            "Step 4: Let me verify: 10% of 280 = 28, 5% = 14, 28+14 = 42. ✓\n"
            "</think>\n\n"
            "15% of 280 is **42**."
        )
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "0.15 × 280 = 42" in result.thinking
        assert "verify" in result.thinking.lower()
        assert "42" in result.answer

    def test_logic_problem(self):
        text = (
            "<think>\n"
            "The question asks: If all cats are animals, and Whiskers is a cat, is Whiskers an animal?\n"
            "This is a classic syllogism:\n"
            "- Premise 1: All cats are animals\n"
            "- Premise 2: Whiskers is a cat\n"
            "- Conclusion: Therefore, Whiskers is an animal\n"
            "This follows by universal instantiation.\n"
            "</think>\n\n"
            "Yes, Whiskers is an animal. Since all cats are animals and Whiskers is a cat, "
            "it logically follows that Whiskers is an animal."
        )
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "syllogism" in result.thinking
        assert "Whiskers is an animal" in result.answer

    def test_coding_problem(self):
        text = (
            "<think>\n"
            "The user wants to reverse a string in Python.\n"
            "Option 1: Use slicing — s[::-1]\n"
            "Option 2: Use reversed() — ''.join(reversed(s))\n"
            "Option 3: Loop — build reversed string character by character\n"
            "Option 1 is the most Pythonic and efficient.\n"
            "</think>\n\n"
            "You can reverse a string in Python using slicing:\n\n"
            "```python\nreversed_str = my_string[::-1]\n```"
        )
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "Option 1" in result.thinking
        assert "[::-1]" in result.answer

    def test_simple_question_no_thinking(self):
        text = "The capital of France is Paris."
        result = self.parser.parse(text)
        assert not result.has_thinking
        assert result.answer == "The capital of France is Paris."

    def test_partial_thinking_tag_no_close(self):
        text = "<think>Started reasoning but never finished"
        result = self.parser.parse(text)
        # No closing tag = not valid thinking
        assert not result.has_thinking
        assert "<think>" in result.answer

    def test_nested_angle_brackets_in_thinking(self):
        text = (
            "<think>\n"
            "Check if x > 5 and y < 10:\n"
            "x=7, so x > 5 is True\n"
            "y=3, so y < 10 is True\n"
            "Both conditions are True.\n"
            "</think>\n\n"
            "Both conditions are satisfied."
        )
        result = self.parser.parse(text)
        assert result.has_thinking
        assert "x > 5" in result.thinking
        assert result.answer == "Both conditions are satisfied."
