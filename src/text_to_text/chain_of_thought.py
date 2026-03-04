"""Chain-of-Thought Prompting — Deep Thinking Mode (Phase 5, Milestone 1)

Forces the LLM to reason step-by-step before giving a final answer.
The model's internal reasoning ("thinking") is captured in <think>...</think>
tags and can be shown or hidden from the user.

Architecture:
    1. CoT system prompt instructs the model to reason inside <think> tags
    2. The model generates: <think>step-by-step reasoning</think> Final answer
    3. ThinkingParser extracts the thinking trace and final answer
    4. The API can return both, or only the final answer

Usage:
    from src.text_to_text.chain_of_thought import ThinkingMode

    thinker = ThinkingMode()
    messages = [{"role": "user", "content": "What is 15% of 280?"}]

    # Build messages with CoT system prompt
    cot_messages = thinker.prepare_messages(messages)

    # After generation, parse the output
    result = thinker.parse_response(raw_output)
    print(result.thinking)      # Step-by-step reasoning
    print(result.answer)        # Final concise answer
    print(result.has_thinking)  # True if model produced thinking
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class ThinkingDepth(str, Enum):
    """Controls how much reasoning the model should do."""
    BRIEF = "brief"       # 1-2 sentences of reasoning
    STANDARD = "standard" # Full step-by-step breakdown
    THOROUGH = "thorough"  # Exhaustive analysis with verification


# ──────────────────────────────────────────────
# System Prompt Templates
# ──────────────────────────────────────────────

_COT_SYSTEM_PROMPTS = {
    ThinkingDepth.BRIEF: (
        "You are a helpful AI assistant with step-by-step reasoning.\n\n"
        "Before answering, briefly reason through the problem inside <think> tags.\n"
        "Then give your final answer outside the tags.\n\n"
        "Format:\n"
        "<think>\n"
        "Brief reasoning here (1-2 sentences).\n"
        "</think>\n\n"
        "Your final answer here."
    ),
    ThinkingDepth.STANDARD: (
        "You are a helpful AI assistant with deep thinking capabilities.\n\n"
        "For every question, you MUST first reason step-by-step inside <think> tags "
        "before providing your final answer.\n\n"
        "Rules:\n"
        "1. Always start your response with <think>\n"
        "2. Break the problem into clear steps\n"
        "3. Show your work — calculations, logic, and intermediate conclusions\n"
        "4. Close with </think> and then give a clear, concise final answer\n"
        "5. The final answer should stand on its own without needing the thinking\n\n"
        "Format:\n"
        "<think>\n"
        "Step 1: [identify what we need to find]\n"
        "Step 2: [work through the logic]\n"
        "Step 3: [arrive at conclusion]\n"
        "</think>\n\n"
        "Your final answer here."
    ),
    ThinkingDepth.THOROUGH: (
        "You are a helpful AI assistant with deep analytical capabilities.\n\n"
        "For every question, you MUST perform exhaustive reasoning inside <think> tags "
        "before providing your final answer.\n\n"
        "Rules:\n"
        "1. Always start your response with <think>\n"
        "2. Break the problem into clear, numbered steps\n"
        "3. Show ALL work — calculations, logic chains, and intermediate results\n"
        "4. Consider edge cases and alternative approaches\n"
        "5. Verify your answer: re-check calculations or logic before concluding\n"
        "6. Close with </think> and then give a clear, concise final answer\n"
        "7. The final answer should stand on its own without needing the thinking\n\n"
        "Format:\n"
        "<think>\n"
        "Step 1: [understand the problem]\n"
        "Step 2: [plan the approach]\n"
        "Step 3: [execute step-by-step]\n"
        "Step 4: [verify the result]\n"
        "</think>\n\n"
        "Your final answer here."
    ),
}


# ──────────────────────────────────────────────
# Response Parsing
# ──────────────────────────────────────────────

# Regex to extract <think>...</think> blocks (supports multiline)
_THINK_PATTERN = re.compile(
    r"<think>\s*(.*?)\s*</think>",
    re.DOTALL,
)


@dataclass
class ThinkingResult:
    """Parsed result from a chain-of-thought response."""
    thinking: str          # The step-by-step reasoning (empty if none)
    answer: str            # The final answer
    raw: str               # The original unparsed response
    has_thinking: bool     # Whether thinking tags were found

    def to_dict(self, include_thinking: bool = True) -> dict:
        """Convert to a dict suitable for API responses.

        Args:
            include_thinking: If True, include the thinking trace.
        """
        result = {"answer": self.answer}
        if include_thinking and self.has_thinking:
            result["thinking"] = self.thinking
        return result


class ThinkingParser:
    """Extracts thinking traces and final answers from model output."""

    def parse(self, text: str) -> ThinkingResult:
        """Parse a model response that may contain <think>...</think> tags.

        Args:
            text: Raw model output, possibly containing thinking tags.

        Returns:
            ThinkingResult with separated thinking and answer.
        """
        match = _THINK_PATTERN.search(text)

        if not match:
            return ThinkingResult(
                thinking="",
                answer=text.strip(),
                raw=text,
                has_thinking=False,
            )

        thinking = match.group(1).strip()
        # Everything after the </think> tag is the final answer
        answer = text[match.end():].strip()

        # If the model put text before <think> too, include it in the answer
        prefix = text[:match.start()].strip()
        if prefix:
            answer = f"{prefix}\n\n{answer}" if answer else prefix

        return ThinkingResult(
            thinking=thinking,
            answer=answer,
            raw=text,
            has_thinking=True,
        )


# ──────────────────────────────────────────────
# Thinking Mode Orchestrator
# ──────────────────────────────────────────────

@dataclass
class ThinkingConfig:
    """Configuration for chain-of-thought prompting."""
    depth: ThinkingDepth = ThinkingDepth.STANDARD
    show_thinking: bool = True          # Include thinking in API response
    max_thinking_tokens: int = 1024     # Max tokens for the thinking phase
    max_answer_tokens: int = 512        # Max tokens for the final answer


class ThinkingMode:
    """Orchestrates chain-of-thought prompting.

    Prepares messages with CoT system prompts and parses responses
    to extract thinking traces.

    Usage:
        thinker = ThinkingMode()

        # Prepare messages
        messages = [{"role": "user", "content": "Solve: 3x + 7 = 22"}]
        cot_messages = thinker.prepare_messages(messages)

        # ... generate response with the model ...

        # Parse the response
        result = thinker.parse_response(raw_output)
        print(result.thinking)  # "Step 1: Subtract 7... Step 2: Divide by 3..."
        print(result.answer)    # "x = 5"
    """

    def __init__(self, config: ThinkingConfig | None = None):
        self.config = config or ThinkingConfig()
        self.parser = ThinkingParser()

    @property
    def system_prompt(self) -> str:
        """Get the CoT system prompt for the configured depth."""
        return _COT_SYSTEM_PROMPTS[self.config.depth]

    def prepare_messages(
        self,
        messages: list[dict],
        extra_system_content: str = "",
    ) -> list[dict]:
        """Prepare messages with the CoT system prompt injected.

        Replaces or prepends a system message with the CoT instructions.
        Any existing system message content is preserved and appended
        after the CoT instructions.

        Args:
            messages: List of chat messages [{"role": "...", "content": "..."}].
            extra_system_content: Additional content to append to the system
                prompt (e.g., tool instructions, RAG context).

        Returns:
            New message list with CoT system prompt at the front.
        """
        cot_prompt = self.system_prompt

        # Collect any existing system message content
        existing_system = []
        non_system = []
        for msg in messages:
            if msg["role"] == "system":
                existing_system.append(msg["content"])
            else:
                non_system.append(msg)

        # Build the combined system prompt
        parts = [cot_prompt]
        if existing_system:
            parts.extend(existing_system)
        if extra_system_content:
            parts.append(extra_system_content)

        system_msg = {"role": "system", "content": "\n\n".join(parts)}
        return [system_msg] + non_system

    def parse_response(self, text: str) -> ThinkingResult:
        """Parse a model response to extract thinking and answer."""
        return self.parser.parse(text)

    def total_max_tokens(self) -> int:
        """Total max tokens to request from the model (thinking + answer)."""
        return self.config.max_thinking_tokens + self.config.max_answer_tokens

    def format_for_display(self, result: ThinkingResult) -> str:
        """Format a ThinkingResult for human-readable display.

        Shows the thinking in a collapsible/dimmed section, then the answer.
        """
        if not result.has_thinking:
            return result.answer

        parts = []
        if self.config.show_thinking:
            parts.append(f"**Thinking:**\n{result.thinking}")
            parts.append("---")
        parts.append(result.answer)
        return "\n\n".join(parts)
