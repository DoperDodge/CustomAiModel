"""Tool Dispatch — Detect and execute tool calls in LLM output.

Strategy:
    1. Inject a tool-use instruction into the system prompt
    2. The LLM emits a special tag: [TOOL: tool_name(args)]
    3. This module detects the tag, executes the tool, and formats the result
    4. The result is appended to the conversation and the LLM continues

This is a prompt-engineering approach that works with ANY model (no function-calling
fine-tuning needed). The model learns the format from the system prompt examples.

Tag format:
    [TOOL: calculator(2 + 3 * 4)]
    [TOOL: calculator(sqrt(144) + 10)]
"""

import re
from dataclasses import dataclass, field

from src.tools.calculator import Calculator, CalculatorResult


# Pattern to detect tool calls in LLM output
# Matches: [TOOL: name(args)] or [TOOL:name(args)]
_TOOL_PATTERN = re.compile(
    r"\[TOOL:\s*(\w+)\((.+?)\)\]",
    re.DOTALL,
)


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM output."""
    tool_name: str
    arguments: str
    raw_match: str


@dataclass
class ToolResult:
    """Result of executing a tool call."""
    call: ToolCall
    output: str
    success: bool


@dataclass
class ToolRegistry:
    """Registry of available tools.

    Each tool has a name, a callable, and a description for the system prompt.
    """
    _tools: dict[str, dict] = field(default_factory=dict)

    def register(self, name: str, handler, description: str, examples: list[str] | None = None):
        """Register a tool."""
        self._tools[name] = {
            "handler": handler,
            "description": description,
            "examples": examples or [],
        }

    def get(self, name: str):
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def system_prompt_section(self) -> str:
        """Generate the tool-use instructions to inject into the system prompt."""
        if not self._tools:
            return ""

        lines = [
            "\n\n## Available Tools",
            "When you need to perform calculations or operations, use this exact format:",
            "[TOOL: tool_name(arguments)]",
            "",
            "The system will execute the tool and give you the result. Then continue your response using that result.",
            "",
        ]

        for name, info in self._tools.items():
            lines.append(f"### {name}")
            lines.append(info["description"])
            if info["examples"]:
                lines.append("Examples:")
                for ex in info["examples"]:
                    lines.append(f"  {ex}")
            lines.append("")

        lines.append("IMPORTANT: Only use tools when you need precise results. For simple questions, just answer directly.")
        return "\n".join(lines)


class ToolDispatcher:
    """Detects tool calls in LLM output and executes them.

    Usage:
        dispatcher = create_default_dispatcher()

        # Augment the system prompt
        system_prompt = base_prompt + dispatcher.registry.system_prompt_section()

        # After LLM generates a response, check for tool calls
        response_text = "Let me calculate that: [TOOL: calculator(15% of 200)]"
        processed, had_tools = dispatcher.process(response_text)
        # processed = "Let me calculate that: **15% of 200 = 30**"
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def detect_tool_calls(self, text: str) -> list[ToolCall]:
        """Find all tool call tags in text."""
        calls = []
        for match in _TOOL_PATTERN.finditer(text):
            calls.append(ToolCall(
                tool_name=match.group(1),
                arguments=match.group(2).strip(),
                raw_match=match.group(0),
            ))
        return calls

    def execute(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self.registry.get(call.tool_name)
        if tool is None:
            return ToolResult(
                call=call,
                output=f"Unknown tool: {call.tool_name}",
                success=False,
            )

        try:
            result = tool["handler"](call.arguments)
            return ToolResult(call=call, output=str(result), success=True)
        except Exception as e:
            return ToolResult(call=call, output=f"Error: {e}", success=False)

    def process(self, text: str) -> tuple[str, list[ToolResult]]:
        """Detect, execute, and replace all tool calls in the text.

        Returns:
            (processed_text, tool_results) — the text with tool tags replaced
            by formatted results, and the list of results for logging.
        """
        calls = self.detect_tool_calls(text)
        if not calls:
            return text, []

        results = []
        for call in calls:
            result = self.execute(call)
            results.append(result)

            if result.success:
                replacement = f"**{call.arguments} = {result.output}**"
            else:
                replacement = f"[Tool error: {result.output}]"

            text = text.replace(call.raw_match, replacement, 1)

        return text, results


def _calculator_handler(expression: str) -> str:
    """Handler that bridges ToolDispatcher to Calculator."""
    calc = Calculator()
    result = calc.evaluate(expression)
    if result.success:
        return _format_number(result.result)
    return f"Error: {result.error}"


def _format_number(value) -> str:
    """Format a number for display."""
    if isinstance(value, float):
        # Avoid ugly floating point: 2.0000000000000004 → 2.0
        if value == int(value) and abs(value) < 2**53:
            return str(int(value))
        # Round to 10 significant digits
        return f"{value:.10g}"
    return str(value)


def create_default_dispatcher() -> ToolDispatcher:
    """Create a ToolDispatcher with all default tools registered."""
    registry = ToolRegistry()

    registry.register(
        name="calculator",
        handler=_calculator_handler,
        description="Evaluate math expressions. Supports: +, -, *, /, **, sqrt, sin, cos, log, pi, e, etc.",
        examples=[
            "[TOOL: calculator(2 + 3 * 4)]",
            "[TOOL: calculator(sqrt(144))]",
            "[TOOL: calculator(sin(pi / 2))]",
            "[TOOL: calculator(15 * 200 / 100)]",
            "[TOOL: calculator(log(1000, 10))]",
        ],
    )

    return ToolDispatcher(registry)
