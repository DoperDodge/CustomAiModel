"""Tests for the Calculator tool and Tool Dispatch system."""

import pytest

from src.tools.calculator import Calculator, CalculatorResult
from src.tools.dispatch import (
    ToolDispatcher,
    ToolRegistry,
    create_default_dispatcher,
)


# ──────────────────────────────────────────────
# Calculator — Basic Arithmetic
# ──────────────────────────────────────────────

class TestCalculatorArithmetic:
    def setup_method(self):
        self.calc = Calculator()

    def test_addition(self):
        r = self.calc.evaluate("2 + 3")
        assert r.success
        assert r.result == 5

    def test_subtraction(self):
        r = self.calc.evaluate("10 - 4")
        assert r.success
        assert r.result == 6

    def test_multiplication(self):
        r = self.calc.evaluate("6 * 7")
        assert r.success
        assert r.result == 42

    def test_division(self):
        r = self.calc.evaluate("15 / 4")
        assert r.success
        assert r.result == 3.75

    def test_floor_division(self):
        r = self.calc.evaluate("15 // 4")
        assert r.success
        assert r.result == 3

    def test_modulo(self):
        r = self.calc.evaluate("17 % 5")
        assert r.success
        assert r.result == 2

    def test_power(self):
        r = self.calc.evaluate("2 ** 10")
        assert r.success
        assert r.result == 1024

    def test_order_of_operations(self):
        r = self.calc.evaluate("2 + 3 * 4")
        assert r.success
        assert r.result == 14

    def test_parentheses(self):
        r = self.calc.evaluate("(2 + 3) * 4")
        assert r.success
        assert r.result == 20

    def test_negative_numbers(self):
        r = self.calc.evaluate("-5 + 3")
        assert r.success
        assert r.result == -2

    def test_float_result(self):
        r = self.calc.evaluate("1 / 3")
        assert r.success
        assert abs(r.result - 0.3333333333) < 0.0001

    def test_nested_expression(self):
        r = self.calc.evaluate("(10 + 5) * (3 - 1) / 2")
        assert r.success
        assert r.result == 15


# ──────────────────────────────────────────────
# Calculator — Math Functions
# ──────────────────────────────────────────────

class TestCalculatorFunctions:
    def setup_method(self):
        self.calc = Calculator()

    def test_sqrt(self):
        r = self.calc.evaluate("sqrt(144)")
        assert r.success
        assert r.result == 12

    def test_sin_pi_half(self):
        r = self.calc.evaluate("sin(pi / 2)")
        assert r.success
        assert abs(r.result - 1.0) < 1e-10

    def test_cos_zero(self):
        r = self.calc.evaluate("cos(0)")
        assert r.success
        assert r.result == 1

    def test_log_natural(self):
        r = self.calc.evaluate("log(e)")
        assert r.success
        assert abs(r.result - 1.0) < 1e-10

    def test_log10(self):
        r = self.calc.evaluate("log10(1000)")
        assert r.success
        assert r.result == 3

    def test_abs(self):
        r = self.calc.evaluate("abs(-42)")
        assert r.success
        assert r.result == 42

    def test_floor(self):
        r = self.calc.evaluate("floor(3.7)")
        assert r.success
        assert r.result == 3

    def test_ceil(self):
        r = self.calc.evaluate("ceil(3.2)")
        assert r.success
        assert r.result == 4

    def test_factorial(self):
        r = self.calc.evaluate("factorial(5)")
        assert r.success
        assert r.result == 120

    def test_nested_functions(self):
        r = self.calc.evaluate("sqrt(2**2 + 3**2)")
        assert r.success
        assert abs(r.result - 3.605551275) < 0.001

    def test_pi_constant(self):
        r = self.calc.evaluate("pi")
        assert r.success
        assert abs(r.result - 3.14159265) < 0.0001

    def test_e_constant(self):
        r = self.calc.evaluate("e")
        assert r.success
        assert abs(r.result - 2.71828182) < 0.0001


# ──────────────────────────────────────────────
# Calculator — Error Handling & Security
# ──────────────────────────────────────────────

class TestCalculatorSafety:
    def setup_method(self):
        self.calc = Calculator()

    def test_division_by_zero(self):
        r = self.calc.evaluate("1 / 0")
        assert not r.success
        assert r.error is not None

    def test_empty_expression(self):
        r = self.calc.evaluate("")
        assert not r.success

    def test_invalid_syntax(self):
        r = self.calc.evaluate("2 +")
        assert not r.success

    def test_rejects_variables(self):
        r = self.calc.evaluate("x + 1")
        assert not r.success
        assert "unknown variable" in r.error.lower() or "unsupported" in r.error.lower()

    def test_rejects_import(self):
        r = self.calc.evaluate("__import__('os').system('ls')")
        assert not r.success

    def test_rejects_eval(self):
        r = self.calc.evaluate("eval('1+1')")
        assert not r.success

    def test_rejects_string_literal(self):
        r = self.calc.evaluate("'hello'")
        assert not r.success

    def test_huge_exponent_blocked(self):
        r = self.calc.evaluate("2 ** 999999")
        assert not r.success
        assert "exponent" in r.error.lower()

    def test_expression_too_long(self):
        r = self.calc.evaluate("1 + " * 200)
        assert not r.success
        assert "too long" in r.error.lower()

    def test_unknown_function(self):
        r = self.calc.evaluate("system('ls')")
        assert not r.success


# ──────────────────────────────────────────────
# Tool Dispatch — Detection & Execution
# ──────────────────────────────────────────────

class TestToolDispatch:
    def setup_method(self):
        self.dispatcher = create_default_dispatcher()

    def test_detect_single_tool_call(self):
        text = "Let me calculate: [TOOL: calculator(2 + 3)]"
        calls = self.dispatcher.detect_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].tool_name == "calculator"
        assert calls[0].arguments == "2 + 3"

    def test_detect_multiple_tool_calls(self):
        text = "First [TOOL: calculator(10 * 5)] then [TOOL: calculator(sqrt(9))]"
        calls = self.dispatcher.detect_tool_calls(text)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        text = "Just a regular response with no tools."
        calls = self.dispatcher.detect_tool_calls(text)
        assert len(calls) == 0

    def test_process_replaces_tool_tag(self):
        text = "The answer is [TOOL: calculator(2 + 3)]."
        processed, results = self.dispatcher.process(text)
        assert "[TOOL:" not in processed
        assert "5" in processed
        assert len(results) == 1
        assert results[0].success

    def test_process_no_tools(self):
        text = "Hello, how are you?"
        processed, results = self.dispatcher.process(text)
        assert processed == text
        assert results == []

    def test_process_unknown_tool(self):
        text = "[TOOL: nonexistent(arg)]"
        processed, results = self.dispatcher.process(text)
        assert "Unknown tool" in processed
        assert not results[0].success

    def test_process_calculator_error(self):
        text = "[TOOL: calculator(1/0)]"
        processed, results = self.dispatcher.process(text)
        assert "Error" in processed or "error" in processed

    def test_process_complex_expression(self):
        text = "The hypotenuse is [TOOL: calculator(sqrt(3**2 + 4**2))]"
        processed, results = self.dispatcher.process(text)
        assert "5" in processed
        assert results[0].success

    def test_process_percentage_calculation(self):
        text = "15% of 200 is [TOOL: calculator(15 * 200 / 100)]"
        processed, results = self.dispatcher.process(text)
        assert "30" in processed


# ──────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────

class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        registry.register("test", lambda x: x, "A test tool")
        assert registry.get("test") is not None
        assert "test" in registry.names

    def test_unknown_tool(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_system_prompt_section(self):
        registry = ToolRegistry()
        registry.register(
            "calculator",
            lambda x: x,
            "Math evaluator",
            examples=["[TOOL: calculator(1+1)]"],
        )
        section = registry.system_prompt_section()
        assert "calculator" in section
        assert "Math evaluator" in section
        assert "[TOOL: calculator(1+1)]" in section

    def test_empty_registry_prompt(self):
        registry = ToolRegistry()
        assert registry.system_prompt_section() == ""
