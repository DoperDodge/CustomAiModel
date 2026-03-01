"""Calculator Tool — Safe math evaluation for the LLM.

Allows the model to delegate arithmetic and math operations to a real
calculator instead of attempting mental math (which LLMs are bad at).

Security: Uses AST-based parsing — NO eval()/exec(). Only allows
numeric literals, arithmetic operators, and a curated set of math
functions.

Usage:
    calc = Calculator()
    result = calc.evaluate("2 + 3 * 4")        # 14
    result = calc.evaluate("sqrt(144)")          # 12.0
    result = calc.evaluate("sin(pi / 2)")        # 1.0
    result = calc.evaluate("2 ** 10")            # 1024
"""

import ast
import math
import operator
from dataclasses import dataclass


@dataclass
class CalculatorResult:
    """Result from a calculator evaluation."""
    expression: str
    result: float | int | None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.error is None


# Allowed binary operators
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Allowed unary operators
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed math functions (name → callable)
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "cbrt": math.cbrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "radians": math.radians,
    "degrees": math.degrees,
}

# Allowed constants
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

# Safety limits
_MAX_EXPRESSION_LENGTH = 500
_MAX_EXPONENT = 10000


class Calculator:
    """Safe math expression evaluator using AST parsing.

    Supports:
        - Basic arithmetic: +, -, *, /, //, %, **
        - Math functions: sqrt, sin, cos, log, etc.
        - Constants: pi, e, tau
        - Nested expressions: sqrt(2**2 + 3**2)
    """

    def evaluate(self, expression: str) -> CalculatorResult:
        """Evaluate a math expression safely.

        Args:
            expression: Math expression string (e.g. "2 + 3 * 4").

        Returns:
            CalculatorResult with the answer or an error message.
        """
        expression = expression.strip()

        if not expression:
            return CalculatorResult(expression=expression, result=None, error="Empty expression")

        if len(expression) > _MAX_EXPRESSION_LENGTH:
            return CalculatorResult(expression=expression, result=None, error="Expression too long")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return CalculatorResult(expression=expression, result=None, error=f"Invalid syntax: {e}")

        try:
            result = self._eval_node(tree.body)
        except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
            return CalculatorResult(expression=expression, result=None, error=str(e))
        except _UnsafeNode as e:
            return CalculatorResult(expression=expression, result=None, error=f"Unsupported: {e}")

        # Return int when possible for cleaner output
        if isinstance(result, float) and result.is_integer() and abs(result) < 2**53:
            result = int(result)

        return CalculatorResult(expression=expression, result=result)

    def _eval_node(self, node: ast.AST) -> float | int:
        """Recursively evaluate an AST node."""

        # Numeric literal: 42, 3.14
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value

        # Named constant: pi, e
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            raise _UnsafeNode(f"unknown variable '{node.id}'")

        # Unary operator: -x, +x
        if isinstance(node, ast.UnaryOp):
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise _UnsafeNode(f"operator {type(node.op).__name__}")
            return op_fn(self._eval_node(node.operand))

        # Binary operator: x + y, x ** y
        if isinstance(node, ast.BinOp):
            op_fn = _BINARY_OPS.get(type(node.op))
            if op_fn is None:
                raise _UnsafeNode(f"operator {type(node.op).__name__}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            # Guard against huge exponents
            if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)) and abs(right) > _MAX_EXPONENT:
                raise ValueError(f"Exponent too large: {right} (max {_MAX_EXPONENT})")
            return op_fn(left, right)

        # Function call: sqrt(4), log(100, 10)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise _UnsafeNode("only named functions allowed")
            fn_name = node.func.id
            if fn_name not in _FUNCTIONS:
                raise _UnsafeNode(f"function '{fn_name}'")
            args = [self._eval_node(arg) for arg in node.args]
            return _FUNCTIONS[fn_name](*args)

        raise _UnsafeNode(f"node type {type(node).__name__}")


class _UnsafeNode(Exception):
    """Raised when the AST contains a disallowed node."""
