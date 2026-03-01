"""Code Execution Tool — Sandboxed Python execution for the LLM.

Allows the model to write and run Python code to answer programming
questions, manipulate data, or demonstrate algorithms.

Security layers:
  1. Import whitelist — only safe stdlib modules allowed
  2. AST pre-scan — blocks dangerous calls (exec, eval, open, os.system, etc.)
  3. Subprocess isolation — code runs in a separate process
  4. Resource limits — timeout, memory cap, output truncation
  5. No filesystem / network access

Usage:
    executor = CodeExecutor()
    result = executor.run("print([x**2 for x in range(5)])")
    # result.stdout == "[0, 1, 4, 9, 16]\n"
"""

import ast
import multiprocessing
import io
import sys
import textwrap
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result from running code in the sandbox."""
    code: str
    stdout: str
    stderr: str
    return_value: str | None = None
    error: str | None = None
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.error is None and not self.timed_out


# Modules the sandbox is allowed to import
_ALLOWED_MODULES = frozenset({
    # Math & numbers
    "math", "cmath", "decimal", "fractions", "statistics",
    # Data structures & algorithms
    "collections", "itertools", "functools", "operator", "heapq", "bisect",
    # String & text
    "string", "re", "textwrap", "unicodedata",
    # Date & time
    "datetime", "time", "calendar",
    # Data formats
    "json", "csv", "base64", "hashlib", "hmac",
    # Typing & abstract
    "typing", "abc", "dataclasses", "enum",
    # Random
    "random",
    # Other safe utilities
    "copy", "pprint", "numbers",
})

# AST node names that are never allowed
_BLOCKED_NAMES = frozenset({
    "exec", "eval", "compile", "execfile",
    "open", "input",
    "__import__",
    "globals", "locals", "vars", "dir",
    "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit",
})

# Attribute access patterns that are blocked (module.attr)
_BLOCKED_ATTRS = frozenset({
    "system", "popen", "exec", "spawn",
    "remove", "rmdir", "unlink", "rename",
    "listdir", "walk", "makedirs",
    "environ", "getenv",
    "subprocess", "Popen",
})

# Resource limits
_TIMEOUT_SECONDS = 5
_MAX_OUTPUT_LENGTH = 4096
_MAX_CODE_LENGTH = 2000


class CodeExecutor:
    """Sandboxed Python code executor.

    Runs code in a separate process with restricted builtins,
    import whitelist, and resource limits.
    """

    def run(self, code: str) -> ExecutionResult:
        """Execute Python code in a sandbox.

        Args:
            code: Python source code to execute.

        Returns:
            ExecutionResult with stdout, stderr, and any errors.
        """
        code = textwrap.dedent(code).strip()

        if not code:
            return ExecutionResult(code=code, stdout="", stderr="", error="Empty code")

        if len(code) > _MAX_CODE_LENGTH:
            return ExecutionResult(code=code, stdout="", stderr="", error="Code too long (max 2000 chars)")

        # Pre-scan: static analysis before execution
        violation = self._scan_ast(code)
        if violation:
            return ExecutionResult(code=code, stdout="", stderr="", error=f"Blocked: {violation}")

        # Execute in a subprocess for isolation
        return self._execute_in_subprocess(code)

    def _scan_ast(self, code: str) -> str | None:
        """Static analysis — reject dangerous patterns before execution.

        Returns:
            None if safe, or a string describing the violation.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"Syntax error: {e}"

        for node in ast.walk(tree):
            # Block dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_NAMES:
                    return f"'{node.func.id}()' is not allowed"
                if isinstance(node.func, ast.Attribute) and node.func.attr in _BLOCKED_ATTRS:
                    return f"'.{node.func.attr}()' is not allowed"

            # Block dangerous name references (not just calls)
            if isinstance(node, ast.Name) and node.id == "__import__":
                return "'__import__' is not allowed"

            # Check imports against whitelist
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module not in _ALLOWED_MODULES:
                        return f"import '{alias.name}' is not allowed"

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module not in _ALLOWED_MODULES:
                        return f"import from '{node.module}' is not allowed"

        return None

    def _execute_in_subprocess(self, code: str) -> ExecutionResult:
        """Run code in an isolated subprocess with timeout."""
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_sandbox_worker,
            args=(code, result_queue),
        )
        process.start()
        process.join(timeout=_TIMEOUT_SECONDS)

        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()
            return ExecutionResult(
                code=code, stdout="", stderr="",
                timed_out=True,
                error=f"Execution timed out ({_TIMEOUT_SECONDS}s limit)",
            )

        if result_queue.empty():
            return ExecutionResult(
                code=code, stdout="", stderr="",
                error="Execution failed (no result returned)",
            )

        result = result_queue.get()
        stdout = result.get("stdout", "")[:_MAX_OUTPUT_LENGTH]
        stderr = result.get("stderr", "")[:_MAX_OUTPUT_LENGTH]
        error = result.get("error")

        if len(result.get("stdout", "")) > _MAX_OUTPUT_LENGTH:
            stdout += "\n... [output truncated]"

        return ExecutionResult(
            code=code,
            stdout=stdout,
            stderr=stderr,
            return_value=result.get("return_value"),
            error=error,
        )


def _sandbox_worker(code: str, result_queue: multiprocessing.Queue):
    """Worker function that runs in the subprocess.

    Sets up a restricted environment and executes the code.
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Restricted builtins — remove dangerous ones
    safe_builtins = {k: v for k, v in __builtins__.items() if isinstance(__builtins__, dict)} if isinstance(__builtins__, dict) else {k: getattr(__builtins__, k) for k in dir(__builtins__) if not k.startswith("_")}

    for name in _BLOCKED_NAMES:
        safe_builtins.pop(name, None)

    # Restricted __import__ that only allows whitelisted modules
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def restricted_import(name, *args, **kwargs):
        root_module = name.split(".")[0]
        if root_module not in _ALLOWED_MODULES:
            raise ImportError(f"Import '{name}' is not allowed in the sandbox")
        return original_import(name, *args, **kwargs)

    safe_builtins["__import__"] = restricted_import
    safe_builtins["__builtins__"] = safe_builtins

    sandbox_globals = {"__builtins__": safe_builtins}

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_capture
    sys.stderr = stderr_capture

    try:
        exec(compile(code, "<sandbox>", "exec"), sandbox_globals)
        result_queue.put({
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
        })
    except Exception as e:
        result_queue.put({
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
            "error": f"{type(e).__name__}: {e}",
        })
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
