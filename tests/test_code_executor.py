"""Tests for the Code Execution tool and its dispatch integration."""

import pytest

from src.tools.code_executor import CodeExecutor
from src.tools.dispatch import create_default_dispatcher


# ──────────────────────────────────────────────
# CodeExecutor — Basic Execution
# ──────────────────────────────────────────────

class TestCodeExecutorBasic:
    def setup_method(self):
        self.executor = CodeExecutor()

    def test_print_string(self):
        r = self.executor.run('print("hello world")')
        assert r.success
        assert "hello world" in r.stdout

    def test_print_number(self):
        r = self.executor.run("print(42)")
        assert r.success
        assert "42" in r.stdout

    def test_arithmetic(self):
        r = self.executor.run("print(2 + 3 * 4)")
        assert r.success
        assert "14" in r.stdout

    def test_list_comprehension(self):
        r = self.executor.run("print([x**2 for x in range(5)])")
        assert r.success
        assert "[0, 1, 4, 9, 16]" in r.stdout

    def test_multiline_code(self):
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
        r = self.executor.run(code)
        assert r.success
        assert "120" in r.stdout

    def test_for_loop(self):
        code = """
result = 0
for i in range(10):
    result += i
print(result)
"""
        r = self.executor.run(code)
        assert r.success
        assert "45" in r.stdout

    def test_dict_operations(self):
        code = """
d = {"a": 1, "b": 2, "c": 3}
print(sum(d.values()))
"""
        r = self.executor.run(code)
        assert r.success
        assert "6" in r.stdout

    def test_string_methods(self):
        r = self.executor.run('print("Hello World".lower().split())')
        assert r.success
        assert "['hello', 'world']" in r.stdout

    def test_no_output(self):
        r = self.executor.run("x = 42")
        assert r.success
        assert r.stdout.strip() == ""

    def test_empty_code(self):
        r = self.executor.run("")
        assert not r.success
        assert "empty" in r.error.lower()


# ──────────────────────────────────────────────
# CodeExecutor — Allowed Imports
# ──────────────────────────────────────────────

class TestCodeExecutorImports:
    def setup_method(self):
        self.executor = CodeExecutor()

    def test_import_math(self):
        r = self.executor.run("import math\nprint(math.sqrt(144))")
        assert r.success
        assert "12" in r.stdout

    def test_import_json(self):
        code = """
import json
data = {"name": "test", "value": 42}
print(json.dumps(data))
"""
        r = self.executor.run(code)
        assert r.success
        assert '"name"' in r.stdout

    def test_import_collections(self):
        code = """
from collections import Counter
print(Counter("abracadabra").most_common(3))
"""
        r = self.executor.run(code)
        assert r.success
        assert "a" in r.stdout

    def test_import_itertools(self):
        code = """
import itertools
print(list(itertools.combinations([1,2,3], 2)))
"""
        r = self.executor.run(code)
        assert r.success
        assert "(1, 2)" in r.stdout

    def test_import_re(self):
        code = """
import re
matches = re.findall(r'\\d+', 'abc 123 def 456')
print(matches)
"""
        r = self.executor.run(code)
        assert r.success
        assert "123" in r.stdout

    def test_import_random(self):
        code = """
import random
random.seed(42)
print(random.randint(1, 100))
"""
        r = self.executor.run(code)
        assert r.success
        assert r.stdout.strip().isdigit()

    def test_import_datetime(self):
        code = """
from datetime import datetime
dt = datetime(2024, 1, 15)
print(dt.strftime('%Y-%m-%d'))
"""
        r = self.executor.run(code)
        assert r.success
        assert "2024-01-15" in r.stdout

    def test_import_statistics(self):
        code = """
import statistics
data = [1, 2, 3, 4, 5]
print(statistics.mean(data))
"""
        r = self.executor.run(code)
        assert r.success
        assert "3" in r.stdout


# ──────────────────────────────────────────────
# CodeExecutor — Security: Blocked Operations
# ──────────────────────────────────────────────

class TestCodeExecutorSecurity:
    def setup_method(self):
        self.executor = CodeExecutor()

    def test_blocks_os_import(self):
        r = self.executor.run("import os\nos.system('ls')")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_subprocess(self):
        r = self.executor.run("import subprocess\nsubprocess.run(['ls'])")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_open_file(self):
        r = self.executor.run("open('/etc/passwd').read()")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_eval(self):
        r = self.executor.run("eval('1+1')")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_exec(self):
        r = self.executor.run("exec('print(1)')")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_dunder_import(self):
        r = self.executor.run("__import__('os')")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_sys_import(self):
        r = self.executor.run("import sys\nsys.exit(1)")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_socket_import(self):
        r = self.executor.run("import socket")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_blocks_shutil_import(self):
        r = self.executor.run("import shutil")
        assert not r.success
        assert "not allowed" in r.error.lower()

    def test_code_too_long(self):
        r = self.executor.run("x = 1\n" * 500)
        assert not r.success
        assert "too long" in r.error.lower()

    def test_syntax_error(self):
        r = self.executor.run("def foo(:")
        assert not r.success
        assert "syntax" in r.error.lower()

    def test_blocks_input(self):
        r = self.executor.run("input('Enter: ')")
        assert not r.success
        assert "not allowed" in r.error.lower()


# ──────────────────────────────────────────────
# CodeExecutor — Timeout
# ──────────────────────────────────────────────

class TestCodeExecutorTimeout:
    def setup_method(self):
        self.executor = CodeExecutor()

    def test_infinite_loop_times_out(self):
        r = self.executor.run("while True: pass")
        assert not r.success
        assert r.timed_out


# ──────────────────────────────────────────────
# CodeExecutor — Runtime Errors
# ──────────────────────────────────────────────

class TestCodeExecutorRuntimeErrors:
    def setup_method(self):
        self.executor = CodeExecutor()

    def test_division_by_zero(self):
        r = self.executor.run("print(1/0)")
        assert not r.success
        assert "ZeroDivision" in r.error

    def test_name_error(self):
        r = self.executor.run("print(undefined_variable)")
        assert not r.success
        assert "NameError" in r.error

    def test_type_error(self):
        r = self.executor.run("print('hello' + 5)")
        assert not r.success
        assert "TypeError" in r.error

    def test_index_error(self):
        r = self.executor.run("lst = [1,2,3]\nprint(lst[10])")
        assert not r.success
        assert "IndexError" in r.error


# ──────────────────────────────────────────────
# Dispatch Integration — [CODE] tags
# ──────────────────────────────────────────────

class TestCodeDispatch:
    def setup_method(self):
        self.dispatcher = create_default_dispatcher()

    def test_detect_code_block(self):
        text = "Here's the code:\n[CODE]\nprint('hello')\n[/CODE]"
        calls = self.dispatcher.detect_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].tool_name == "python"
        assert "print('hello')" in calls[0].arguments

    def test_process_code_block(self):
        text = "Let me run that:\n[CODE]\nprint(2 + 2)\n[/CODE]"
        processed, results = self.dispatcher.process(text)
        assert "[CODE]" not in processed
        assert "4" in processed
        assert len(results) == 1
        assert results[0].success

    def test_process_code_with_error(self):
        text = "[CODE]\nimport os\n[/CODE]"
        processed, results = self.dispatcher.process(text)
        assert "Error" in processed or "error" in processed
        # The handler catches the error and returns it as a string,
        # so the dispatch layer sees it as a successful handler call
        assert "not allowed" in results[0].output.lower()

    def test_mixed_tools(self):
        text = "Calculate [TOOL: calculator(5 * 5)] and run [CODE]\nprint('hi')\n[/CODE]"
        processed, results = self.dispatcher.process(text)
        assert "25" in processed
        assert "hi" in processed
        assert len(results) == 2

    def test_no_code_blocks(self):
        text = "Just regular text with no code."
        processed, results = self.dispatcher.process(text)
        assert processed == text
        assert results == []

    def test_multiline_code_block(self):
        text = """[CODE]
def greet(name):
    return f"Hello, {name}!"

print(greet("World"))
[/CODE]"""
        processed, results = self.dispatcher.process(text)
        assert "Hello, World!" in processed
        assert results[0].success
