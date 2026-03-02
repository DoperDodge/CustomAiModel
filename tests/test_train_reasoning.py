"""Tests for reasoning fine-tuning — dataset formatters, mixing, and CLI."""

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset

from src.text_to_text.train_reasoning import (
    DATASET_REGISTRY,
    REASONING_SYSTEM_PROMPT,
    format_gsm8k,
    format_openorca,
    format_metamath,
    format_example_for_training,
    list_available_datasets,
    load_reasoning_dataset,
    mix_datasets,
)


# ──────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────

class TestDatasetRegistry:
    def test_registry_has_expected_datasets(self):
        assert "gsm8k" in DATASET_REGISTRY
        assert "openorca" in DATASET_REGISTRY
        assert "metamath" in DATASET_REGISTRY

    def test_registry_entries_have_required_fields(self):
        for name, info in DATASET_REGISTRY.items():
            assert "hf_id" in info, f"{name} missing hf_id"
            assert "split" in info, f"{name} missing split"
            assert "description" in info, f"{name} missing description"

    def test_list_available_datasets(self):
        names = list_available_datasets()
        assert isinstance(names, list)
        assert len(names) == 3
        assert "gsm8k" in names
        assert "openorca" in names
        assert "metamath" in names


# ──────────────────────────────────────────────
# GSM8K formatter
# ──────────────────────────────────────────────

class TestFormatGsm8k:
    def test_basic_formatting(self):
        example = {
            "question": "What is 2 + 2?",
            "answer": "2 + 2 = 4\n#### 4",
        }
        messages = format_gsm8k(example)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_question_preserved(self):
        example = {
            "question": "How many apples does Jane have?",
            "answer": "Jane has 5 apples.\n#### 5",
        }
        messages = format_gsm8k(example)
        assert messages[1]["content"] == "How many apples does Jane have?"

    def test_answer_formatting_with_separator(self):
        example = {
            "question": "What is 10 / 2?",
            "answer": "10 divided by 2 equals 5.\n#### 5",
        }
        messages = format_gsm8k(example)
        assert "**Answer: 5**" in messages[2]["content"]
        assert "10 divided by 2 equals 5." in messages[2]["content"]

    def test_answer_without_separator(self):
        example = {
            "question": "Simple question?",
            "answer": "The answer is 42.",
        }
        messages = format_gsm8k(example)
        assert messages[2]["content"] == "The answer is 42."

    def test_system_prompt_is_reasoning(self):
        example = {"question": "Q?", "answer": "A.\n#### A"}
        messages = format_gsm8k(example)
        assert messages[0]["content"] == REASONING_SYSTEM_PROMPT

    def test_multiline_reasoning(self):
        example = {
            "question": "A store has 50 apples. They sell 20. How many left?",
            "answer": "The store starts with 50 apples.\nThey sell 20 apples.\n50 - 20 = 30\n#### 30",
        }
        messages = format_gsm8k(example)
        assert "**Answer: 30**" in messages[2]["content"]
        assert "50 - 20 = 30" in messages[2]["content"]

    def test_multiple_hash_separators(self):
        """Only split on the last #### in case the reasoning contains hashes."""
        example = {
            "question": "What is 2+2?",
            "answer": "Step 1: #### shows work\nStep 2: add\n#### 4",
        }
        messages = format_gsm8k(example)
        assert "**Answer: 4**" in messages[2]["content"]
        # The earlier #### should be in the reasoning
        assert "####" in messages[2]["content"].split("**Answer:")[0]


# ──────────────────────────────────────────────
# OpenOrca formatter
# ──────────────────────────────────────────────

class TestFormatOpenOrca:
    def test_basic_formatting(self):
        example = {
            "system_prompt": "You are a math tutor.",
            "question": "What is calculus?",
            "response": "Calculus is the study of change.",
        }
        messages = format_openorca(example)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a math tutor."
        assert messages[1]["content"] == "What is calculus?"
        assert messages[2]["content"] == "Calculus is the study of change."

    def test_empty_system_prompt_uses_default(self):
        example = {
            "system_prompt": "",
            "question": "Hello?",
            "response": "Hi!",
        }
        messages = format_openorca(example)
        assert messages[0]["content"] == REASONING_SYSTEM_PROMPT

    def test_missing_system_prompt_uses_default(self):
        example = {
            "question": "Hello?",
            "response": "Hi!",
        }
        messages = format_openorca(example)
        assert messages[0]["content"] == REASONING_SYSTEM_PROMPT

    def test_whitespace_system_prompt_uses_default(self):
        example = {
            "system_prompt": "   ",
            "question": "Hello?",
            "response": "Hi!",
        }
        messages = format_openorca(example)
        assert messages[0]["content"] == REASONING_SYSTEM_PROMPT


# ──────────────────────────────────────────────
# MetaMathQA formatter
# ──────────────────────────────────────────────

class TestFormatMetaMath:
    def test_basic_formatting(self):
        example = {
            "query": "Solve for x: 2x + 3 = 7",
            "response": "2x = 4, so x = 2.",
            "type": "algebra",
        }
        messages = format_metamath(example)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Solve for x: 2x + 3 = 7"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "2x = 4, so x = 2."

    def test_system_prompt_is_reasoning(self):
        example = {"query": "Q?", "response": "A.", "type": "math"}
        messages = format_metamath(example)
        assert messages[0]["content"] == REASONING_SYSTEM_PROMPT


# ──────────────────────────────────────────────
# format_example_for_training
# ──────────────────────────────────────────────

class TestFormatExampleForTraining:
    def _mock_tokenizer(self):
        tok = MagicMock()
        tok.apply_chat_template = MagicMock(
            side_effect=lambda msgs, tokenize=False: "\n".join(
                f"<|{m['role']}|>{m['content']}" for m in msgs
            )
        )
        return tok

    def test_gsm8k_source(self):
        tok = self._mock_tokenizer()
        example = {
            "_source": "gsm8k",
            "question": "What is 1+1?",
            "answer": "1+1=2\n#### 2",
        }
        result = format_example_for_training(example, tok)
        assert "<|user|>What is 1+1?" in result
        assert "**Answer: 2**" in result

    def test_openorca_source(self):
        tok = self._mock_tokenizer()
        example = {
            "_source": "openorca",
            "system_prompt": "Be helpful.",
            "question": "Hi",
            "response": "Hello!",
        }
        result = format_example_for_training(example, tok)
        assert "<|system|>Be helpful." in result
        assert "<|user|>Hi" in result

    def test_metamath_source(self):
        tok = self._mock_tokenizer()
        example = {
            "_source": "metamath",
            "query": "2+2?",
            "response": "4",
            "type": "math",
        }
        result = format_example_for_training(example, tok)
        assert "<|user|>2+2?" in result
        assert "<|assistant|>4" in result

    def test_fallback_gsm8k_format(self):
        """Without _source, should infer from keys."""
        tok = self._mock_tokenizer()
        example = {
            "_source": "",
            "question": "What is 3+3?",
            "answer": "3+3=6\n#### 6",
        }
        result = format_example_for_training(example, tok)
        assert "**Answer: 6**" in result

    def test_fallback_metamath_format(self):
        tok = self._mock_tokenizer()
        example = {
            "_source": "",
            "query": "Solve x=1",
            "response": "x=1",
        }
        result = format_example_for_training(example, tok)
        assert "<|user|>Solve x=1" in result

    def test_unknown_format_raises(self):
        tok = self._mock_tokenizer()
        example = {"_source": "", "foo": "bar"}
        with pytest.raises(ValueError, match="Cannot format"):
            format_example_for_training(example, tok)


# ──────────────────────────────────────────────
# Dataset loading (mocked HuggingFace)
# ──────────────────────────────────────────────

class TestLoadReasoningDataset:
    def _make_fake_dataset(self, n=100):
        return Dataset.from_dict({
            "question": [f"Q{i}" for i in range(n)],
            "answer": [f"A{i}\n#### {i}" for i in range(n)],
        })

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_load_gsm8k(self, mock_load):
        mock_load.return_value = self._make_fake_dataset(50)
        ds = load_reasoning_dataset("gsm8k", max_samples=10)
        assert len(ds) == 10
        assert "_source" in ds.column_names
        assert ds[0]["_source"] == "gsm8k"

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_load_without_sampling(self, mock_load):
        mock_load.return_value = self._make_fake_dataset(50)
        ds = load_reasoning_dataset("gsm8k", max_samples=100)
        assert len(ds) == 50  # Dataset only has 50, don't over-sample

    def test_load_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_reasoning_dataset("nonexistent")

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_default_max_samples_applied(self, mock_load):
        """Large datasets (openorca) should be sampled by default."""
        big_ds = Dataset.from_dict({
            "system_prompt": ["sys"] * 50000,
            "question": [f"Q{i}" for i in range(50000)],
            "response": [f"R{i}" for i in range(50000)],
        })
        mock_load.return_value = big_ds
        ds = load_reasoning_dataset("openorca")
        # Should be capped at default_max_samples (10000)
        assert len(ds) == 10000

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_source_tag_added(self, mock_load):
        mock_load.return_value = self._make_fake_dataset(5)
        ds = load_reasoning_dataset("gsm8k")
        for row in ds:
            assert row["_source"] == "gsm8k"


# ──────────────────────────────────────────────
# Dataset mixing
# ──────────────────────────────────────────────

class TestMixDatasets:
    def _make_gsm8k_dataset(self, n=20):
        return Dataset.from_dict({
            "question": [f"Q{i}" for i in range(n)],
            "answer": [f"A{i}\n#### {i}" for i in range(n)],
        })

    def _make_metamath_dataset(self, n=20):
        return Dataset.from_dict({
            "query": [f"Q{i}" for i in range(n)],
            "response": [f"R{i}" for i in range(n)],
            "type": ["math"] * n,
        })

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_single_dataset(self, mock_load):
        mock_load.return_value = self._make_gsm8k_dataset(10)
        ds = mix_datasets(["gsm8k"])
        assert len(ds) == 10

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_mixed_datasets(self, mock_load):
        def side_effect(hf_id, **kwargs):
            if "gsm8k" in hf_id:
                return self._make_gsm8k_dataset(15)
            elif "MetaMath" in hf_id:
                return self._make_metamath_dataset(10)
            raise ValueError(f"Unexpected: {hf_id}")

        mock_load.side_effect = side_effect
        ds = mix_datasets(["gsm8k", "metamath"])
        assert len(ds) == 25  # 15 + 10

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_mixed_datasets_have_source_tags(self, mock_load):
        def side_effect(hf_id, **kwargs):
            if "gsm8k" in hf_id:
                return self._make_gsm8k_dataset(5)
            elif "MetaMath" in hf_id:
                return self._make_metamath_dataset(5)
            raise ValueError(f"Unexpected: {hf_id}")

        mock_load.side_effect = side_effect
        ds = mix_datasets(["gsm8k", "metamath"])
        sources = set(ds["_source"])
        assert "gsm8k" in sources
        assert "metamath" in sources

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_max_samples_per_dataset(self, mock_load):
        mock_load.return_value = self._make_gsm8k_dataset(100)
        ds = mix_datasets(["gsm8k"], max_samples_per_dataset=10)
        assert len(ds) == 10

    @patch("src.text_to_text.train_reasoning.load_dataset")
    def test_shuffle_is_deterministic(self, mock_load):
        mock_load.return_value = self._make_gsm8k_dataset(50)
        ds1 = mix_datasets(["gsm8k"], seed=42)
        mock_load.return_value = self._make_gsm8k_dataset(50)
        ds2 = mix_datasets(["gsm8k"], seed=42)
        assert ds1["_source"] == ds2["_source"]


# ──────────────────────────────────────────────
# CLI argument parsing
# ──────────────────────────────────────────────

class TestCLI:
    def test_main_function_exists(self):
        from src.text_to_text.train_reasoning import main
        assert callable(main)

    def test_default_args(self):
        import argparse
        from src.text_to_text.train_reasoning import main
        # Verify defaults by parsing empty args
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_model", default="microsoft/Phi-3-mini-4k-instruct")
        parser.add_argument("--datasets", nargs="+", default=["gsm8k", "metamath"])
        parser.add_argument("--output_dir", default="./checkpoints/t2t-reasoning")
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--quantize", action="store_true", default=True)
        args = parser.parse_args([])
        assert args.base_model == "microsoft/Phi-3-mini-4k-instruct"
        assert args.datasets == ["gsm8k", "metamath"]
        assert args.output_dir == "./checkpoints/t2t-reasoning"
        assert args.epochs == 3
        assert args.quantize is True


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────

class TestEdgeCases:
    def test_gsm8k_empty_question(self):
        example = {"question": "", "answer": "#### 0"}
        messages = format_gsm8k(example)
        assert messages[1]["content"] == ""
        assert "**Answer: 0**" in messages[2]["content"]

    def test_openorca_long_response(self):
        example = {
            "system_prompt": "Be concise.",
            "question": "Explain gravity.",
            "response": "Gravity is " + "a force " * 1000,
        }
        messages = format_openorca(example)
        assert len(messages[2]["content"]) > 1000

    def test_metamath_preserves_latex(self):
        example = {
            "query": "What is $\\frac{1}{2} + \\frac{1}{3}$?",
            "response": "$\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}$",
            "type": "algebra",
        }
        messages = format_metamath(example)
        assert "\\frac" in messages[1]["content"]
        assert "\\frac" in messages[2]["content"]

    def test_reasoning_system_prompt_not_empty(self):
        assert len(REASONING_SYSTEM_PROMPT) > 20
        assert "step" in REASONING_SYSTEM_PROMPT.lower()
