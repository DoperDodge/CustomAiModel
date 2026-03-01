"""Tests for Context Injection (RAG prompt augmentation)."""

import uuid

import pytest

from src.rag.vector_store import VectorStore, Document
from src.rag.context import ContextInjector, RetrievedContext


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def store():
    name = f"test_{uuid.uuid4().hex[:8]}"
    s = VectorStore.ephemeral(collection_name=name)
    # Pre-load some knowledge
    s.add_documents([
        Document(text="The Eiffel Tower is a wrought-iron lattice tower in Paris, France.", source="wiki"),
        Document(text="Python is a high-level programming language created by Guido van Rossum.", source="docs"),
        Document(text="The Great Wall of China stretches over 13,000 miles.", source="wiki"),
        Document(text="Machine learning is a subset of artificial intelligence.", source="docs"),
        Document(text="Water boils at 100 degrees Celsius at sea level.", source="science"),
    ])
    return s


@pytest.fixture
def injector(store):
    return ContextInjector(store, n_results=3, min_score=0.1)


BASE_PROMPT = "You are a helpful AI assistant."


# ──────────────────────────────────────────────
# retrieve_context
# ──────────────────────────────────────────────

class TestRetrieveContext:
    def test_basic_retrieval(self, injector):
        ctx = injector.retrieve_context("Tell me about the Eiffel Tower")
        assert ctx.num_chunks > 0
        assert ctx.context_text != ""
        assert "Eiffel Tower" in ctx.context_text

    def test_returns_sources(self, injector):
        ctx = injector.retrieve_context("programming language")
        assert len(ctx.sources) > 0

    def test_empty_store(self):
        empty = VectorStore.ephemeral(collection_name=f"empty_{uuid.uuid4().hex[:8]}")
        inj = ContextInjector(empty)
        ctx = inj.retrieve_context("anything")
        assert ctx.num_chunks == 0
        assert ctx.context_text == ""
        assert ctx.sources == []

    def test_min_score_filtering(self, store):
        # Very high min_score should filter out most results
        strict = ContextInjector(store, n_results=5, min_score=0.99)
        ctx = strict.retrieve_context("random query about nothing specific")
        assert ctx.num_chunks == 0

    def test_result_format(self, injector):
        ctx = injector.retrieve_context("Paris France")
        assert ctx.query == "Paris France"
        assert isinstance(ctx.results, list)
        if ctx.results:
            assert hasattr(ctx.results[0], "text")
            assert hasattr(ctx.results[0], "score")

    def test_metadata_filter(self, injector):
        ctx = injector.retrieve_context("information", where={"source": "wiki"})
        for result in ctx.results:
            assert result.source == "wiki"

    def test_n_results_limit(self, store):
        inj = ContextInjector(store, n_results=2, min_score=0.0)
        ctx = inj.retrieve_context("tell me something")
        assert ctx.num_chunks <= 2

    def test_document_numbering_in_context(self, injector):
        ctx = injector.retrieve_context("programming")
        if ctx.num_chunks > 0:
            assert "[Document 1" in ctx.context_text


# ──────────────────────────────────────────────
# build_augmented_prompt
# ──────────────────────────────────────────────

class TestBuildAugmentedPrompt:
    def test_augments_with_context(self, injector):
        prompt = injector.build_augmented_prompt("Eiffel Tower", BASE_PROMPT)
        assert prompt.startswith(BASE_PROMPT)
        assert "Eiffel Tower" in prompt
        assert len(prompt) > len(BASE_PROMPT)

    def test_no_context_returns_base(self):
        empty = VectorStore.ephemeral(collection_name=f"empty_{uuid.uuid4().hex[:8]}")
        inj = ContextInjector(empty)
        prompt = inj.build_augmented_prompt("anything", BASE_PROMPT)
        assert prompt == BASE_PROMPT

    def test_includes_header(self, injector):
        prompt = injector.build_augmented_prompt("Python", BASE_PROMPT)
        assert "reference documents" in prompt.lower() or "documents" in prompt.lower()

    def test_custom_header(self, store):
        inj = ContextInjector(store, context_header="CUSTOM HEADER:")
        prompt = inj.build_augmented_prompt("Python", BASE_PROMPT)
        assert "CUSTOM HEADER:" in prompt

    def test_strict_score_no_augmentation(self, store):
        strict = ContextInjector(store, min_score=0.99)
        prompt = strict.build_augmented_prompt("xyzzy gibberish", BASE_PROMPT)
        assert prompt == BASE_PROMPT


# ──────────────────────────────────────────────
# augment_messages
# ──────────────────────────────────────────────

class TestAugmentMessages:
    def test_injects_system_message(self, injector):
        messages = [
            {"role": "user", "content": "What is the Eiffel Tower?"},
        ]
        result = injector.augment_messages(messages, BASE_PROMPT)
        assert result[0]["role"] == "system"
        assert "Eiffel Tower" in result[0]["content"]
        assert result[1] == messages[0]

    def test_replaces_existing_system_message(self, injector):
        messages = [
            {"role": "system", "content": "Old system prompt"},
            {"role": "user", "content": "Tell me about Python"},
        ]
        result = injector.augment_messages(messages, BASE_PROMPT)
        assert result[0]["role"] == "system"
        # Should be the augmented prompt, not the old one
        assert BASE_PROMPT in result[0]["content"]
        assert len(result) == 2

    def test_does_not_mutate_original(self, injector):
        messages = [
            {"role": "user", "content": "Eiffel Tower?"},
        ]
        original = [m.copy() for m in messages]
        injector.augment_messages(messages, BASE_PROMPT)
        assert messages == original

    def test_uses_last_user_message(self, injector):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Tell me about the Eiffel Tower in Paris"},
        ]
        result = injector.augment_messages(messages, BASE_PROMPT)
        assert "Paris" in result[0]["content"] or "Eiffel" in result[0]["content"]

    def test_no_user_message_returns_unchanged(self, injector):
        messages = [
            {"role": "system", "content": "System only"},
        ]
        result = injector.augment_messages(messages, BASE_PROMPT)
        assert result == messages

    def test_empty_store_adds_plain_system(self):
        empty = VectorStore.ephemeral(collection_name=f"empty_{uuid.uuid4().hex[:8]}")
        inj = ContextInjector(empty)
        messages = [
            {"role": "user", "content": "Hello"},
        ]
        result = inj.augment_messages(messages, BASE_PROMPT)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == BASE_PROMPT

    def test_preserves_conversation_order(self, injector):
        messages = [
            {"role": "user", "content": "First question about Python"},
            {"role": "assistant", "content": "Python is great"},
            {"role": "user", "content": "Tell me more about machine learning"},
        ]
        result = injector.augment_messages(messages, BASE_PROMPT)
        # System prompt should be first, then all original messages
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["role"] == "assistant"
        assert result[3]["role"] == "user"
        assert len(result) == 4
