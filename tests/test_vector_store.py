"""Tests for the Vector Store (ChromaDB wrapper)."""

import uuid

import pytest

from src.rag.vector_store import VectorStore, Document, SearchResult


@pytest.fixture
def store():
    """Create a fresh ephemeral vector store with a unique collection per test."""
    name = f"test_{uuid.uuid4().hex[:8]}"
    return VectorStore.ephemeral(collection_name=name)


# ──────────────────────────────────────────────
# Adding Documents
# ──────────────────────────────────────────────

class TestAddDocuments:
    def test_add_single_document(self, store):
        doc = Document(text="Python is a programming language")
        added = store.add_documents([doc])
        assert added == 1
        assert store.count() == 1

    def test_add_multiple_documents(self, store):
        docs = [
            Document(text="The sky is blue"),
            Document(text="Water is wet"),
            Document(text="Fire is hot"),
        ]
        added = store.add_documents(docs)
        assert added == 3
        assert store.count() == 3

    def test_add_with_source(self, store):
        doc = Document(text="Hello world", source="test.txt")
        store.add_documents([doc])
        results = store.search("Hello")
        assert results[0].source == "test.txt"

    def test_add_with_metadata(self, store):
        doc = Document(text="Custom metadata", metadata={"category": "test"})
        store.add_documents([doc])
        results = store.search("Custom metadata")
        assert results[0].metadata.get("category") == "test"

    def test_deduplication(self, store):
        doc1 = Document(text="Duplicate content")
        doc2 = Document(text="Duplicate content")
        store.add_documents([doc1])
        added = store.add_documents([doc2])
        assert added == 0  # Already exists
        assert store.count() == 1

    def test_add_empty_list(self, store):
        added = store.add_documents([])
        assert added == 0
        assert store.count() == 0

    def test_custom_doc_id(self, store):
        doc = Document(text="Custom ID doc", doc_id="my-custom-id")
        store.add_documents([doc])
        assert store.count() == 1

    def test_deterministic_ids(self):
        doc1 = Document(text="Same content")
        doc2 = Document(text="Same content")
        assert doc1.doc_id == doc2.doc_id

    def test_different_content_different_ids(self):
        doc1 = Document(text="Content A")
        doc2 = Document(text="Content B")
        assert doc1.doc_id != doc2.doc_id


# ──────────────────────────────────────────────
# Searching
# ──────────────────────────────────────────────

class TestSearch:
    def test_basic_search(self, store):
        store.add_documents([
            Document(text="Python is a programming language"),
            Document(text="The weather is sunny today"),
            Document(text="JavaScript runs in the browser"),
        ])
        results = store.search("programming languages")
        assert len(results) > 0
        # Python doc should be most relevant
        assert "Python" in results[0].text or "JavaScript" in results[0].text

    def test_search_returns_scores(self, store):
        store.add_documents([
            Document(text="Machine learning is a subset of AI"),
        ])
        results = store.search("artificial intelligence")
        assert len(results) == 1
        assert isinstance(results[0].score, float)
        assert 0 <= results[0].score <= 1

    def test_search_n_results(self, store):
        docs = [Document(text=f"Document number {i}") for i in range(10)]
        store.add_documents(docs)
        results = store.search("document", n_results=3)
        assert len(results) == 3

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert results == []

    def test_search_relevance_order(self, store):
        store.add_documents([
            Document(text="Cats are fluffy pets that purr"),
            Document(text="The stock market crashed today"),
            Document(text="Dogs are loyal companions"),
        ])
        results = store.search("pet animals")
        # Animal-related docs should score higher than stock market
        animal_scores = [r.score for r in results if "pet" in r.text or "companion" in r.text]
        stock_scores = [r.score for r in results if "stock" in r.text]
        if animal_scores and stock_scores:
            assert max(animal_scores) > max(stock_scores)

    def test_search_with_metadata_filter(self, store):
        store.add_documents([
            Document(text="Python tutorial", source="docs"),
            Document(text="Python news update", source="blog"),
            Document(text="JavaScript tutorial", source="docs"),
        ])
        results = store.search("Python", where={"source": "docs"})
        for r in results:
            assert r.source == "docs"

    def test_search_result_has_doc_id(self, store):
        store.add_documents([Document(text="Test document")])
        results = store.search("test")
        assert results[0].doc_id != ""


# ──────────────────────────────────────────────
# Deletion
# ──────────────────────────────────────────────

class TestDelete:
    def test_delete_by_id(self, store):
        doc = Document(text="Delete me", doc_id="to-delete")
        store.add_documents([doc])
        assert store.count() == 1
        store.delete(["to-delete"])
        assert store.count() == 0

    def test_delete_by_source(self, store):
        store.add_documents([
            Document(text="Keep this", source="keep"),
            Document(text="Remove this", source="remove"),
            Document(text="Also remove", source="remove"),
        ])
        deleted = store.delete_by_source("remove")
        assert deleted == 2
        assert store.count() == 1

    def test_delete_nonexistent_source(self, store):
        store.add_documents([Document(text="Something")])
        deleted = store.delete_by_source("nonexistent")
        assert deleted == 0
        assert store.count() == 1


# ──────────────────────────────────────────────
# Stats & Utility
# ──────────────────────────────────────────────

class TestStatsAndUtility:
    def test_count_empty(self, store):
        assert store.count() == 0

    def test_count_after_add(self, store):
        store.add_documents([
            Document(text="One"),
            Document(text="Two"),
        ])
        assert store.count() == 2

    def test_stats(self, store):
        store.add_documents([Document(text="Test")])
        stats = store.stats()
        assert stats.name.startswith("test_")
        assert stats.count == 1

    def test_list_sources(self, store):
        store.add_documents([
            Document(text="From A", source="source_a"),
            Document(text="From B", source="source_b"),
            Document(text="Also from A", source="source_a"),
        ])
        sources = store.list_sources()
        assert sorted(sources) == ["source_a", "source_b"]

    def test_list_sources_empty(self, store):
        assert store.list_sources() == []

    def test_reset(self, store):
        store.add_documents([
            Document(text="One"),
            Document(text="Two"),
            Document(text="Three"),
        ])
        assert store.count() == 3
        store.reset()
        assert store.count() == 0


# ──────────────────────────────────────────────
# Multiple Collections
# ──────────────────────────────────────────────

class TestMultipleCollections:
    def test_separate_collections(self):
        store_a = VectorStore.ephemeral(collection_name="collection_a")
        store_b = VectorStore.ephemeral(collection_name="collection_b")

        store_a.add_documents([Document(text="In collection A")])
        store_b.add_documents([Document(text="In collection B")])

        assert store_a.count() == 1
        assert store_b.count() == 1

        results_a = store_a.search("collection")
        assert "A" in results_a[0].text
