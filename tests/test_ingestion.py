"""Tests for the Document Ingestion pipeline (loader, chunker, pipeline)."""

import json
import uuid

import pytest

from src.rag.loader import DocumentLoader, RawDocument
from src.rag.chunker import TextChunker, Chunk
from src.rag.ingest import IngestionPipeline
from src.rag.vector_store import VectorStore


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def store():
    name = f"test_{uuid.uuid4().hex[:8]}"
    return VectorStore.ephemeral(collection_name=name)


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


# ──────────────────────────────────────────────
# DocumentLoader — Text & Markdown
# ──────────────────────────────────────────────

class TestLoaderText:
    def test_load_txt_file(self, tmp_dir):
        f = tmp_dir / "hello.txt"
        f.write_text("Hello world!")
        docs = DocumentLoader().load(str(f))
        assert len(docs) == 1
        assert docs[0].text == "Hello world!"
        assert docs[0].source == str(f)

    def test_load_md_file(self, tmp_dir):
        f = tmp_dir / "readme.md"
        f.write_text("# Title\nSome content")
        docs = DocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "# Title" in docs[0].text

    def test_load_empty_file(self, tmp_dir):
        f = tmp_dir / "empty.txt"
        f.write_text("")
        docs = DocumentLoader().load(str(f))
        assert docs == []

    def test_load_whitespace_only(self, tmp_dir):
        f = tmp_dir / "blank.txt"
        f.write_text("   \n\n  ")
        docs = DocumentLoader().load(str(f))
        assert docs == []

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            DocumentLoader().load("/nonexistent/path.txt")


# ──────────────────────────────────────────────
# DocumentLoader — JSON
# ──────────────────────────────────────────────

class TestLoaderJSON:
    def test_load_json_object_with_text_field(self, tmp_dir):
        f = tmp_dir / "doc.json"
        f.write_text(json.dumps({"text": "Hello from JSON"}))
        docs = DocumentLoader().load(str(f))
        assert len(docs) == 1
        assert docs[0].text == "Hello from JSON"

    def test_load_json_object_with_content_field(self, tmp_dir):
        f = tmp_dir / "doc.json"
        f.write_text(json.dumps({"content": "Content field"}))
        docs = DocumentLoader().load(str(f))
        assert docs[0].text == "Content field"

    def test_load_json_array_of_strings(self, tmp_dir):
        f = tmp_dir / "items.json"
        f.write_text(json.dumps(["one", "two", "three"]))
        docs = DocumentLoader().load(str(f))
        assert len(docs) == 3
        assert docs[0].text == "one"
        assert docs[0].metadata["json_index"] == 0

    def test_load_json_array_of_objects(self, tmp_dir):
        f = tmp_dir / "items.json"
        data = [{"text": "First"}, {"text": "Second"}]
        f.write_text(json.dumps(data))
        docs = DocumentLoader().load(str(f))
        assert len(docs) == 2
        assert docs[1].text == "Second"


# ──────────────────────────────────────────────
# DocumentLoader — CSV
# ──────────────────────────────────────────────

class TestLoaderCSV:
    def test_load_csv(self, tmp_dir):
        f = tmp_dir / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25")
        docs = DocumentLoader().load(str(f))
        assert len(docs) == 2
        assert "Alice" in docs[0].text
        assert "name: Alice" in docs[0].text
        assert docs[0].metadata["csv_row"] == 0


# ──────────────────────────────────────────────
# DocumentLoader — HTML
# ──────────────────────────────────────────────

class TestLoaderHTML:
    def test_load_html(self, tmp_dir):
        f = tmp_dir / "page.html"
        f.write_text("<html><body><h1>Hello</h1><p>World</p></body></html>")
        docs = DocumentLoader().load(str(f))
        assert len(docs) == 1
        assert "Hello" in docs[0].text
        assert "World" in docs[0].text

    def test_html_strips_script_tags(self, tmp_dir):
        f = tmp_dir / "page.html"
        f.write_text("<html><script>var x = 1;</script><body>Visible</body></html>")
        docs = DocumentLoader().load(str(f))
        assert "var x" not in docs[0].text
        assert "Visible" in docs[0].text


# ──────────────────────────────────────────────
# DocumentLoader — Directory
# ──────────────────────────────────────────────

class TestLoaderDirectory:
    def test_load_directory(self, tmp_dir):
        (tmp_dir / "a.txt").write_text("File A")
        (tmp_dir / "b.md").write_text("File B")
        (tmp_dir / "c.py").write_text("# Not loaded")
        docs = DocumentLoader().load(str(tmp_dir))
        assert len(docs) == 2  # .py not in supported extensions

    def test_load_directory_with_glob(self, tmp_dir):
        (tmp_dir / "a.txt").write_text("Text")
        (tmp_dir / "b.md").write_text("Markdown")
        docs = DocumentLoader().load(str(tmp_dir), glob="*.txt")
        assert len(docs) == 1
        assert docs[0].text == "Text"

    def test_load_nested_directory(self, tmp_dir):
        sub = tmp_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested file")
        docs = DocumentLoader().load(str(tmp_dir))
        assert len(docs) == 1
        assert docs[0].text == "Nested file"


# ──────────────────────────────────────────────
# TextChunker
# ──────────────────────────────────────────────

class TestChunker:
    def test_short_text_no_split(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.split("Short text.")
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."
        assert chunks[0].index == 0

    def test_empty_text(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.split("") == []
        assert chunker.split("   ") == []

    def test_paragraph_split(self):
        text = "Paragraph one. " * 20 + "\n\n" + "Paragraph two. " * 20
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split(text)
        assert len(chunks) >= 2

    def test_chunks_have_position(self):
        text = "A" * 100 + "\n\n" + "B" * 100
        chunker = TextChunker(chunk_size=120, chunk_overlap=10)
        chunks = chunker.split(text)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char

    def test_chunk_indexes_sequential(self):
        text = ("Word " * 100) * 5
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.split(text)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_fixed_strategy(self):
        text = "A" * 500
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="fixed")
        chunks = chunker.split(text)
        assert all(len(c.text) <= 100 for c in chunks)
        assert len(chunks) >= 5

    def test_overlap_validation(self):
        with pytest.raises(ValueError, match="chunk_overlap must be less"):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_all_text_preserved(self):
        """Verify no content is silently dropped during chunking."""
        words = ["word" + str(i) for i in range(50)]
        text = " ".join(words)
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.split(text)
        combined = " ".join(c.text for c in chunks)
        # Every word should appear in at least one chunk
        for word in words:
            assert word in combined, f"{word} missing from chunks"


# ──────────────────────────────────────────────
# Ingestion Pipeline
# ──────────────────────────────────────────────

class TestIngestionPipeline:
    def test_ingest_single_file(self, store, tmp_dir):
        f = tmp_dir / "doc.txt"
        f.write_text("This is a test document about machine learning.")
        pipeline = IngestionPipeline(store)
        stats = pipeline.ingest(str(f))
        assert stats.files_loaded == 1
        assert stats.raw_documents == 1
        assert stats.chunks_created >= 1
        assert stats.chunks_added >= 1
        assert store.count() == stats.chunks_added

    def test_ingest_directory(self, store, tmp_dir):
        (tmp_dir / "a.txt").write_text("Machine learning is a branch of AI.")
        (tmp_dir / "b.txt").write_text("Python is a popular programming language.")
        pipeline = IngestionPipeline(store)
        stats = pipeline.ingest(str(tmp_dir))
        assert stats.files_loaded == 2
        assert stats.raw_documents == 2
        assert store.count() >= 2

    def test_ingest_then_search(self, store, tmp_dir):
        f = tmp_dir / "knowledge.txt"
        f.write_text(
            "The Eiffel Tower is a wrought-iron lattice tower in Paris. "
            "It was constructed from 1887 to 1889 as the centerpiece of "
            "the 1889 World's Fair."
        )
        pipeline = IngestionPipeline(store)
        pipeline.ingest(str(f))
        results = store.search("Where is the Eiffel Tower?")
        assert len(results) > 0
        assert "Paris" in results[0].text

    def test_ingest_text_directly(self, store):
        pipeline = IngestionPipeline(store)
        stats = pipeline.ingest_text(
            "Python was created by Guido van Rossum and released in 1991.",
            source="manual_input",
        )
        assert stats.chunks_added >= 1
        results = store.search("Who created Python?")
        assert "Guido" in results[0].text

    def test_ingest_large_document_chunks(self, store, tmp_dir):
        f = tmp_dir / "large.txt"
        # Create a document larger than one chunk
        f.write_text("This is sentence number one. " * 200)
        pipeline = IngestionPipeline(store, chunk_size=256, chunk_overlap=32)
        stats = pipeline.ingest(str(f))
        assert stats.chunks_created > 1
        assert stats.chunks_added == stats.chunks_created

    def test_ingest_dedup_across_runs(self, store, tmp_dir):
        f = tmp_dir / "doc.txt"
        f.write_text("Exact same content both times.")
        pipeline = IngestionPipeline(store)
        stats1 = pipeline.ingest(str(f))
        stats2 = pipeline.ingest(str(f))
        assert stats1.chunks_added >= 1
        assert stats2.chunks_added == 0  # Already indexed

    def test_ingest_with_glob(self, store, tmp_dir):
        (tmp_dir / "notes.md").write_text("Markdown notes")
        (tmp_dir / "data.txt").write_text("Text data")
        pipeline = IngestionPipeline(store)
        stats = pipeline.ingest(str(tmp_dir), glob="*.md")
        assert stats.files_loaded == 1
        assert stats.sources == [str(tmp_dir / "notes.md")]

    def test_ingest_stats_sources(self, store, tmp_dir):
        (tmp_dir / "a.txt").write_text("File A content")
        (tmp_dir / "b.txt").write_text("File B content")
        pipeline = IngestionPipeline(store)
        stats = pipeline.ingest(str(tmp_dir))
        assert len(stats.sources) == 2

    def test_ingest_preserves_source_in_store(self, store, tmp_dir):
        f = tmp_dir / "tagged.txt"
        f.write_text("Document with source tracking")
        pipeline = IngestionPipeline(store)
        pipeline.ingest(str(f))
        sources = store.list_sources()
        assert str(f) in sources
