"""Ingestion Pipeline — Load, chunk, and index documents into the vector store.

Ties together DocumentLoader + TextChunker + VectorStore into a single
pipeline for ingesting knowledge sources.

Usage:
    from src.rag.vector_store import VectorStore
    from src.rag.ingest import IngestionPipeline

    store = VectorStore.persistent("./data/vectordb")
    pipeline = IngestionPipeline(store)

    # Ingest a directory of knowledge files
    stats = pipeline.ingest("./knowledge/")
    print(f"Ingested {stats.chunks_added} chunks from {stats.files_loaded} files")

    # Ingest a single file
    stats = pipeline.ingest("notes.md")
"""

from dataclasses import dataclass, field
from pathlib import Path
import hashlib

from src.rag.loader import DocumentLoader, RawDocument
from src.rag.chunker import TextChunker
from src.rag.vector_store import VectorStore, Document


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""
    files_loaded: int = 0
    raw_documents: int = 0
    chunks_created: int = 0
    chunks_added: int = 0    # After dedup
    sources: list[str] = field(default_factory=list)


class IngestionPipeline:
    """End-to-end pipeline: files → chunks → vector store.

    Args:
        store: VectorStore instance to index into.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        chunk_strategy: Chunking strategy ("recursive" or "fixed").
    """

    def __init__(
        self,
        store: VectorStore,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        chunk_strategy: str = "recursive",
    ):
        self.store = store
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy,
        )

    def ingest(self, path: str, glob: str | None = None) -> IngestionStats:
        """Load, chunk, and index documents from a path.

        Args:
            path: File or directory path.
            glob: Optional glob filter for directories.

        Returns:
            IngestionStats with counts from this run.
        """
        stats = IngestionStats()

        # 1. Load raw documents
        raw_docs = self.loader.load(path, glob=glob)
        stats.raw_documents = len(raw_docs)

        # Count unique source files
        sources = set()
        for doc in raw_docs:
            sources.add(doc.source)
        stats.files_loaded = len(sources)
        stats.sources = sorted(sources)

        # 2. Chunk each document
        all_chunks: list[Document] = []
        for raw_doc in raw_docs:
            chunks = self.chunker.split(raw_doc.text)
            for chunk in chunks:
                # ID includes source + chunk index so identical text in
                # different positions doesn't collide
                chunk_id = hashlib.sha256(
                    f"{raw_doc.source}::{chunk.index}::{chunk.text}".encode()
                ).hexdigest()[:16]
                all_chunks.append(Document(
                    text=chunk.text,
                    source=raw_doc.source,
                    doc_id=chunk_id,
                    metadata={
                        "chunk_index": chunk.index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        **raw_doc.metadata,
                    },
                ))

        stats.chunks_created = len(all_chunks)

        # 3. Index into vector store
        if all_chunks:
            stats.chunks_added = self.store.add_documents(all_chunks)

        return stats

    def ingest_text(self, text: str, source: str = "manual") -> IngestionStats:
        """Ingest raw text directly (no file loading).

        Args:
            text: The text to chunk and index.
            source: Source label for the text.

        Returns:
            IngestionStats with counts from this run.
        """
        stats = IngestionStats()
        stats.files_loaded = 0
        stats.raw_documents = 1
        stats.sources = [source]

        chunks = self.chunker.split(text)
        documents = [
            Document(
                text=chunk.text,
                source=source,
                doc_id=hashlib.sha256(
                    f"{source}::{chunk.index}::{chunk.text}".encode()
                ).hexdigest()[:16],
                metadata={"chunk_index": chunk.index, "start_char": chunk.start_char, "end_char": chunk.end_char},
            )
            for chunk in chunks
        ]
        stats.chunks_created = len(documents)

        if documents:
            stats.chunks_added = self.store.add_documents(documents)

        return stats
