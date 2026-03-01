"""Vector Store — Document embedding storage and similarity search.

Wraps ChromaDB to provide a clean interface for the RAG pipeline.
ChromaDB bundles its own embedding model (all-MiniLM-L6-v2) so no
additional model downloads are needed.

Supports two modes:
  - Ephemeral (in-memory): Fast, no disk I/O, lost on restart
  - Persistent (on-disk): Survives restarts, good for production

Usage:
    store = VectorStore.ephemeral()          # In-memory
    store = VectorStore.persistent("./data") # On-disk

    # Add documents
    store.add_documents([
        Document(text="Python is a programming language", source="wiki"),
        Document(text="The sky is blue", source="facts"),
    ])

    # Search
    results = store.search("What is Python?", n_results=3)
    for r in results:
        print(f"[{r.score:.2f}] {r.text}")
"""

from dataclasses import dataclass, field
from typing import Sequence
import hashlib
import time

import chromadb


@dataclass
class Document:
    """A document to store in the vector database."""
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)
    doc_id: str | None = None

    def __post_init__(self):
        if self.doc_id is None:
            # Deterministic ID from content — deduplicates identical documents
            self.doc_id = hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """A single result from a similarity search."""
    text: str
    score: float
    source: str = ""
    metadata: dict = field(default_factory=dict)
    doc_id: str = ""


@dataclass
class CollectionStats:
    """Statistics about a collection."""
    name: str
    count: int


class VectorStore:
    """Vector database for document embedding storage and retrieval.

    Uses ChromaDB with its built-in all-MiniLM-L6-v2 embedding model
    (384-dimensional sentence embeddings, runs on CPU).
    """

    def __init__(self, client: chromadb.ClientAPI, collection_name: str = "documents"):
        self._client = client
        self._collection_name = collection_name
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Cosine similarity
        )

    @classmethod
    def ephemeral(cls, collection_name: str = "documents") -> "VectorStore":
        """Create an in-memory vector store (fast, lost on restart)."""
        client = chromadb.Client()
        return cls(client, collection_name)

    @classmethod
    def persistent(cls, path: str = "./data/vectordb", collection_name: str = "documents") -> "VectorStore":
        """Create a persistent vector store (survives restarts)."""
        client = chromadb.PersistentClient(path=path)
        return cls(client, collection_name)

    def add_documents(self, documents: Sequence[Document]) -> int:
        """Add documents to the vector store.

        Duplicate documents (same text content) are automatically
        deduplicated by their content hash ID.

        Args:
            documents: List of Document objects to add.

        Returns:
            Number of documents actually added (after dedup).
        """
        if not documents:
            return 0

        # Filter out documents that already exist
        ids = [doc.doc_id for doc in documents]
        existing = set()
        try:
            result = self._collection.get(ids=ids)
            if result and result["ids"]:
                existing = set(result["ids"])
        except Exception:
            pass

        new_docs = [d for d in documents if d.doc_id not in existing]
        if not new_docs:
            return 0

        self._collection.add(
            ids=[d.doc_id for d in new_docs],
            documents=[d.text for d in new_docs],
            metadatas=[
                {"source": d.source, "added_at": time.time(), **d.metadata}
                for d in new_docs
            ],
        )
        return len(new_docs)

    def search(self, query: str, n_results: int = 5, where: dict | None = None) -> list[SearchResult]:
        """Search for documents similar to the query.

        Args:
            query: Natural language search query.
            n_results: Maximum number of results to return.
            where: Optional metadata filter (ChromaDB where clause).

        Returns:
            List of SearchResult objects, sorted by relevance (best first).
        """
        kwargs = {
            "query_texts": [query],
            "n_results": min(n_results, self.count()),
        }
        if where:
            kwargs["where"] = where

        if self.count() == 0:
            return []

        results = self._collection.query(**kwargs)

        search_results = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distance; convert to similarity score (0-1 for cosine)
            distance = results["distances"][0][i]
            score = 1.0 - distance  # cosine distance → cosine similarity

            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            source = metadata.pop("source", "")
            metadata.pop("added_at", None)

            search_results.append(SearchResult(
                text=results["documents"][0][i],
                score=score,
                source=source,
                metadata=metadata,
                doc_id=results["ids"][0][i],
            ))

        return search_results

    def delete(self, doc_ids: list[str]) -> None:
        """Delete documents by their IDs."""
        if doc_ids:
            self._collection.delete(ids=doc_ids)

    def delete_by_source(self, source: str) -> int:
        """Delete all documents from a specific source.

        Returns:
            Number of documents deleted.
        """
        results = self._collection.get(where={"source": source})
        if results and results["ids"]:
            self._collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def count(self) -> int:
        """Return the total number of documents in the collection."""
        return self._collection.count()

    def stats(self) -> CollectionStats:
        """Return collection statistics."""
        return CollectionStats(
            name=self._collection_name,
            count=self.count(),
        )

    def list_sources(self) -> list[str]:
        """Return all unique source names in the collection."""
        all_meta = self._collection.get()["metadatas"]
        sources = set()
        for meta in all_meta:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)

    def reset(self) -> None:
        """Delete the collection and recreate it (clears all data)."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
