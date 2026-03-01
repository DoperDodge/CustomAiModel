"""RAG — Retrieval-Augmented Generation pipeline.

Modules:
    vector_store: ChromaDB wrapper for document embedding storage & search
    loader:       Load files (txt, md, json, csv, pdf, html) into raw text
    chunker:      Split text into overlapping chunks for embedding
    ingest:       End-to-end pipeline: files → chunks → vector store
"""
