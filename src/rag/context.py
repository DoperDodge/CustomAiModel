"""Context Injection — Retrieve relevant documents and inject into the prompt.

Bridges the vector store and the LLM by:
  1. Searching for chunks relevant to the user's query
  2. Formatting them into a context block
  3. Injecting that block into the system prompt

Usage:
    from src.rag.vector_store import VectorStore
    from src.rag.context import ContextInjector

    store = VectorStore.persistent("./data/vectordb")
    injector = ContextInjector(store)

    # Build an augmented system prompt
    augmented = injector.build_augmented_prompt(
        query="What is the Eiffel Tower?",
        base_system_prompt="You are a helpful assistant.",
    )

    # Or get raw context for custom use
    context_block = injector.retrieve_context("What is the Eiffel Tower?")
"""

from dataclasses import dataclass, field

from src.rag.vector_store import VectorStore, SearchResult


@dataclass
class RetrievedContext:
    """The result of a context retrieval operation."""
    query: str
    results: list[SearchResult]
    context_text: str          # Formatted text block ready for injection
    sources: list[str]         # Unique source files that contributed
    num_chunks: int            # Number of chunks used


class ContextInjector:
    """Retrieve relevant documents and inject them into LLM prompts.

    Args:
        store: VectorStore instance to search.
        n_results: Max number of chunks to retrieve per query.
        min_score: Minimum similarity score to include a chunk (0-1).
        context_header: Header text before the context block.
    """

    DEFAULT_HEADER = (
        "Use the following reference documents to help answer the user's question. "
        "If the documents don't contain relevant information, rely on your own knowledge. "
        "Do not fabricate information that isn't supported by the documents or your training."
    )

    def __init__(
        self,
        store: VectorStore,
        n_results: int = 5,
        min_score: float = 0.3,
        context_header: str | None = None,
    ):
        self.store = store
        self.n_results = n_results
        self.min_score = min_score
        self.context_header = context_header or self.DEFAULT_HEADER

    def retrieve_context(self, query: str, where: dict | None = None) -> RetrievedContext:
        """Search the vector store and build a formatted context block.

        Args:
            query: The user's question or search query.
            where: Optional metadata filter for the search.

        Returns:
            RetrievedContext with the formatted text and metadata.
        """
        if self.store.count() == 0:
            return RetrievedContext(
                query=query,
                results=[],
                context_text="",
                sources=[],
                num_chunks=0,
            )

        results = self.store.search(query, n_results=self.n_results, where=where)

        # Filter by minimum score
        results = [r for r in results if r.score >= self.min_score]

        if not results:
            return RetrievedContext(
                query=query,
                results=results,
                context_text="",
                sources=[],
                num_chunks=0,
            )

        # Format the context block
        context_text = self._format_context(results)
        sources = sorted(set(r.source for r in results if r.source))

        return RetrievedContext(
            query=query,
            results=results,
            context_text=context_text,
            sources=sources,
            num_chunks=len(results),
        )

    def build_augmented_prompt(
        self,
        query: str,
        base_system_prompt: str,
        where: dict | None = None,
    ) -> str:
        """Build a system prompt augmented with retrieved context.

        If no relevant documents are found, returns the base prompt unchanged.

        Args:
            query: The user's question (used to search for relevant docs).
            base_system_prompt: The original system prompt to augment.
            where: Optional metadata filter.

        Returns:
            The augmented system prompt string.
        """
        ctx = self.retrieve_context(query, where=where)

        if not ctx.context_text:
            return base_system_prompt

        return f"{base_system_prompt}\n\n{self.context_header}\n\n{ctx.context_text}"

    def augment_messages(
        self,
        messages: list[dict],
        base_system_prompt: str,
    ) -> list[dict]:
        """Augment a message list with RAG context based on the latest user message.

        Extracts the last user message as the query, retrieves context,
        and injects it into the system prompt. Returns a new message list
        (does not mutate the original).

        Args:
            messages: List of message dicts with "role" and "content" keys.
            base_system_prompt: The base system prompt to augment.

        Returns:
            New message list with augmented system prompt.
        """
        # Find the last user message to use as the search query
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg["content"]
                break

        if not query:
            return messages

        augmented_prompt = self.build_augmented_prompt(query, base_system_prompt)

        # Build new message list with augmented system prompt
        new_messages = []
        system_added = False
        for msg in messages:
            if msg.get("role") == "system" and not system_added:
                new_messages.append({"role": "system", "content": augmented_prompt})
                system_added = True
            else:
                new_messages.append(msg)

        # If there was no system message, prepend one
        if not system_added:
            new_messages.insert(0, {"role": "system", "content": augmented_prompt})

        return new_messages

    def _format_context(self, results: list[SearchResult]) -> str:
        """Format search results into a readable context block."""
        parts = []
        for i, result in enumerate(results, 1):
            source_tag = f" (source: {result.source})" if result.source else ""
            parts.append(f"[Document {i}{source_tag}]\n{result.text}")

        return "\n\n".join(parts)
