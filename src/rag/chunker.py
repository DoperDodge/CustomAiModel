"""Text Chunker — Split documents into embedding-sized pieces.

Strategies:
  - recursive: Split by paragraph → sentence → word boundaries (default)
  - fixed:     Fixed character-count windows with overlap

Usage:
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.split("Very long document text ...")
"""

from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """A single chunk of text from a larger document."""
    text: str
    index: int        # Chunk position within the source document
    start_char: int   # Character offset in the original text
    end_char: int     # End character offset


class TextChunker:
    """Split text into overlapping chunks for embedding.

    Args:
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        strategy: "recursive" (paragraph→sentence→word) or "fixed" (character windows).
    """

    # Recursive split separators, tried in order of preference
    _SEPARATORS = [
        "\n\n",   # Paragraph break
        "\n",     # Line break
        ". ",     # Sentence boundary
        "! ",
        "? ",
        "; ",
        ", ",
        " ",      # Word boundary (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        strategy: str = "recursive",
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def split(self, text: str) -> list[Chunk]:
        """Split text into chunks.

        Returns:
            List of Chunk objects preserving position information.
        """
        if not text or not text.strip():
            return []

        if self.strategy == "fixed":
            return self._split_fixed(text)
        else:
            return self._split_recursive(text)

    def _split_fixed(self, text: str) -> list[Chunk]:
        """Split into fixed-size windows with overlap."""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=idx,
                    start_char=start,
                    end_char=end,
                ))
                idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _split_recursive(self, text: str) -> list[Chunk]:
        """Split by trying separators from coarsest to finest."""
        raw_pieces = self._recursive_split(text, 0)

        # Merge small pieces, split oversized ones
        merged = self._merge_pieces(raw_pieces)

        # Build Chunk objects with position tracking
        chunks = []
        offset = 0
        for idx, piece in enumerate(merged):
            # Find the actual position in original text
            pos = text.find(piece, offset)
            if pos == -1:
                pos = offset
            chunks.append(Chunk(
                text=piece,
                index=idx,
                start_char=pos,
                end_char=pos + len(piece),
            ))
            offset = pos + len(piece)

        return chunks

    def _recursive_split(self, text: str, sep_idx: int) -> list[str]:
        """Recursively split text using increasingly fine separators."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if sep_idx >= len(self._SEPARATORS):
            # No more separators — hard-cut at chunk_size
            return self._hard_split(text)

        sep = self._SEPARATORS[sep_idx]
        parts = text.split(sep)

        # If the separator didn't actually split anything useful, try next
        if len(parts) <= 1:
            return self._recursive_split(text, sep_idx + 1)

        result = []
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            if len(stripped) <= self.chunk_size:
                result.append(stripped)
            else:
                # This piece is still too big — recurse with finer separator
                result.extend(self._recursive_split(stripped, sep_idx + 1))

        return result

    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split at chunk_size boundaries."""
        pieces = []
        for i in range(0, len(text), self.chunk_size):
            piece = text[i : i + self.chunk_size].strip()
            if piece:
                pieces.append(piece)
        return pieces

    def _merge_pieces(self, pieces: list[str]) -> list[str]:
        """Merge small consecutive pieces up to chunk_size."""
        if not pieces:
            return []

        merged = []
        current = pieces[0]

        for piece in pieces[1:]:
            combined = current + "\n\n" + piece
            if len(combined) <= self.chunk_size:
                current = combined
            else:
                if current.strip():
                    merged.append(current.strip())
                current = piece

        if current.strip():
            merged.append(current.strip())

        return merged
