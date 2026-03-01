"""Document Loader — Read files from disk into raw text.

Supports: .txt, .md, .json, .csv, .pdf (if pypdf installed), .html

Usage:
    loader = DocumentLoader()
    raw_docs = loader.load("knowledge/")          # Load entire directory
    raw_docs = loader.load("notes.txt")            # Single file
    raw_docs = loader.load("data/", glob="*.md")   # Only markdown files
"""

from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
import csv
import io
import json


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".pdf", ".html", ".htm"}


@dataclass
class RawDocument:
    """Raw text loaded from a file before chunking."""
    text: str
    source: str  # File path or URI
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """Load documents from files and directories."""

    def load(self, path: str, glob: str | None = None) -> list[RawDocument]:
        """Load documents from a file or directory.

        Args:
            path: File path or directory path.
            glob: Optional glob pattern to filter files (e.g. "*.md").
                  Only used when path is a directory.

        Returns:
            List of RawDocument objects.
        """
        p = Path(path)
        if p.is_file():
            return self._load_file(p)
        elif p.is_dir():
            return self._load_directory(p, glob)
        else:
            raise FileNotFoundError(f"Path not found: {path}")

    def _load_directory(self, directory: Path, glob: str | None = None) -> list[RawDocument]:
        """Recursively load all supported files from a directory."""
        pattern = glob or "*"
        docs = []
        for filepath in sorted(directory.rglob(pattern)):
            if filepath.is_file() and filepath.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.extend(self._load_file(filepath))
        return docs

    def _load_file(self, filepath: Path) -> list[RawDocument]:
        """Load a single file based on its extension."""
        ext = filepath.suffix.lower()
        source = str(filepath)

        if ext in (".txt", ".md"):
            return self._load_text(filepath, source)
        elif ext == ".json":
            return self._load_json(filepath, source)
        elif ext == ".csv":
            return self._load_csv(filepath, source)
        elif ext == ".pdf":
            return self._load_pdf(filepath, source)
        elif ext in (".html", ".htm"):
            return self._load_html(filepath, source)
        else:
            return []

    def _load_text(self, filepath: Path, source: str) -> list[RawDocument]:
        text = filepath.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return []
        return [RawDocument(text=text, source=source)]

    def _load_json(self, filepath: Path, source: str) -> list[RawDocument]:
        """Load JSON — handles objects, arrays of strings, and arrays of objects."""
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        data = json.loads(raw)

        if isinstance(data, str):
            return [RawDocument(text=data, source=source)] if data.strip() else []

        if isinstance(data, list):
            docs = []
            for i, item in enumerate(data):
                if isinstance(item, str) and item.strip():
                    docs.append(RawDocument(
                        text=item,
                        source=source,
                        metadata={"json_index": i},
                    ))
                elif isinstance(item, dict):
                    # Try common text fields, fall back to full JSON
                    text = (
                        item.get("text")
                        or item.get("content")
                        or item.get("body")
                        or json.dumps(item, ensure_ascii=False)
                    )
                    if text.strip():
                        docs.append(RawDocument(
                            text=text,
                            source=source,
                            metadata={"json_index": i},
                        ))
            return docs

        if isinstance(data, dict):
            text = (
                data.get("text")
                or data.get("content")
                or data.get("body")
                or json.dumps(data, ensure_ascii=False)
            )
            return [RawDocument(text=text, source=source)] if text.strip() else []

        return []

    def _load_csv(self, filepath: Path, source: str) -> list[RawDocument]:
        """Load CSV — each row becomes a document (columns joined by newlines)."""
        text = filepath.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        docs = []
        for i, row in enumerate(reader):
            lines = [f"{k}: {v}" for k, v in row.items() if v and v.strip()]
            if lines:
                docs.append(RawDocument(
                    text="\n".join(lines),
                    source=source,
                    metadata={"csv_row": i},
                ))
        return docs

    def _load_pdf(self, filepath: Path, source: str) -> list[RawDocument]:
        """Load PDF — one document per page. Requires pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. Install it: pip install pypdf"
            )

        reader = PdfReader(filepath)
        docs = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(RawDocument(
                    text=text,
                    source=source,
                    metadata={"page": i + 1},
                ))
        return docs

    def _load_html(self, filepath: Path, source: str) -> list[RawDocument]:
        """Load HTML — strip tags to extract text content."""
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        text = _strip_html_tags(raw)
        if not text.strip():
            return []
        return [RawDocument(text=text, source=source)]


class _HTMLTextExtractor(HTMLParser):
    """Simple HTML parser that extracts visible text."""

    SKIP_TAGS = {"script", "style", "head", "meta", "link"}

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags and return visible text."""
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()
