"""Parser package initialization."""

from pathlib import Path
from verigraph.parsers.base import BaseParser
from verigraph.parsers.text_parser import TextParser
from verigraph.parsers.pdf_parser import PDFParser


def get_parser_for_file(file_path: Path) -> BaseParser:
    """Returns the appropriate parser based on file suffix."""
    suffix = file_path.suffix.lower()
    if suffix in [".pdf"]:
        return PDFParser()
    return TextParser()


__all__ = ["BaseParser", "TextParser", "PDFParser", "get_parser_for_file"]
