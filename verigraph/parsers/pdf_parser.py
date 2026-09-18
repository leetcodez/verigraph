"""PDF document parser extracting structured text and page boundaries."""

from pathlib import Path
from typing import List, Tuple
from pypdf import PdfReader
from verigraph.parsers.base import BaseParser
from verigraph.core.exceptions import DocumentParsingError


class PDFParser(BaseParser):
    """Parses PDF documents into page-bounded logical sections."""

    def parse(self, file_path: Path) -> Tuple[str, List[Tuple[str, List[str], int]]]:
        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            raise DocumentParsingError(f"Failed to parse PDF file {file_path}: {e}") from e

        full_text_parts: List[str] = []
        sections: List[Tuple[str, List[str], int]] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            page_text = page_text.strip()
            if page_text:
                full_text_parts.append(page_text)
                # Split page into paragraphs
                paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                for para in paragraphs:
                    sections.append((para, [f"Page {page_idx}"], page_idx))

        full_text = "\n\n".join(full_text_parts)
        if not sections and full_text.strip():
            sections.append((full_text.strip(), ["Page 1"], 1))

        return full_text, sections
