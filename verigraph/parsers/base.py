"""Base parser interface for document ingestion."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> Tuple[str, List[Tuple[str, List[str], int]]]:
        """
        Parses a file and returns:
        1. Cleaned full text of document.
        2. List of sections: (section_text, heading_breadcrumbs, page_number).
        """
        pass
