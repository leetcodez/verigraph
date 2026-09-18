"""Text and Markdown parser preserving hierarchical structure."""

import re
from pathlib import Path
from typing import List, Tuple
from verigraph.parsers.base import BaseParser
from verigraph.core.exceptions import DocumentParsingError


class TextParser(BaseParser):
    """Parses plain text and markdown documents preserving heading hierarchies."""

    def parse(self, file_path: Path) -> Tuple[str, List[Tuple[str, List[str], int]]]:
        """
        Parses markdown/plain text and splits it into logical sections based on headers.
        Returns:
            full_text: Entire raw content.
            sections: List of (section_content, heading_breadcrumbs, page_number).
        """
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise DocumentParsingError(f"Failed to read file {file_path}: {e}") from e

        lines = content.splitlines()
        sections: List[Tuple[str, List[str], int]] = []
        
        current_headings: List[Tuple[int, str]] = []  # (level, title)
        current_section_lines: List[str] = []
        
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

        for line in lines:
            header_match = header_pattern.match(line.strip())
            if header_match:
                # Flush previous section if it has content
                if current_section_lines:
                    text_block = "\n".join(current_section_lines).strip()
                    if text_block:
                        breadcrumbs = [h[1] for h in current_headings]
                        sections.append((text_block, breadcrumbs, 1))
                    current_section_lines = []

                level = len(header_match.group(1))
                title = header_match.group(2).strip()

                # Pop higher or equal level headers from stack
                while current_headings and current_headings[-1][0] >= level:
                    current_headings.pop()
                current_headings.append((level, title))
            else:
                current_section_lines.append(line)

        # Flush final section
        if current_section_lines:
            text_block = "\n".join(current_section_lines).strip()
            if text_block:
                breadcrumbs = [h[1] for h in current_headings]
                sections.append((text_block, breadcrumbs, 1))

        # Fallback if no markdown headers were found
        if not sections and content.strip():
            sections.append((content.strip(), ["Document Root"], 1))

        return content, sections
