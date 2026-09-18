"""Decomposes generated answers into atomic, verifiable assertions."""

import re
from typing import List


class ClaimSplitter:
    """
    Decomposes generated responses into discrete propositional assertions
    suitable for granular Natural Language Inference (NLI) verification.
    """

    CONVERSATIONAL_PREFIXES = [
        r"^here\s+is\s+.*?:",
        r"^in\s+summary.*?:",
        r"^based\s+on\s+the\s+provided\s+context.*?:",
        r"^in\s+conclusion.*?:",
        r"^to\s+answer\s+your\s+question.*?:",
    ]

    def __init__(self, min_words: int = 4):
        self.min_words = min_words
        self._sentence_regex = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\-\*•])")
        self._bullet_regex = re.compile(r"^\s*[-*•\d+\.]\s*")

    def split_into_claims(self, response_text: str) -> List[str]:
        """
        Extracts testable declarative claims from a raw LLM response.
        """
        claims: List[str] = []
        # Pre-normalize inline bullets into distinct lines
        normalized_text = re.sub(r"\s+[-*•]\s+", "\n", response_text)
        raw_lines = normalized_text.splitlines()

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue

            # Strip markdown bullets or numbers: '- ', '1. '
            cleaned_line = self._bullet_regex.sub("", line).strip()

            # Remove markdown bold/italics
            cleaned_line = re.sub(r"[*_`]", "", cleaned_line)

            # Check for conversational prefixes
            for prefix in self.CONVERSATIONAL_PREFIXES:
                cleaned_line = re.sub(prefix, "", cleaned_line, flags=re.IGNORECASE).strip()

            # Split into individual sentences
            sentences = self._sentence_regex.split(cleaned_line)
            for sent in sentences:
                sent = self._bullet_regex.sub("", sent).strip().rstrip(".,;:")
                # Skip questions or short fragments
                if not sent or sent.endswith("?"):
                    continue
                words = sent.split()
                if len(words) >= self.min_words:
                    claims.append(sent)

        return claims
