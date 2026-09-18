"""Natural Language Inference (NLI) Verification Engine."""

import re
from typing import List, Optional, Set, Tuple
from verigraph.config import settings
from verigraph.core.models import ClaimVerification, RetrievalCandidate, VerificationStatus
from verigraph.embeddings.base import BaseEmbedder
import numpy as np


class NLIVerifier:
    """
    Evaluates semantic entailment, neutrality, and contradiction
    between candidate context passages and generated claims.
    """

    NEGATION_WORDS = {"not", "never", "no", "cannot", "neither", "nor", "fails", "without"}

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self._word_regex = re.compile(r"\b[a-zA-Z0-9_\-\.]+\b")
        self._num_regex = re.compile(r"\b\d+(?:\.\d+)?\b")

    def _stem(self, word: str) -> str:
        """Basic suffix stemming for inflectional match."""
        w = word.lower()
        for suffix in ["ing", "ed", "es", "s"]:
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                return w[:-len(suffix)]
        return w

    def _tokenize(self, text: str) -> Set[str]:
        cleaned = text.replace("-", " ").replace("_", " ")
        words = self._word_regex.findall(cleaned)
        return {self._stem(w) for w in words if len(w) > 2 and w.lower() not in self.NEGATION_WORDS}

    def _extract_numbers(self, text: str) -> Set[str]:
        return set(self._num_regex.findall(text))

    def _check_contradiction(self, claim: str, context: str) -> Tuple[bool, str]:
        """
        Heuristic contradiction check for polarity flips and conflicting numbers.
        """
        claim_nums = self._extract_numbers(claim)
        context_nums = self._extract_numbers(context)

        # If claim asserts a specific number that directly conflicts with the context
        if claim_nums and context_nums and not (claim_nums & context_nums):
            # Check if claim and context share the same subject
            shared_words = self._tokenize(claim) & self._tokenize(context)
            if len(shared_words) >= 3:
                return True, f"Numerical discrepancy: Claim cites {claim_nums} vs Context {context_nums}"

        # Polarity conflict: claim explicitly negates facts affirmed in context
        claim_lower = claim.lower()
        has_direct_negation = bool(re.search(r"\b(not|never|cannot|no longer|fails to)\b", claim_lower))
        if has_direct_negation:
            shared = self._tokenize(claim) & self._tokenize(context)
            if len(shared) >= 3:
                return True, "Polarity conflict: claim negates a fact affirmed in the context"

        return False, ""

    def verify_claim(
        self,
        claim: str,
        retrieved_contexts: List[RetrievalCandidate],
    ) -> ClaimVerification:
        """
        Evaluates a single claim against the retrieved context passages.
        """
        if not retrieved_contexts:
            return ClaimVerification(
                claim_text=claim,
                status=VerificationStatus.NEUTRAL,
                confidence=0.1,
                reasoning="No context passages available for verification.",
            )

        claim_vec = np.array(self.embedder.embed_text(claim), dtype=np.float32)
        claim_tokens = self._tokenize(claim)

        best_score = 0.0
        best_candidate: Optional[RetrievalCandidate] = None
        best_reasoning = ""
        is_contradiction = False
        contradiction_reason = ""

        for cand in retrieved_contexts:
            context_text = cand.content
            # 1. Contradiction check
            has_contra, contra_msg = self._check_contradiction(claim, context_text)
            if has_contra:
                is_contradiction = True
                contradiction_reason = contra_msg
                best_candidate = cand
                break

            # 2. Semantic vector similarity
            ctx_vec = np.array(self.embedder.embed_text(context_text[:512]), dtype=np.float32)
            c_norm = np.linalg.norm(claim_vec)
            p_norm = np.linalg.norm(ctx_vec)
            cosine_sim = float(np.dot(claim_vec, ctx_vec) / (c_norm * p_norm)) if (c_norm * p_norm) > 0 else 0.0

            # 3. Lexical keyword recall
            ctx_tokens = self._tokenize(context_text)
            overlap = len(claim_tokens & ctx_tokens)
            recall = overlap / max(len(claim_tokens), 1)

            # Composite alignment score
            composite = (cosine_sim * 0.6) + (recall * 0.4)

            if composite > best_score:
                best_score = composite
                best_candidate = cand
                cand_cid = getattr(cand, "chunk_id", getattr(cand, "id", "unknown"))
                best_reasoning = (
                    f"Aligned with '{cand.document_title}' (Chunk {cand_cid[:8]}) "
                    f"[Cosine: {cosine_sim:.2f}, Token Recall: {recall:.2f}]"
                )

        best_cid = getattr(best_candidate, "chunk_id", getattr(best_candidate, "id", "unknown")) if best_candidate else None

        # Map to VerificationStatus
        if is_contradiction:
            return ClaimVerification(
                claim_text=claim,
                status=VerificationStatus.CONTRADICTION,
                confidence=0.88,
                reasoning=contradiction_reason,
                cited_chunk_ids=[best_cid] if best_cid else [],
                citation_badge="❌ CONTRADICTION",
            )
        elif best_score >= settings.nli_entailment_threshold and best_candidate:
            return ClaimVerification(
                claim_text=claim,
                status=VerificationStatus.ENTAILED,
                confidence=round(min(best_score, 0.99), 3),
                reasoning=best_reasoning,
                cited_chunk_ids=[best_cid] if best_cid else [],
                citation_badge=f"✅ Verified [{best_candidate.document_title}]",
            )
        else:
            return ClaimVerification(
                claim_text=claim,
                status=VerificationStatus.NEUTRAL,
                confidence=round(best_score, 3),
                reasoning="Claim contains facts not fully corroborated by retrieved context.",
                cited_chunk_ids=[best_cid] if (best_cid and best_score > 0.4) else [],
                citation_badge="⚠️ UNVERIFIED",
            )

    def verify_response(
        self,
        claims: List[str],
        retrieved_contexts: List[RetrievalCandidate],
    ) -> Tuple[List[ClaimVerification], float]:
        """
        Verifies all claims in a response and calculates the overall Faithfulness score.
        Faithfulness = (Count of Entailed Claims) / (Total Claims)
        """
        if not claims:
            return [], 1.0

        verifications: List[ClaimVerification] = []
        entailed_count = 0

        for claim in claims:
            ver = self.verify_claim(claim, retrieved_contexts)
            verifications.append(ver)
            if ver.status == VerificationStatus.ENTAILED:
                entailed_count += 1

        faithfulness = round(entailed_count / len(claims), 3)
        return verifications, faithfulness
