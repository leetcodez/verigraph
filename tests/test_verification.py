"""Tests for Claim Decomposition, NLI Verification, and Citation Attribution."""

from verigraph.core.models import RetrievalCandidate, VerificationStatus
from verigraph.embeddings.local_embedder import LocalSemanticEmbedder
from verigraph.verification.claim_splitter import ClaimSplitter
from verigraph.verification.nli_verifier import NLIVerifier
from verigraph.verification.attribution import AttributionBuilder


def test_claim_splitter():
    splitter = ClaimSplitter(min_words=4)
    response_text = (
        "Based on the provided context: "
        "- Raft strictly requires a majority quorum to elect a leader. "
        "- In addition, Paxos uses two distinct consensus phases. "
        "What do you think?"
    )
    claims = splitter.split_into_claims(response_text)
    assert len(claims) >= 2
    assert any("majority quorum" in c for c in claims)
    assert any("Paxos uses two" in c for c in claims)
    # Question should be omitted
    assert not any("What do you think" in c for c in claims)


def test_nli_verifier_entailment_and_contradiction():
    embedder = LocalSemanticEmbedder(dimension=128)
    verifier = NLIVerifier(embedder=embedder)

    contexts = [
        RetrievalCandidate(
            chunk_id="c1",
            document_title="Raft Paper",
            content="Raft strictly prevents split-brain by requiring a majority quorum of 3 nodes out of 5.",
            score=0.9,
        )
    ]

    # Test Entailment
    entailed_claim = "Raft prevents split-brain using a majority quorum of nodes."
    ver_entailed = verifier.verify_claim(entailed_claim, contexts)
    assert ver_entailed.status == VerificationStatus.ENTAILED
    assert ver_entailed.confidence > 0.65
    assert "c1" in ver_entailed.cited_chunk_ids

    # Test Contradiction
    contradiction_claim = "Raft does not prevent split-brain during partitions."
    ver_contra = verifier.verify_claim(contradiction_claim, contexts)
    assert ver_contra.status == VerificationStatus.CONTRADICTION


def test_attribution_builder():
    builder = AttributionBuilder()
    from verigraph.core.models import ClaimVerification

    claims = [
        ClaimVerification(
            claim_text="Raft guarantees linearizability.",
            status=VerificationStatus.ENTAILED,
            confidence=0.95,
            reasoning="Direct match",
            cited_chunk_ids=["chunk_123"],
        )
    ]
    contexts = [
        RetrievalCandidate(
            chunk_id="chunk_123",
            document_title="Consensus Doc",
            content="Raft guarantees linearizability across state machine transitions.",
            score=0.9,
            heading_breadcrumbs=["Safety", "Linearizability"],
        )
    ]

    formatted_text, sources = builder.format_attributed_response(claims, contexts)
    assert "[1]" in formatted_text
    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "chunk_123"
