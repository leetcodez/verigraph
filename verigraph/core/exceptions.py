"""Domain-specific exceptions for VeriGraph."""

class VeriGraphError(Exception):
    """Base exception for all VeriGraph domain errors."""
    pass

class DocumentParsingError(VeriGraphError):
    """Raised when document parsing fails."""
    pass

class VectorStoreError(VeriGraphError):
    """Raised when vector storage or similarity search fails."""
    pass

class GraphEngineError(VeriGraphError):
    """Raised when graph manipulation or Cypher traversal fails."""
    pass

class VerificationError(VeriGraphError):
    """Raised when claim verification or NLI scoring fails."""
    pass

class LLMProviderError(VeriGraphError):
    """Raised when LLM synthesis fails or upstream API errors."""
    pass
