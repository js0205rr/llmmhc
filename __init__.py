"""
Models package for UniDEC (Universal Document Embedding and Classification).

This package contains various model implementations for document embedding and classification:
- UniDEC: Universal Document Embedding and Classification model
- SupConDR: Supervised Contrastive Document Retrieval model
- LLM: Language Model based implementation
- ANNS: Approximate Nearest Neighbor Search implementations

Example:
    >>> from models import UniDEC, SupConDR, LLM
    >>> model = UniDEC(params)
    >>> embeddings = model.encode(documents)
"""

from .supcon import SupConDR
from .unidec import UniDEC
from .llm import LLM
from .anns import FaissMIPSIndex, HNSW

__all__ = [
    'UniDEC',
    'SupConDR', 
    'LLM',
    'FaissMIPSIndex',
    'HNSW'
]