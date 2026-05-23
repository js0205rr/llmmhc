"""
Approximate Nearest Neighbor Search (ANNS) implementations.

This module provides implementations of various ANNS algorithms for efficient similarity search:
- FaissMIPSIndex: GPU-accelerated Maximum Inner Product Search using FAISS
- HNSW: Hierarchical Navigable Small World graph for approximate nearest neighbor search

Example:
    >>> from models.anns import FaissMIPSIndex
    >>> index = FaissMIPSIndex(device=0)
    >>> index.build_index(embeddings)
    >>> results = index.search(query, k=10)
"""

import faiss
import torch
import time
import faiss.contrib.torch_utils
import torch.nn.functional as F
import hnswlib
import numpy as np
from tqdm import tqdm
import math
from typing import Tuple, Optional, Union

class FaissMIPSIndex:
    """
    GPU-accelerated Maximum Inner Product Search using FAISS.
    
    This class implements a GPU-accelerated MIPS index using FAISS library.
    It supports efficient similarity search for high-dimensional vectors.
    
    Args:
        device: CUDA device ID to use for GPU acceleration
        
    Example:
        >>> index = FaissMIPSIndex(device=0)
        >>> index.build_index(embeddings)
        >>> results = index.search(query, k=10)
    """
    
    def __init__(self, device: int):
        """
        Initialize FAISS MIPS index.
        
        Args:
            device: CUDA device ID
        """
        self.device = device
        
    def search(self, query_batch: torch.Tensor, k: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_batch: Query vectors of shape (n_queries, dim)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (indices, distances) for k nearest neighbors
        """
        dists, keys = self.anns.search(query_batch, k)
        return keys, dists

    def build_index(self, embs: torch.Tensor) -> None:
        """
        Build the FAISS index with given embeddings.
        
        Args:
            embs: Embedding vectors of shape (n_vectors, dim)
        """
        if hasattr(self, 'anns'):
            del self.anns
            
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.device
        resource = faiss.StandardGpuResources()
        self.anns = faiss.GpuIndexFlatIP(resource, embs.shape[1], cfg)
        self.anns.add(embs)
        embs = embs.cpu()
        del embs

class HNSW:
    """
    Hierarchical Navigable Small World graph for approximate nearest neighbor search.
    
    This class implements the HNSW algorithm for efficient approximate nearest neighbor search.
    It provides a good balance between search speed and accuracy.
    
    Args:
        M: Maximum number of connections per element
        efC: Size of the dynamic candidate list during construction
        efS: Size of the dynamic candidate list during search
        num_threads: Number of threads to use
        device: CUDA device ID
        
    Example:
        >>> index = HNSW(M=110, efC=100, efS=1000)
        >>> index.build_index(embeddings)
        >>> results = index.search(query, k=25)
    """
    
    def __init__(
        self,
        M: int = 110,
        efC: int = 100,
        efS: int = 1000,
        num_threads: int = 90,
        device: int = 0
    ):
        """
        Initialize HNSW index.
        
        Args:
            M: Maximum number of connections per element
            efC: Size of the dynamic candidate list during construction
            efS: Size of the dynamic candidate list during search
            num_threads: Number of threads to use
            device: CUDA device ID
        """
        self.M = M
        self.num_threads = num_threads
        self.efC = efC
        self.efS = efS
        self.device = device

    def build_index(self, data: torch.Tensor, print_progress: bool = True) -> None:
        """
        Build the HNSW index with given data.
        
        Args:
            data: Input vectors of shape (n_vectors, dim)
            print_progress: Whether to show progress bar
        """
        if hasattr(self, 'anns'):
            del self.anns
            
        data = data.cpu().numpy()
        self.anns = hnswlib.Index(space='ip', dim=data.shape[1])
        self.anns.init_index(max_elements=data.shape[0], ef_construction=self.efC, M=self.M)
        data_labels = np.arange(data.shape[0]).astype(np.int64)
        self.anns.add_items(data, data_labels, num_threads=self.num_threads)
        del data

    def search(
        self,
        query_batch: torch.Tensor,
        k: int = 25
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_batch: Query vectors of shape (n_queries, dim)
            k: Number of nearest neighbors to retrieve
            
        Returns:
            Tuple of (indices, distances) for k nearest neighbors
        """
        self.anns.set_ef(self.efS)
        k = min(k, self.efS)
        keys, dists = self.anns.knn_query(query_batch.cpu().numpy(), k=k)
        keys = keys.astype(np.int64)
        dists *= -1

        return torch.from_numpy(keys).to(self.device), torch.from_numpy(dists).to(self.device)
