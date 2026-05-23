"""
Data processing and clustering utilities for extreme classification.

This package provides tools for hierarchical label clustering and tree-based organization
of labels for extreme classification problems. It includes implementations of various
clustering algorithms and tree structures optimized for large-scale multi-label classification.

Main components:
- Hierarchical label clustering (cluster.py)
- Tree-based label organization (tree.py)
- ECLARE tree implementation (eclare_tree.py)
- Data preprocessing utilities (preprocessing.py)
- Random walk generation (random_walks.py)
- Dataset handling (datasets.py)
"""

from .cluster import build_tree_by_level, get_sparse_feature, split_node
from .tree import build_tree, hash_map_index, b_kmeans_dense, b_kmeans_sparse
from .eclare_tree import EclareTree
from .preprocessing import DataUtils
from .random_walks import RandomWalks
from .datasets import XMLDataset

__version__ = '1.0.0'

__all__ = [
    # Cluster module
    'build_tree_by_level',
    'get_sparse_feature',
    'split_node',
    
    # Tree module
    'build_tree',
    'hash_map_index',
    'b_kmeans_dense',
    'b_kmeans_sparse',
    
    # Other modules
    'EclareTree',
    'DataUtils',
    'RandomWalks',
    'XMLDataset'
] 
