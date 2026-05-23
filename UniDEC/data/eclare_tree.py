"""
ECLARE Tree Implementation for Extreme Classification

This module implements a tree-based label organization system specifically designed for ECLARE
(Extreme Classification with Label Embeddings). It provides efficient label clustering and
tree building capabilities optimized for extreme classification tasks.

Key Features:
- Multi-objective label clustering
- Parallel processing support
- Efficient tree building and merging
- Support for both dense and sparse label features

Example:
    >>> from eclare_tree import BuildTree
    >>> tree = BuildTree(b_factors=[2], M=1, method='random')
    >>> tree.fit(label_index=[np.array([0,1,2])], lbl_repr=np.random.rand(3, 10))
"""

import numpy as np
import time
import scipy.sparse as sp
from sklearn.preprocessing import normalize as scale
from functools import partial, reduce
import operator
import _pickle as pik
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Union, Optional, Callable

# Type aliases for better readability
LabelFeatures = Union[np.ndarray, sp.spmatrix]
LabelIndices = List[np.ndarray]
ClusterResult = Tuple[LabelIndices, LabelIndices]

# Constants
DEFAULT_TOLERANCE = 1e-4
DEFAULT_METRIC = 'cosine'
MIN_CLUSTER_SIZE = 1

def _normalize(X: np.ndarray, norm: str = 'l2') -> np.ndarray:
    """
    Normalize input features using specified norm.
    
    Args:
        X: Input feature matrix of shape (n_samples, n_features)
        norm: Normalization method ('l2' by default)
    
    Returns:
        Normalized feature matrix of same shape as input
    
    Raises:
        ValueError: If norm is not supported
    """
    if norm not in ['l1', 'l2', 'max']:
        raise ValueError(f"Unsupported norm: {norm}. Must be one of ['l1', 'l2', 'max']")
    return scale(X, norm=norm)


def _initialize_clusters(n_samples: int) -> np.ndarray:
    """
    Initialize two random cluster centers.
    
    Args:
        n_samples: Number of samples to choose from
        
    Returns:
        Array of two distinct cluster indices
    """
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for clustering")
    
    cluster = np.random.randint(low=0, high=n_samples, size=(2))
    while cluster[0] == cluster[1]:
        cluster = np.random.randint(low=0, high=n_samples, size=(2))
    return cluster


def _compute_similarity(features: LabelFeatures, centroids: np.ndarray) -> np.ndarray:
    """
    Compute similarity between features and centroids.
    
    Args:
        features: Input features (dense or sparse)
        centroids: Cluster centroids
        
    Returns:
        Similarity matrix
    """
    if isinstance(features, sp.spmatrix):
        return _sdist(features, centroids)
    return np.dot(features, centroids.T)


def _update_centroids(features: LabelFeatures, clusters: List[np.ndarray]) -> np.ndarray:
    """
    Update cluster centroids based on current assignments.
    
    Args:
        features: Input features
        clusters: List of cluster assignments
        
    Returns:
        Updated centroids
    """
    if isinstance(features, sp.spmatrix):
        centroids = np.vstack([features[x, :].mean(axis=0) for x in clusters])
    else:
        centroids = np.vstack([np.mean(features[x, :], axis=0) for x in clusters])
    return _normalize(centroids)


def b_kmeans_base(
    features: LabelFeatures,
    index: np.ndarray,
    metric: str = DEFAULT_METRIC,
    tol: float = DEFAULT_TOLERANCE,
    leakage: Optional[float] = None
) -> List[np.ndarray]:
    """
    Base binary k-means clustering implementation.
    
    Args:
        features: Label features (dense or sparse)
        index: Label indices
        metric: Distance metric
        tol: Convergence tolerance
        leakage: Optional leakage parameter
    
    Returns:
        List of two clusters containing label indices
        
    Raises:
        ValueError: If features are empty or invalid
    """
    if features.shape[0] == 0:
        raise ValueError("Empty feature matrix provided")
        
    features = _normalize(features)
    if features.shape[0] == MIN_CLUSTER_SIZE:
        return [index]
        
    cluster = _initialize_clusters(features.shape[0])
    centroids = features[cluster]
    similarity = _compute_similarity(features, centroids)
    
    old_sim, new_sim = -float('inf'), -2
    while new_sim - old_sim >= tol:
        clustered_lbs = np.array_split(
            np.argsort(similarity[:, 1] - similarity[:, 0]), 2)
        centroids = _update_centroids(features, clustered_lbs)
        similarity = _compute_similarity(features, centroids)
        old_sim, new_sim = new_sim, np.sum([
            np.sum(similarity[indx, i]) for i, indx in enumerate(clustered_lbs)
        ])
    
    return list(map(lambda x: index[x], clustered_lbs))


def b_kmeans_dense_multi(
    fts_lbl: np.ndarray,
    index: np.ndarray,
    metric: str = DEFAULT_METRIC,
    tol: float = DEFAULT_TOLERANCE,
    leakage: Optional[float] = None
) -> List[np.ndarray]:
    """
    Binary k-means clustering for multi-objective label features.
    
    Args:
        fts_lbl: Label features with shape (n_labels, 2, n_features)
        index: Label indices
        metric: Distance metric
        tol: Convergence tolerance
        leakage: Optional leakage parameter
    
    Returns:
        List of two clusters containing label indices
    """
    lbl_cent = _normalize(np.squeeze(fts_lbl[:, 0, :]))
    lbl_fts = _normalize(np.squeeze(fts_lbl[:, 1, :]))
    
    if lbl_cent.shape[0] == MIN_CLUSTER_SIZE:
        return [index]
        
    cluster = _initialize_clusters(lbl_cent.shape[0])
    centroids = lbl_cent[cluster]
    similarity = np.dot(lbl_cent, centroids.T)
    
    old_sim, new_sim = -float('inf'), -2
    while new_sim - old_sim >= tol:
        c_lbs = np.array_split(np.argsort(similarity[:, 1] - similarity[:, 0]), 2)
        centroids = _normalize(np.vstack([
            np.mean(lbl_cent[x, :], axis=0) for x in c_lbs
        ]))
        similarity = np.dot(lbl_cent, centroids.T)
        centroids = _normalize(np.vstack([
            np.mean(lbl_fts[x, :], axis=0) for x in c_lbs
        ]))
        similarity += np.dot(lbl_fts, centroids.T)
        old_sim, new_sim = new_sim, np.sum([
            np.sum(similarity[c_lbs[0], 0]),
            np.sum(similarity[c_lbs[1], 1])
        ])
    
    return list(map(lambda x: index[x], c_lbs))


def b_kmeans_dense(
    labels_features: np.ndarray,
    index: np.ndarray,
    metric: str = DEFAULT_METRIC,
    tol: float = DEFAULT_TOLERANCE,
    leakage: Optional[float] = None
) -> List[np.ndarray]:
    """
    Binary k-means clustering for dense label features.
    
    Args:
        labels_features: Label feature matrix
        index: Label indices
        metric: Distance metric
        tol: Convergence tolerance
        leakage: Optional leakage parameter
    
    Returns:
        List of two clusters containing label indices
    """
    return b_kmeans_base(labels_features, index, metric, tol, leakage)


def b_kmeans_sparse(
    labels_features: sp.spmatrix,
    index: np.ndarray,
    metric: str = DEFAULT_METRIC,
    tol: float = DEFAULT_TOLERANCE,
    leakage: Optional[float] = None
) -> List[np.ndarray]:
    """
    Binary k-means clustering for sparse label features.
    
    Args:
        labels_features: Sparse label feature matrix
        index: Label indices
        metric: Distance metric
        tol: Convergence tolerance
        leakage: Optional leakage parameter
    
    Returns:
        List of two clusters containing label indices
    """
    return b_kmeans_base(labels_features, index, metric, tol, leakage)


def _sdist(XA: sp.spmatrix, XB: np.ndarray, norm: Optional[str] = None) -> np.ndarray:
    """
    Compute sparse distance between matrices.
    
    Args:
        XA: Sparse matrix
        XB: Dense matrix
        norm: Optional normalization method
    
    Returns:
        Distance matrix
    """
    return XA.dot(XB.transpose())


def _merge_tree(
    cluster: LabelIndices,
    verbose_label_index: LabelIndices,
    avg_size: int = 0,
    force: bool = False
) -> ClusterResult:
    """
    Merge tree clusters with verbose labels.
    
    Args:
        cluster: List of clusters
        verbose_label_index: List of verbose label indices
        avg_size: Average cluster size
        force: Whether to force merging
    
    Returns:
        Tuple of (merged clusters, remaining verbose labels)
        
    Raises:
        ValueError: If input arrays are invalid
    """
    if not cluster or not verbose_label_index:
        raise ValueError("Empty cluster or verbose label index provided")
        
    if cluster[0].size < verbose_label_index[0].size:
        print(f"Merging trees at depth {np.log2(len(cluster))}")
        return cluster + verbose_label_index, [np.asarray([])]
    elif verbose_label_index[0].size > 0 and force:
        if verbose_label_index:
            print("Force merging trees")
            return cluster + verbose_label_index, [np.asarray([])]
        else:
            print("No more trees to merge")
            return cluster, [np.asarray([])]
    else:
        return cluster, verbose_label_index


def cluster_labels(
    labels: Union[LabelFeatures, List[LabelFeatures]],
    clusters: LabelIndices,
    verbose_label_index: LabelIndices,
    num_nodes: int,
    splitter: Callable
) -> ClusterResult:
    """
    Cluster labels using parallel processing.
    
    Args:
        labels: Label features or list of label features
        clusters: Current clusters
        verbose_label_index: Verbose label indices
        num_nodes: Number of nodes to create
        splitter: Function to split clusters
    
    Returns:
        Tuple of (new clusters, updated verbose labels)
        
    Raises:
        ValueError: If invalid input parameters
        RuntimeError: If clustering fails
    """
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be positive")
        
    start = time.time()
    clusters, verbose_label_index = _merge_tree(clusters, verbose_label_index)
    
    try:
        with Pool(cpu_count()-1) as p:
            while len(clusters) < num_nodes:
                if isinstance(labels, list):
                    temp_cluster_list = reduce(
                        operator.iconcat,
                        p.starmap(splitter, map(lambda x: (labels[0][x], labels[1][x], x), clusters)), [])
                else:    
                    temp_cluster_list = reduce(
                        operator.iconcat,
                        p.starmap(splitter, map(lambda x: (labels[x], x), clusters)), [])

                end = time.time()
                print(f"Total clusters: {len(temp_cluster_list)}")
                print(f"Avg. Cluster size: {np.mean(list(map(len, temp_cluster_list+verbose_label_index)))}")
                print(f"Total time: {end-start:.2f} sec")
                
                clusters = temp_cluster_list
                clusters, verbose_label_index = _merge_tree(clusters, verbose_label_index)
                del temp_cluster_list
    except Exception as e:
        raise RuntimeError(f"Clustering failed: {str(e)}")
    
    return clusters, verbose_label_index


def representative(lbl_fts: np.ndarray) -> np.ndarray:
    """
    Find representative label features.
    
    Args:
        lbl_fts: Label features of shape (n_samples, n_features)
    
    Returns:
        Representative feature vector of shape (n_features,)
        
    Raises:
        ValueError: If input features are invalid
    """
    if lbl_fts.shape[0] == 0:
        raise ValueError("Empty feature matrix provided")
        
    scores = np.ravel(np.sum(np.dot(lbl_fts, lbl_fts.T), axis=1))
    return lbl_fts[np.argmax(scores)]


class HashMapIndex:
    """
    Hash map for label indices with optional padding.
    """
    
    def __init__(
        self,
        clusters: Optional[List[np.ndarray]],
        label_to_idx: np.ndarray,
        total_elements: int,
        total_valid_nodes: int,
        padding_idx: Optional[int] = None
    ):
        """
        Initialize hash map index.
        
        Args:
            clusters: List of clusters
            label_to_idx: Label to index mapping
            total_elements: Total number of elements
            total_valid_nodes: Total number of valid nodes
            padding_idx: Optional padding index
        """
        self.clusters = clusters
        self.padding_idx = padding_idx
        self.total_elements = total_elements
        self.size = total_valid_nodes
        self.weights = None
        
        if padding_idx is not None:
            self.weights = np.zeros((self.total_elements), np.float)
            self.weights[label_to_idx == padding_idx] = -np.inf
        
        self.hash_map = label_to_idx

    def _get_hash(self) -> np.ndarray:
        """Get hash map."""
        return self.hash_map

    def _get_weights(self) -> Optional[np.ndarray]:
        """Get weights."""
        return self.weights


class BuildTree:
    """
    Tree builder for label organization.
    
    This class implements a tree-based label organization system for extreme classification.
    It supports both dense and sparse label features, and provides efficient clustering
    and tree building capabilities.
    
    Attributes:
        b_factors (List[int]): Branching factors for each level
        C (List[int]): Maximum cluster sizes at each level
        method (str): Clustering method to use
        leaf_size (int): Maximum leaf size
        force_shallow (bool): Whether to force shallow trees
        height (int): Tree height
        num_labels (int): Total number of labels
        hash_map_array (List[HashMapIndex]): Array of hash maps for each level
        max_node_idx (int): Maximum node index
        
    Example:
        >>> tree = BuildTree(b_factors=[2], M=1, method='random')
        >>> tree.fit(label_index=[np.array([0,1,2])], lbl_repr=np.random.rand(3, 10))
    """
    
    def __init__(
        self,
        b_factors: List[int] = [2],
        M: int = 1,
        method: str = 'random',
        leaf_size: int = 0,
        force_shallow: bool = True
    ):
        """
        Initialize tree builder.
        
        Args:
            b_factors: Branching factors for each level
            M: Number of trees
            method: Clustering method ('random' or 'NoCluster')
            leaf_size: Maximum leaf size
            force_shallow: Whether to force shallow trees
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if not b_factors or any(f <= 0 for f in b_factors):
            raise ValueError("Branching factors must be positive")
        if M <= 0:
            raise ValueError("Number of trees must be positive")
        if method not in ['random', 'NoCluster']:
            raise ValueError("Method must be 'random' or 'NoCluster'")
            
        self.b_factors = b_factors
        self.C: List[int] = []
        self.method = method
        self.leaf_size = leaf_size
        self.force_shallow = force_shallow
        self.height = 2
        self.num_labels = 0
        self.hash_map_array: List[HashMapIndex] = []
        self.max_node_idx = 0

    def fit(
        self,
        label_index: LabelIndices = [],
        verbose_label_index: LabelIndices = [],
        lbl_repr: Optional[LabelFeatures] = None
    ) -> None:
        """
        Fit tree to label features.
        
        Args:
            label_index: Label indices
            verbose_label_index: Verbose label indices
            lbl_repr: Label representations
            
        Raises:
            ValueError: If invalid input parameters
            RuntimeError: If tree building fails
        """
        if lbl_repr is None:
            raise ValueError("Label representations must be provided")
        if not label_index:
            raise ValueError("Label indices must be provided")
            
        self.num_labels = lbl_repr.shape[0]
        clusters = [label_index]
        self.hash_map_array = []
        print(f"Total verbose labels: {verbose_label_index.size}")

        # Select appropriate clustering method
        try:
            if len(lbl_repr.shape) > 2:
                print("Using multi objective kmeans++")
                b_kmeans = b_kmeans_dense_multi
            elif isinstance(lbl_repr, np.ndarray):
                print("Using dense kmeans++")
                b_kmeans = b_kmeans_dense
            else:
                lbl_repr = lbl_repr.tocsr()
                b_kmeans = b_kmeans_sparse

            if self.method == "NoCluster":
                self.height = 1
                print("No need to create splits")
                n_lb = self.num_labels
                self.hash_map_array.append(HashMapIndex(
                    None, np.concatenate(clusters), n_lb, n_lb, n_lb))
                return

            self._parabel(lbl_repr, clusters, [verbose_label_index],
                          b_kmeans, self.force_shallow)
        except Exception as e:
            raise RuntimeError(f"Tree building failed: {str(e)}")

    def _parabel(
        self,
        labels: LabelFeatures,
        clusters: LabelIndices,
        verbose_label_index: LabelIndices,
        splitter: Callable,
        force_shallow: bool
    ) -> None:
        """
        Build tree using Parabel algorithm.
        
        Args:
            labels: Label features
            clusters: Current clusters
            verbose_label_index: Verbose label indices
            splitter: Function to split clusters
            force_shallow: Whether to force shallow trees
            
        Raises:
            RuntimeError: If tree building fails
        """
        try:
            depth = 0
            T_verb_lbl = verbose_label_index[0].size
            
            while True:
                # Calculate number of nodes
                original_num_nodes = 2**self.b_factors[depth]
                n_child_nodes = original_num_nodes
                
                if self.num_labels/n_child_nodes < T_verb_lbl or len(self.b_factors) == 1:
                    if T_verb_lbl > 0:
                        add_at = np.floor(np.log2(self.num_labels/T_verb_lbl))+1
                        addition = 2**(self.b_factors[depth]-add_at)
                        n_child_nodes += addition
                
                depth += 1
                print(f"Building tree at height {depth} with nodes: {n_child_nodes}")
                
                if n_child_nodes >= self.num_labels:
                    print("No need to do clustering")
                    clusters = list(np.arange(self.num_labels).reshape(-1, 1))
                else:
                    clusters, verbose_label_index = cluster_labels(
                        labels, clusters, verbose_label_index,
                        original_num_nodes, splitter)
                    if depth == len(self.b_factors):
                        clusters, verbose_label_index = _merge_tree(
                            clusters, verbose_label_index, True)
                
                self.hash_map_array.append(
                    HashMapIndex(
                        clusters,
                        np.arange(n_child_nodes),
                        n_child_nodes,
                        n_child_nodes
                    )
                )
                self.C.append(max(list(map(lambda x: x.size, clusters))))
                
                if depth == len(self.b_factors):
                    print("Preparing Leaf")
                    break

            self.height = depth+1
            self.max_node_idx = np.int(n_child_nodes*self.C[-1])
            print(f"Building tree at height {self.height} with max leafs: {self.max_node_idx}")
            
            _labels_path_array = np.full(
                (self.max_node_idx), self.num_labels,
                dtype=np.int)
            
            for idx, c in enumerate(clusters):
                index = np.arange(c.size) + idx*self.C[-1]
                _labels_path_array[index] = clusters[idx]
            
            self.hash_map_array.append(
                HashMapIndex(None,
                            _labels_path_array,
                            self.max_node_idx,
                            self.num_labels,
                            self.num_labels))
            
            print(f"Sparsity of leaf is {((1-(self.num_labels/self.max_node_idx))*100):.2f}%")
        except Exception as e:
            raise RuntimeError(f"Parabel tree building failed: {str(e)}")

    def _get_cluster_depth(self, depth: int) -> Optional[LabelIndices]:
        """
        Get clusters at specified depth.
        
        Args:
            depth: Tree depth
            
        Returns:
            Clusters at specified depth
            
        Raises:
            ValueError: If depth is invalid
        """
        if depth < 0 or depth >= len(self.hash_map_array):
            raise ValueError(f"Invalid depth: {depth}")
        return self.hash_map_array[depth].clusters

    def load(self, fname: str) -> None:
        """
        Load tree from file.
        
        Args:
            fname: File name
            
        Raises:
            FileNotFoundError: If file doesn't exist
            RuntimeError: If loading fails
        """
        try:
            self.__dict__ = pik.load(open(fname, 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError(f"Tree file not found: {fname}")
        except Exception as e:
            raise RuntimeError(f"Failed to load tree: {str(e)}")

    def save(self, fname: str) -> None:
        """
        Save tree to file.
        
        Args:
            fname: File name
            
        Raises:
            RuntimeError: If saving fails
        """
        try:
            pik.dump(self.__dict__, open(fname, 'wb'))
        except Exception as e:
            raise RuntimeError(f"Failed to save tree: {str(e)}")
