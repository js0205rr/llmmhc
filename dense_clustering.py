"""
Dense clustering module for efficient clustering of dense embeddings.

This module provides functionality for clustering dense embeddings using binary k-means
clustering. It includes utilities for timing function execution and creating sparse
clustering matrices.

Example:
    >>> embeddings = torch.randn(1000, 64)
    >>> clustering_matrix = cluster_dense_embs(embeddings, device='cuda:0', tree_depth=9)
"""

import time
import operator
from functools import wraps, reduce
from typing import List, Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


def timeit(func: Callable) -> Callable:
    """
    Decorator to measure and print the execution time of a function.
    
    Args:
        func: Function to be timed
        
    Returns:
        Wrapped function that prints execution time
    """
    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        result = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print(f"Finished {func.__name__} in {run_time:.4f} secs")
        return result
    return wrapper_timer


def binary_kmeans_dense(
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    metric: str = 'cosine',
    tolerance: float = 1e-4,
    leakage: Optional[float] = None
) -> List[torch.Tensor]:
    """
    Perform binary k-means clustering on dense embeddings.
    
    Args:
        embeddings: Input embeddings tensor
        indices: Indices of the embeddings
        metric: Distance metric to use (currently only 'cosine' supported)
        tolerance: Convergence tolerance
        leakage: Optional leakage parameter
        
    Returns:
        List of clustered indices
    """
    with torch.no_grad():
        num_samples = embeddings.shape[0]
        
        # Handle single sample case
        if num_samples == 1:
            return [indices]
            
        # Initialize random cluster centers
        cluster_centers = np.random.randint(
            low=0,
            high=num_samples,
            size=(2)
        )
        
        # Ensure different cluster centers
        while cluster_centers[0] == cluster_centers[1]:
            cluster_centers = np.random.randint(
                low=0,
                high=num_samples,
                size=(2)
            )
        
        # Initialize centroids and similarity
        centroids = embeddings[cluster_centers]
        similarity = torch.mm(embeddings, centroids.T)
        
        # Initialize convergence tracking
        old_similarity = -1000000
        new_similarity = -2
        
        # Iterate until convergence
        while new_similarity - old_similarity >= tolerance:
            # Split samples based on similarity difference
            split_point = (similarity.shape[0] + 1) // 2
            sorted_indices = torch.argsort(similarity[:, 1] - similarity[:, 0])
            clustered_indices = torch.split(sorted_indices, split_point)
            
            # Update centroids
            centroids = F.normalize(torch.vstack([
                torch.mean(embeddings[idx, :], axis=0)
                for idx in clustered_indices
            ]))
            
            # Update similarity
            similarity = torch.mm(embeddings, centroids.T)
            
            # Update convergence tracking
            old_similarity = new_similarity
            new_similarity = sum([
                torch.sum(similarity[idx, i])
                for i, idx in enumerate(clustered_indices)
            ]).item() / num_samples
        
        # Clean up
        del similarity
        
        # Map indices to original indices
        indices = indices.to(embeddings.device)
        return list(map(lambda x: indices[x], clustered_indices))


def cluster_labels(
    embeddings: torch.Tensor,
    clusters: List[torch.Tensor],
    num_nodes: int,
    splitter: Callable
) -> List[torch.Tensor]:
    """
    Recursively cluster labels until reaching desired number of nodes.
    
    Args:
        embeddings: Input embeddings
        clusters: Initial clusters
        num_nodes: Target number of clusters
        splitter: Function to split clusters
        
    Returns:
        List of final clusters
    """
    start_time = time.time()
    
    while len(clusters) < num_nodes:
        # Split each cluster
        temp_clusters = reduce(
            operator.iconcat,
            map(lambda x: splitter(embeddings[x], x), clusters),
            []
        )
        
        # Log progress
        end_time = time.time()
        avg_cluster_size = np.mean(list(map(len, temp_clusters)))
        print(
            f"Total clusters: {len(temp_clusters)}\t"
            f"Avg. Cluster size: {avg_cluster_size:.2f}\t"
            f"Total time: {end_time - start_time:.2f} sec"
        )
        
        clusters = temp_clusters
        del temp_clusters
        
    return clusters


@timeit
def cluster_dense_embs(
    embeddings: torch.Tensor,
    device: str = 'cpu',
    tree_depth: int = 9
) -> csr_matrix:
    """
    Create clustering matrix from dense embeddings.
    
    Args:
        embeddings: Input embeddings tensor
        device: Device to perform clustering on
        tree_depth: Depth of clustering tree
        
    Returns:
        Sparse clustering matrix
    """
    print(f'Device: {device}')
    
    # Use half precision for large embeddings
    if embeddings.shape[0] >= 1_000_000:
        print(f"Num embeddings: {embeddings.shape[0]} - Using HalfTensor")
        clusters = cluster_labels(
            embeddings.half(),
            [torch.arange(embeddings.shape[0])],
            2**tree_depth,
            binary_kmeans_dense
        )
    else:
        clusters = cluster_labels(
            embeddings,
            [torch.arange(embeddings.shape[0])],
            2**tree_depth,
            binary_kmeans_dense
        )
    
    # Create sparse clustering matrix
    total_elements = sum(len(c) for c in clusters)
    clustering_matrix = csr_matrix(
        (
            np.ones(total_elements),
            torch.cat(clusters).cpu().numpy(),
            np.cumsum([0, *[len(c) for c in clusters]])
        ),
        shape=(len(clusters), embeddings.shape[0])
    )
    
    return clustering_matrix


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cluster dense embeddings')
    parser.add_argument('--model_name', type=str, help='Model directory name')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    args = parser.parse_args()
    
    # Example usage
    embeddings = torch.rand(131073, 64).to('cuda:0')
    clustering_matrix = cluster_dense_embs(
        embeddings,
        device='cuda:0',
        tree_depth=9
    )
    
    # Clean up embeddings
    embeddings = embeddings.detach().cpu().numpy()
    del embeddings
    
    # Create batch sizes
    permuted_matrix = clustering_matrix[np.random.permutation(clustering_matrix.shape[0])]
    batch_sizes = [batch.nnz for batch in permuted_matrix]
    
    # Save batch sizes
    output_path = f'./models/{args.model_name}/{args.dataset}/train_batch_size.dat'
    train_batch_sizes = np.memmap(
        output_path,
        dtype=np.int32,
        mode='w+',
        shape=(len(batch_sizes),)
    )
    train_batch_sizes[:] = batch_sizes