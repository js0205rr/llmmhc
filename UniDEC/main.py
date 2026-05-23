"""
Main module for training and evaluating document embedding models.

This module provides the main entry point for training and evaluating various document
embedding models including UniDEC, SupConDR, and LLM. It supports both single and
multi-GPU training using the accelerate library.

Example:
    >>> python main.py --config-name=default
"""

# Standard library imports
import os
import math
import warnings
from typing import Optional, Tuple, Any

# Third-party imports
import torch
import torch.multiprocessing
import numpy as np
import scipy.sparse as sp
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Local application imports
from data.preprocessing import create_data, load_data
from data.datasets import (
    QCDataset,
    XMLTestDataset,
    DocDataset,
    BatchSampler,
    MySampler
)
from trainer import Runner, ClusteringConfig
from dense_clustering import cluster_dense_embs

# Set multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')


def setup_model_directory(config: DictConfig) -> str:
    """
    Create and return the model directory path.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        Path to model directory
    """
    model_dir = os.path.join(
        os.getcwd(),
        'models',
        config.model.encoder,
        config.data.dataset,
        config.model.version
    )
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def initialize_accelerator(config: DictConfig) -> Accelerator:
    """
    Initialize and return the accelerator for distributed training.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        Initialized accelerator
    """
    ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
    return Accelerator(
        kwargs_handlers=[ddp_handler],
        even_batches=False,
        mixed_precision=config.hardware.mixed_precision
    )


def load_or_create_training_order(
    config: DictConfig,
    accelerator: Accelerator,
    train_dataset: QCDataset,
    model_dir: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[sp.spmatrix]]:
    """
    Load existing training order or create a new one.
    
    Args:
        config: Hydra configuration object
        accelerator: Accelerator instance
        train_dataset: Training dataset
        model_dir: Model directory path
        
    Returns:
        Tuple of (train_order, train_batch_sizes, cluster_matrix)
    """
    train_order_path = os.path.join(model_dir, "train_order.dat")
    train_bs_path = os.path.join(model_dir, "train_batch_size.dat")
    
    if not accelerator.is_main_process:
        return None, None, None
        
    if os.path.exists(train_order_path) and config.model.load_model:
        accelerator.print("Loading existing training order")
        train_order = np.memmap(train_order_path, dtype=np.int32, mode='r+', shape=(len(train_dataset),))
        train_bs = np.memmap(train_bs_path, dtype=np.int32, mode='r+')
        try:
            cluster_mat = sp.load_npz(os.path.join(model_dir, "cluster_mat.npz"))
        except (FileNotFoundError, IOError):
            cluster_mat = None
    else:
        accelerator.print("Creating new training order")
        embeddings = torch.rand(train_dataset.Y.shape[0], 64).to(accelerator.device)
        tree_depth = int(math.log(train_dataset.Y.shape[0]/config.data.batch_size, 2))
        
        cluster_mat = cluster_dense_embs(
            embeddings,
            device=accelerator.device,
            tree_depth=tree_depth
        )
        
        embeddings = embeddings.detach().cpu().numpy()
        del embeddings
        
        permuted_matrix = cluster_mat[np.random.permutation(cluster_mat.shape[0])]
        batch_sizes = [batch.nnz for batch in permuted_matrix]
        
        train_bs = np.memmap(train_bs_path, dtype=np.int32, mode='w+', shape=(len(batch_sizes),))
        train_bs[:] = batch_sizes
        
        train_order = np.memmap(train_order_path, dtype=np.int32, mode='w+', shape=(len(train_dataset),))
        train_order[:] = permuted_matrix.indices
    
    return train_order, train_bs, cluster_mat


def create_dataloaders(
    config: DictConfig,
    train_dataset: QCDataset,
    test_dataset: XMLTestDataset,
    label_dataset: DocDataset,
    doc_dataset: DocDataset,
    train_order_path: str,
    train_bs_path: str
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create and return dataloaders for training and evaluation.
    
    Args:
        config: Hydra configuration object
        train_dataset: Training dataset
        test_dataset: Test dataset
        label_dataset: Label dataset
        doc_dataset: Document dataset
        train_order_path: Path to training order file
        train_bs_path: Path to training batch sizes file
        
    Returns:
        Tuple of (train_dl, test_dl, label_dl, doc_dl)
    """
    train_dl = DataLoader(
        dataset=train_dataset,
        num_workers=config.hardware.num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=config.hardware.pin_memory,
        batch_sampler=BatchSampler(
            MySampler(train_dataset, train_order_path),
            train_bs_path,
            False
        )
    )
    
    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=4*config.data.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        collate_fn=test_dataset.collate_fn,
        pin_memory=config.hardware.pin_memory
    )
    
    label_bs = 512 if config.model.pre_trained_model == "phi3" else config.hardware.label_batch_size
    label_dl = DataLoader(
        dataset=label_dataset,
        batch_size=label_bs,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        collate_fn=label_dataset.collate_fn,
        pin_memory=config.hardware.pin_memory
    )
    
    doc_dl = DataLoader(
        dataset=doc_dataset,
        batch_size=4*config.data.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        collate_fn=doc_dataset.collate_fn,
        pin_memory=config.hardware.pin_memory
    )
    
    return train_dl, test_dl, label_dl, doc_dl


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Hydra configuration object
    """
    # Initialize accelerator
    accelerator = initialize_accelerator(config)
    
    # Setup model directory
    model_dir = setup_model_directory(config)
    accelerator.print(f'Saving Model to: {model_dir}')
    
    # Set random seed
    set_seed(config.model.seed)
    accelerator.print(f"Initialized seed to {config.model.seed}")
    
    # Load or create data
    data_path = os.path.join(config.data.dir, config.data.dataset)
    accelerator.print(f"Loading {config.model.pre_trained_model} tokenized data")
    
    if config.data.create_data:
        X_train, Y_train, X_test, Y_test, X_label, inv_prop = create_data(config)
    else:
        X_train, Y_train, X_test, Y_test, X_label, inv_prop = load_data(config)
    
    # Create datasets
    train_dataset = QCDataset(X_train, X_label, Y_train, config)
    test_dataset = XMLTestDataset(X_test, Y_test, data_path)
    label_dataset = DocDataset(X_label)
    doc_dataset = DocDataset(X_train)
    
    # Load or create training order
    train_order, train_bs, cluster_mat = load_or_create_training_order(
        config,
        accelerator,
        train_dataset,
        model_dir
    )
    
    accelerator.wait_for_everyone()
    
    # Create dataloaders
    train_dl, test_dl, label_dl, doc_dl = create_dataloaders(
        config,
        train_dataset,
        test_dataset,
        label_dataset,
        doc_dataset,
        os.path.join(model_dir, "train_order.dat"),
        os.path.join(model_dir, "train_batch_size.dat")
    )
    
    # Initialize model
    model = globals()[config.model.encoder](config)
    
    # Initialize runner
    clustering_config = ClusteringConfig(
        train_order=train_order,
        train_batch_sizes=train_bs,
        cluster_matrix=cluster_mat
    )
    runner = Runner(
        [train_dl, test_dl, label_dl, doc_dl],
        accelerator,
        inv_prop,
        clustering_config,
        config
    )
    
    # Train model
    runner.train(model, config)


if __name__ == '__main__':
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run main function
    main()

# python
