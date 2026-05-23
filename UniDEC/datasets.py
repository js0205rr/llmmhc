"""
XML Dataset implementation for extreme classification.

This module provides PyTorch dataset and sampler implementations for extreme
classification tasks, including support for dynamic batch sizes and label sampling.
"""

import os
import torch
import numpy as np
import torch.sparse
from torch.utils.data import Dataset, Sampler
from xclib.data import data_utils as du
import scipy.sparse as sp
from xclib.utils.sparse import retain_topk
from typing import Union, Iterable, Sized, List, Iterator, Dict, Optional, Tuple


class MySampler(Sampler[int]):
    """
    Custom sampler that uses a predefined order from a memory-mapped file.
    
    This sampler is useful when you want to maintain a specific order of samples
    across training sessions without loading the entire order into memory.
    """
    
    def __init__(self, data_source: Sized, fname: str):
        """
        Initialize sampler.
        
        Args:
            data_source: Dataset to sample from
            fname: Path to memory-mapped file containing sample order
        """
        self.data_source = data_source
        self.order = np.memmap(fname, dtype=np.int32, mode='r', shape=(len(data_source),))
        assert len(self.order) == len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        """Return iterator over sample indices in predefined order."""
        return iter(self.order)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data_source)


class BatchSampler(Sampler[List[int]]):
    """
    Batch sampler that supports dynamic batch sizes.
    
    This sampler allows for different batch sizes for different batches,
    specified through a memory-mapped file.
    """
    
    def __init__(self, 
                 sampler: Union[Sampler[int], Iterable[int]], 
                 batch_size: str, 
                 drop_last: bool) -> None:
        """
        Initialize batch sampler.
        
        Args:
            sampler: Base sampler to use
            batch_size: Path to memory-mapped file containing batch sizes
            drop_last: Whether to drop the last incomplete batch
        """
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                           f"drop_last={drop_last}")
        self.sampler = sampler
        self.batch_sizes = np.memmap(batch_size, dtype=np.int32, mode='r')
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        """Return iterator over batches."""
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_sizes)]
                    yield batch
                except StopIteration:
                    break
        else:
            b_i = 0
            batch = [0] * self.batch_sizes[b_i]
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_sizes[b_i]:
                    yield batch
                    idx_in_batch = 0
                    b_i += 1
                    if b_i != len(self.batch_sizes):
                        batch = [0] * self.batch_sizes[b_i]
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batch_sizes)


class DocDataset(Dataset):
    """
    Base dataset class for document data.
    
    This class handles the basic document data structure and collation.
    """
    
    def __init__(self, docs: Dict[str, torch.Tensor]):
        """
        Initialize dataset.
        
        Args:
            docs: Dictionary containing document input_ids and attention_mask
        """
        self.docs = docs
    
    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.docs['input_ids'])
    
    def __getitem__(self, idx: int) -> int:
        """Return document index."""
        return idx 
    
    def collate_fn(self, batch: List[int]) -> Dict:
        """
        Collate batch of documents.
        
        Args:
            batch: List of document indices
            
        Returns:
            Dictionary containing collated documents and indices
        """
        collated = {'docs': None}
        indices = np.array([x for x in batch])
        collated['docs'] = {
            'input_ids': self.docs['input_ids'][indices],
            'attention_mask': self.docs['attention_mask'][indices]
        }
        collated['indices'] = indices
        return collated


class XMLTestDataset(DocDataset):
    """
    Dataset class for XML test data.
    
    This class extends DocDataset to include test labels and filtering functionality.
    """
    
    def __init__(self, 
                 docs: Dict[str, torch.Tensor], 
                 XY: sp.spmatrix, 
                 data_path: str):
        """
        Initialize test dataset.
        
        Args:
            docs: Dictionary containing document input_ids and attention_mask
            XY: Sparse matrix of document-label associations
            data_path: Path to data directory
        """
        super(XMLTestDataset, self).__init__(docs)
        
        # Load test labels
        self.test_labels = [
            torch.from_numpy(x) 
            for x in np.split(XY.indices, XY.indptr[1:-1])
        ]
        
        # Load filter file if it exists
        filter_file = os.path.join(data_path, 'filter_labels_test.txt')
        if os.path.exists(filter_file):
            filter_test = np.loadtxt(filter_file).astype(np.int64)
            rows, cols, data = (
                filter_test[:, 0], 
                filter_test[:, 1], 
                [1]*filter_test.shape[0]
            )
            filter_test = sp.csr_matrix(
                (data, (rows, cols)), 
                shape=(XY.shape[0], XY.shape[1])
            )

            self.filter_test = {}
            for i in range(filter_test.shape[0]):
                if len(filter_test[i].indices):
                    self.filter_test[i] = torch.from_numpy(filter_test[i].indices)
            print("Loaded filter test file.")
        else:
            print("Filter test file not found in the dataset folder.")


class QCDataset(Dataset):
    """
    Dataset class for Query-Context pairs with label sampling.
    
    This class implements label sampling strategies for training, including
    hard negative mining and dynamic label pool management.
    """
    
    def __init__(self, 
                 docs: Dict[str, torch.Tensor], 
                 lbls: Dict[str, torch.Tensor], 
                 XY: sp.spmatrix, 
                 params):
        """
        Initialize QC dataset.
        
        Args:
            docs: Dictionary containing document input_ids and attention_mask
            lbls: Dictionary containing label input_ids and attention_mask
            XY: Sparse matrix of document-label associations
            params: Parameters for label sampling and batch construction
        """
        self.docs = docs
        self.lbls = lbls
        self.XY = XY
        
        # Load hard negatives if available
        if params.num_negs > 0:
            self.negatives_file = params.model_dir + '/hard_negatives.npy'
            if os.path.exists(self.negatives_file):
                print("Loading existing hard negatives")
                self.reload_negatives()
                
        self.num_negs = params.num_negs
        self.num_pos = params.num_pos
        self.freq = np.power(np.sum(XY, axis=1).A.squeeze(), -0.5)
        self.label_pool_size = params.label_pool_size
        self.neg_pool_size = (params.num_negs * params.cl_update)
        self.min_batch_gap = params.fill_batch_gap
    
    def __len__(self) -> int:
        """Return number of documents."""
        return self.XY.shape[0]
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get document and its associated labels.
        
        Args:
            idx: Document index
            
        Returns:
            Dictionary containing document index, positive labels, and negative labels
        """
        doc_idx = idx
        labels = self.XY[doc_idx].indices
        sel_pos_lbls = np.random.choice(labels, self.num_pos)

        # Sample negative labels
        if hasattr(self, 'negatives'):
            neg_mask = ~np.isin(self.negatives[doc_idx], labels)
            neg_pool = self.negatives[doc_idx][neg_mask][:self.neg_pool_size]
            
            try:
                neg_lbls = np.random.choice(neg_pool, self.num_negs, replace=False)
            except ValueError:  # Not enough samples for replace=False
                neg_lbls = np.random.choice(neg_pool, self.num_negs, replace=True)
        else:
            neg_lbls = np.array([])

        return {
            'doc_idx': doc_idx,
            'pos_lbls': labels,
            'neg_lbls': neg_lbls,
            'sel_pos_lbls': sel_pos_lbls
        }

    def reload_negatives(self):
        """Reload hard negative examples from file."""
        self.negatives = np.load(self.negatives_file).astype(np.int32)
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of documents and labels.
        
        Args:
            batch: List of dictionaries containing document and label information
            
        Returns:
            Dictionary containing collated documents, labels, and targets
        """
        # Get document indices
        batch_docs = np.array([x['doc_idx'] for x in batch])
        
        # Get unique selected positive labels
        batch_labels, batch_stats = np.unique(
            np.concatenate([x['sel_pos_lbls'] for x in batch], axis=None),
            return_counts=True
        )

        # Handle label pool size constraints
        batch_gap = self.label_pool_size - len(batch_labels)
        if batch_gap < 0:
            pos_impt = np.argsort(-batch_stats)
            batch_labels = batch_labels[pos_impt][:batch_gap]
        
        # Add negative labels if needed
        elif hasattr(self, 'negatives') and batch_gap > self.min_batch_gap:
            neg_batch_lbls, neg_stats = np.unique(
                np.concatenate([x['neg_lbls'] for x in batch], axis=None),
                return_counts=True
            )
            neg_mask = np.isin(neg_batch_lbls, batch_labels, invert=True)
            neg_batch_lbls, neg_stats = neg_batch_lbls[neg_mask], neg_stats[neg_mask]
            
            if len(neg_batch_lbls) > batch_gap:
                neg_impt = np.argsort(-neg_stats)
                neg_batch_lbls = neg_batch_lbls[neg_impt][:batch_gap]
            batch_labels = np.concatenate((batch_labels, neg_batch_lbls))

        # Add extra positive labels if needed
        batch_gap = self.label_pool_size - len(batch_labels)
        if batch_gap > self.min_batch_gap:
            all_pos_lbls, all_pos_stats = np.unique(
                np.concatenate([x['pos_lbls'] for x in batch], axis=None),
                return_counts=True
            )
            pos_mask = np.isin(all_pos_lbls, batch_labels, invert=True)
            all_pos_lbls, all_pos_stats = all_pos_lbls[pos_mask], all_pos_stats[pos_mask]
            pos_impt = np.argsort(-all_pos_stats)
            extra_pos_lbls = all_pos_lbls[pos_impt][:batch_gap]
            batch_labels = np.concatenate((batch_labels, extra_pos_lbls))

        # Create target matrix
        target_cnst = np.zeros((len(batch_docs), len(batch_labels)), dtype=np.float32)
        positive_labels = []
        
        for i, b in enumerate(batch):
            positive_labels.append(torch.tensor(b['pos_lbls']))
            target_cnst[i] = np.isin(batch_labels, b['pos_lbls']).astype(np.float32)

        # Filter out documents with no labels
        lbls_per_doc = target_cnst.sum(1)
        doc_mask = lbls_per_doc > 0

        # Prepare document and label tensors
        doc_ii = self.docs['input_ids'][batch_docs][doc_mask]
        doc_am = self.docs['attention_mask'][batch_docs][doc_mask]        
        target_cnst = torch.from_numpy(target_cnst[doc_mask]) 
        lbl_ii = self.lbls['input_ids'][batch_labels]
        lbl_am = self.lbls['attention_mask'][batch_labels]

        # Return collated batch
        return {
            'target': {
                'target_cnst': target_cnst,
                'lbls': positive_labels,
                'cnst_labels': torch.from_numpy(batch_labels)
            },
            'docs': {
                'input_ids': doc_ii,
                'attention_mask': doc_am
            },
            'lbls': {
                'input_ids': lbl_ii,
                'attention_mask': lbl_am
            }
        }
