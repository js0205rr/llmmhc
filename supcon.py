"""
Supervised Contrastive Document Retrieval (SupConDR) model.

This module implements a supervised contrastive learning approach for document retrieval
using sentence transformers and contrastive learning.

Example:
    >>> from models import SupConDR
    >>> model = SupConDR(params)
    >>> loss, predictions = model(documents, labels, targets)
"""

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
from models.anns import FaissMIPSIndex
from models.losses import ContrastiveLossMixin
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import Dict, Optional, Tuple, Union, Any


class SupConDR(ContrastiveLossMixin, nn.Module):
    """
    Supervised Contrastive Document Retrieval model.
    
    This class implements a supervised contrastive learning approach for document retrieval
    using sentence transformers and contrastive learning.
    
    Args:
        params: Model parameters including dataset, device, etc.
        
    Example:
        >>> model = SupConDR(params)
        >>> loss, predictions = model(documents, labels, targets)
    """
    
    def __init__(self, params: Any):
        """
        Initialize SupConDR model.
        
        Args:
            params: Model parameters
        """
        super(SupConDR, self).__init__()
        
        # Initialize model parameters
        self.embed_dim = 768
        self.dataset = params.dataset
        self.device = params.device
        self.num_proc = params.num_proc
        self.proc_id = int(str(params.device)[-1])
        self.num_labels = params.num_labels
        self.shortlist_size = params.shortlist_size
        self.anns = FaissMIPSIndex(torch.cuda.current_device())

        # Initialize model components
        self.hidden_dims = params.hidden_dims
        self.contrastive_dims = params.contrastive_dims
        self.temp = nn.Parameter(torch.tensor(params.temp))
        self.init_classifiers()
        self.loss_crit = params.DE_loss
        self.add_dual_loss = params.add_dual_loss

        print(f"Using temp = {self.temp.item()} for this model training.")

        # Initialize sentence transformer
        self.bert = SentenceTransformer("msmarco-distilbert-base-v4")
        
        # Configure optimizer parameters
        lr = params.lr
        no_decay = ['LayerNorm.weight']
        wd = 0.05
        
        self.dense_grouped_params = [
            {
                'params': [p for n, p in [*self.bert.named_parameters()]
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': wd,
                'lr': lr
            },
            {
                'params': [p for n, p in [*self.bert.named_parameters()]
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': lr
            },
            {
                'params': [*self.cnst_head[0].parameters()],
                'weight_decay': wd,
                'lr': lr*2
            }
        ]

    def init_classifiers(self) -> None:
        """Initialize contrastive head for embeddings."""
        self.cnst_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.contrastive_dims),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        nn.init.xavier_uniform_(self.cnst_head[0].weight)
        nn.init.zeros_(self.cnst_head[0].bias)

    def encode(self, x: Dict[str, torch.Tensor], is_doc: bool = False) -> torch.Tensor:
        """
        Encode input into embeddings.
        
        Args:
            x: Input dictionary containing input_ids and attention_mask
            is_doc: Whether input is documents
            
        Returns:
            Encoded embeddings
        """
        input_ids, attn_mask = x['input_ids'], x['attention_mask']

        max_len = torch.max(torch.sum(attn_mask, dim=1))
        input_ids, attn_mask = input_ids[:, :max_len], attn_mask[:, :max_len] 

        bert_out = self.bert({
            'input_ids': input_ids,
            'attention_mask': attn_mask
        })
        cnst_emb = F.normalize(self.cnst_head(bert_out['sentence_embedding']))
        return cnst_emb

    def get_dataset_embeddings(
        self,
        dataloader: Any,
        is_doc: bool = False,
        tqdm_disable: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, None]]:
        """
        Get embeddings for entire dataset.
        
        Args:
            dataloader: DataLoader containing dataset
            is_doc: Whether input is documents
            tqdm_disable: Whether to disable progress bar
            
        Returns:
            Tensor of embeddings or tuple of (embeddings, None)
        """
        cnst_embeddings = torch.zeros(
            (len(dataloader.dataset), self.contrastive_dims),
            dtype=torch.float32, device=self.device
        )

        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=tqdm_disable):
                x = {k: v.to(self.device) for k, v in batch['docs'].items()}
                cnst_emb = self.encode(x)
                cnst_embeddings[batch["indices"]] = cnst_emb.detach()
        self.train()

        if self.num_proc > 1:
            dist.all_reduce(cnst_embeddings, op=dist.ReduceOp.SUM)

        if is_doc:
            return cnst_embeddings, None
        return cnst_embeddings

    def forward(
        self,
        docs: Dict[str, torch.Tensor],
        lbls: Optional[Dict[str, torch.Tensor]] = None,
        target: Optional[Dict[str, torch.Tensor]] = None,
        doc_batch_size: Optional[int] = None,
        lbl_batch_size: Optional[int] = None,
        step: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        Forward pass through the model.
        
        Args:
            docs: Document inputs
            lbls: Label inputs
            target: Target labels
            doc_batch_size: Document batch size
            lbl_batch_size: Label batch size
            step: Current step
            
        Returns:
            Tuple of (loss, predictions, label loss, None, None)
        """
        target = target['target_cnst']
        pos_lbl_idx = target.sum(0).nonzero().squeeze()

        doc_embs = self.encode(docs)
        lbl_embs = self.encode(lbls)

        if self.num_proc > 1:            
            global_doc_embs = torch.zeros(
                target.shape[0], doc_embs.shape[1]
            ).to(self.device)
            global_doc_embs[
                doc_batch_size*self.proc_id : doc_batch_size*(self.proc_id + 1)
            ] = doc_embs
            dist.all_reduce(global_doc_embs, op=dist.ReduceOp.SUM)
            doc_embs = global_doc_embs

            global_lbl_embs = torch.zeros(
                target.shape[1], lbl_embs.shape[1]
            ).to(self.device)
            global_lbl_embs[
                lbl_batch_size*self.proc_id : lbl_batch_size*(self.proc_id + 1)
            ] = lbl_embs
            dist.all_reduce(global_lbl_embs, op=dist.ReduceOp.SUM)
            lbl_embs = global_lbl_embs

        cos_sim = (doc_embs @ lbl_embs.T)*self.temp
        
        loss_doc = self.compute_loss(cos_sim, target, self.loss_crit)
        
        if self.add_dual_loss:
            loss_lbl = self.compute_loss(
                cos_sim.T[pos_lbl_idx],
                target.T[pos_lbl_idx],
                self.loss_crit
            )
        else:
            loss_lbl = torch.tensor(0.).to(self.device)
        
        return loss_doc, cos_sim.sigmoid(), loss_lbl, None, None