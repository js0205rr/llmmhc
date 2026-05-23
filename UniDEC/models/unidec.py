"""
Universal Document Embedding and Classification (UniDEC) model.

This module implements a universal document embedding and classification model
that combines contrastive learning with classification for efficient document
processing.

Example:
    >>> from models import UniDEC
    >>> model = UniDEC(params)
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


class UniDEC(ContrastiveLossMixin, nn.Module):
    """
    Universal Document Embedding and Classification model.
    
    This class implements a universal document embedding and classification model
    that combines contrastive learning with classification for efficient document
    processing.
    
    Args:
        params: Model parameters including dataset, device, etc.
        
    Example:
        >>> model = UniDEC(params)
        >>> loss, predictions = model(documents, labels, targets)
    """
    
    def __init__(self, params: Any):
        """
        Initialize UniDEC model.
        
        Args:
            params: Model parameters
        """
        super(UniDEC, self).__init__()
        
        # Initialize model parameters
        self.embed_dim = 768
        self.dataset = params.dataset
        self.num_proc = params.num_proc
        self.device = params.device
        self.num_labels = params.num_labels
        self.shortlist_size = params.shortlist_size
        self.anns = FaissMIPSIndex(torch.cuda.current_device())
        
        # Initialize loss functions
        self.DE_loss = params.DE_loss
        self.CLF_loss = params.DE_loss if params.CLF_loss is None else params.CLF_loss
        self.add_bce_loss = params.add_bce_loss
        self.add_dual_loss = params.add_dual_loss
        self.add_dual_clf_loss = params.add_dual_clf_loss 

        # Initialize model components
        self.hidden_dims = params.hidden_dims
        self.contrastive_dims = params.contrastive_dims
        self.temp = nn.Parameter(torch.tensor(params.temp))
        self.init_classifiers()
        self.bce_loss = nn.BCEWithLogitsLoss()

        print(f"Using temp = {self.temp.item()} for this model training.")

        # Initialize sentence transformer
        self.bert = SentenceTransformer("msmarco-distilbert-base-v4")
        
        # Configure optimizer parameters
        no_decay = ['LayerNorm.weight']
        wd = 0.05
        clf_lr = 1e-3
        
        self.dense_grouped_params = [
            {
                'params': [p for n, p in [*self.bert.named_parameters()]
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': wd,
                'lr': params.lr
            },
            {
                'params': [p for n, p in [*self.bert.named_parameters()]
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': params.lr
            },
            {
                'params': [*self.cnst_head[0].parameters()] + [*self.clf_head[0].parameters()],
                'weight_decay': wd,
                'lr': params.lr*2
            },
            {
                'params': [*self.ext_classif_embed.parameters()],
                'weight_decay': wd,
                'lr': clf_lr
            }
        ]
    
    def init_classifiers(self) -> None:
        """Initialize model classifiers and embeddings."""
        # Initialize contrastive head
        self.cnst_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.contrastive_dims),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        nn.init.xavier_uniform_(self.cnst_head[0].weight)
        nn.init.zeros_(self.cnst_head[0].bias)

        # Initialize dataset-specific dropout
        ext_drop = {
            "LF-WikiSeeAlsoTitles-320K": 0.1, 
            "LF-WikiSeeAlso-320K": 0.2, 
            "LF-WikiTitles-500K": 0.1, 
            "LF-Amazon-131K": 0.1, 
            "LF-AmazonTitles-131K": 0.1, 
            "LF-AmazonTitles-1.3M": 0.1
        }
        self.ext_drop = nn.Dropout(ext_drop[self.dataset])

        # Initialize classification head
        self.clf_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dims),
            nn.Dropout(0.1)
        )
        nn.init.xavier_uniform_(self.clf_head[0].weight)
        nn.init.zeros_(self.clf_head[0].bias)
        
        # Initialize classification embeddings
        self.ext_classif_embed = nn.Embedding(
            self.num_labels+1,
            self.hidden_dims,
            padding_idx=-1
        )
        self.ext_classif_embed.weight[-1].data.fill_(0)
        nn.init.xavier_uniform_(self.ext_classif_embed.weight[:-1])

    def encode(
        self,
        x: Dict[str, torch.Tensor],
        is_doc: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input into embeddings.
        
        Args:
            x: Input dictionary containing input_ids and attention_mask
            is_doc: Whether input is documents
            
        Returns:
            Encoded embeddings or tuple of (contrastive embeddings, classification embeddings)
        """
        bert_out = self.bert({
            'input_ids': x['input_ids'],
            'attention_mask': x['attention_mask']
        })
        mean_emb = bert_out['sentence_embedding']

        cnst_emb = F.normalize(self.cnst_head(mean_emb))

        if is_doc:
            cls_emb = bert_out['sentence_embedding'].clone()
            clf_emb = self.clf_head(cls_emb)
            return cnst_emb, clf_emb
        
        return cnst_emb

    def get_dataset_embeddings(
        self,
        dataloader: Any,
        is_doc: bool = False,
        tqdm_disable: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get embeddings for entire dataset.
        
        Args:
            dataloader: DataLoader containing dataset
            is_doc: Whether input is documents
            tqdm_disable: Whether to disable progress bar
            
        Returns:
            Tensor of embeddings or tuple of (contrastive embeddings, classification embeddings)
        """
        cnst_embeddings = torch.zeros(
            (len(dataloader.dataset), self.contrastive_dims),
            dtype=torch.float32, device=self.device
        )

        if is_doc:
            clf_embeddings = torch.zeros(
                (len(dataloader.dataset), self.hidden_dims),
                dtype=torch.float32, device=self.device
            )

        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, disable=tqdm_disable):
                x = {k: v.to(self.device) for k, v in batch['docs'].items()}
                if is_doc:
                    cnst_emb, clf_emb = self.encode(x, True)
                    clf_embeddings[batch["indices"]] = F.normalize(clf_emb.detach())
                else:
                    cnst_emb = self.encode(x)                
                cnst_embeddings[batch["indices"]] = cnst_emb.detach()
        self.train()

        if self.num_proc > 1:
            dist.all_reduce(cnst_embeddings, op=dist.ReduceOp.SUM)
            
        if is_doc:
            if self.num_proc > 1:
                dist.all_reduce(clf_embeddings, op=dist.ReduceOp.SUM)
            return cnst_embeddings, clf_embeddings
        
        return cnst_embeddings

    def forward(
        self,
        docs: Dict[str, torch.Tensor],
        lbls: Optional[Dict[str, torch.Tensor]] = None,
        target: Optional[Dict[str, torch.Tensor]] = None,
        step: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            docs: Document inputs
            lbls: Label inputs
            target: Target labels
            step: Current step
            
        Returns:
            Tuple of (contrastive loss, contrastive predictions, classification loss,
                    classification predictions, candidates)
        """
        target_cnst = target['target_cnst']
        pos_cnst_idx = target_cnst.sum(0).nonzero().squeeze()

        target_clf, pos_cand_idx = target_cnst, pos_cnst_idx
        candidates = target['cnst_labels'].to(self.device)

        doc_embs, doc_clf_embs = self.encode(docs, is_doc=True)
        lbl_embs = self.encode(lbls)

        # Compute contrastive similarity and loss
        cos_sim = (doc_embs @ lbl_embs.T)*self.temp
        loss_cnst = self.compute_loss(cos_sim, target_cnst, self.DE_loss)
        
        if self.add_dual_loss:
            loss_lbl = self.compute_loss(
                cos_sim.T[pos_cnst_idx],
                target_cnst.T[pos_cnst_idx],
                self.DE_loss
            )
            loss_cnst = (loss_cnst + loss_lbl)/2

        # Compute classification similarity and loss
        lbl_clf_embs = self.ext_drop(self.ext_classif_embed(candidates))
        dot_sim = (doc_clf_embs @ lbl_clf_embs.T)*self.temp

        loss_clf = self.compute_loss(dot_sim, target_clf, self.CLF_loss)
        if self.add_dual_clf_loss:
            loss_dot_lbl = self.compute_loss(
                dot_sim.T[pos_cand_idx],
                target_clf.T[pos_cand_idx],
                self.CLF_loss
            )
            loss_clf = (loss_clf + loss_dot_lbl)/2 

        if self.add_bce_loss:
            loss_clf = loss_clf + self.bce_loss(dot_sim, target_clf)
        
        return loss_cnst, cos_sim.sigmoid(), loss_clf, dot_sim.sigmoid(), candidates
