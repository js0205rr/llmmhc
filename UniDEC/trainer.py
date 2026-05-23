"""
Trainer module for training and evaluating document embedding models.

This module provides functionality for training and evaluating various document embedding
models including UniDEC, SupConDR, and LLM. It supports both single and multi-GPU training
using the accelerate library.

Example:
    >>> from trainer import Runner
    >>> runner = Runner(dataloaders, accelerator, inv_prop, query_clustering, params)
    >>> runner.train(model, params)
"""

import os
import gc
import math
import time
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import broadcast, send_to_device
from transformers import AdamW, get_linear_schedule_with_warmup
from dense_clustering import cluster_dense_embs
from accelerate.logging import get_logger
import bitsandbytes as bnb
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass


@dataclass
class ClusteringConfig:
    """Configuration for query clustering data."""
    train_order: Optional[np.ndarray]
    train_batch_sizes: Optional[np.ndarray]
    cluster_matrix: Optional[sp.spmatrix]


class Runner:
    """
    Runner class for training and evaluating document embedding models.
    
    This class handles the training and evaluation loop for document embedding models,
    supporting both single and multi-GPU training using the accelerate library.
    
    Args:
        dataloaders: List of dataloaders for train, test, label and document data
        accelerator: Accelerator instance for distributed training
        inv_prop: Inverse propensity scores
        query_clustering: Query clustering information
        params: Training parameters
        top_k: Number of top predictions to consider
        
    Example:
        >>> runner = Runner(dataloaders, accelerator, inv_prop, query_clustering, params)
        >>> runner.train(model, params)
    """
    
    def __init__(
        self,
        dataloaders: List[Any],
        accelerator: Accelerator,
        inv_prop: np.ndarray,
        clustering_config: ClusteringConfig,
        params: Any,
        top_k: int = 5
    ):
        """Initialize Runner with dataloaders and parameters."""
        # Initialize dataloaders
        self.train_dl = dataloaders[0]
        self.test_dl = dataloaders[1]
        self.label_dl = dataloaders[2]
        self.doc_dl = dataloaders[3]
        
        # Initialize accelerator and device info
        self.accelerator = accelerator
        self.DEVICE = accelerator.device
        self.num_proc = params.num_proc
        self.proc_id = int(str(accelerator.device)[-1])
        
        # Initialize dataset info
        self.num_train = len(self.train_dl.dataset)
        self.num_test = len(self.test_dl.dataset)
        self.dataset = params.dataset
        self.top_k = top_k
        
        # Initialize model parameters
        self.inv_prop = torch.from_numpy(inv_prop.astype(np.float64))
        self.model_dir = params.model_dir
        self.shortlist_size = params.shortlist_size
        self.train_order = clustering_config.train_order
        self.train_bs = clustering_config.train_batch_sizes
        self.cluster_mat = clustering_config.cluster_matrix
        
        # Initialize training parameters
        self.loss_lambda = -1 if (params.encoder == "SupConDR" and not params.add_dual_loss) else params.loss_lambda
        self.hard_min_negs = params.num_negs > 0
        self.latest_anns_ep = 0
        self.update_pos = params.update_pos
        self.label_pool_size = params.label_pool_size
        
        # Initialize logging
        self.logger = get_logger(name=f"{params.version}.log", log_level="INFO")
        
        # Calculate tree depth for clustering
        self.tree_depth = int(math.log(self.num_train/params.batch_size, 2))
        
        # Initialize test filtering if available
        if hasattr(self.test_dl.dataset, 'filter_test'):
            self.filter_test = self.test_dl.dataset.filter_test

    def create_hard_negatives(
        self,
        model: nn.Module,
        epoch: int,
        doc_embs: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Create hard negative examples for training.
        
        Args:
            model: Model to use for creating embeddings
            epoch: Current epoch number
            doc_embs: Optional pre-computed document embeddings
            
        Returns:
            Document embeddings if computed
        """
        self.accelerator.print("Updating Hard Negatives")
        torch.cuda.empty_cache()
        
        # Update label embeddings if needed
        if self.latest_anns_ep != epoch:
            lbl_enc_embs = model.get_dataset_embeddings(
                self.label_dl,
                tqdm_disable=not self.accelerator.is_main_process
            )
            if self.accelerator.is_main_process:
                model.anns.build_index(lbl_enc_embs)
            
            del lbl_enc_embs
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
        
        # Get document embeddings if not provided
        if doc_embs is None:
            doc_embs = model.get_dataset_embeddings(
                self.doc_dl,
                tqdm_disable=not self.accelerator.is_main_process
            )
        
        # Create hard negatives on main process
        if self.accelerator.is_main_process:
            all_preds = model.anns.search(doc_embs.float(), k=self.shortlist_size)
            hard_negs = all_preds[0].cpu().numpy()
            np.save(self.model_dir + '/hard_negatives.npy', hard_negs)
            del model.anns.anns, all_preds, hard_negs
        
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        self.train_dl.dataset.reload_negatives()
        
        return doc_embs

    def update_clustered_batches(
        self,
        model: nn.Module,
        epoch: int,
        cl_start: int,
        cl_update: int,
        embs: Optional[torch.Tensor] = None,
        force_rebatch: bool = False
    ) -> None:
        """
        Update clustered batches for training.
        
        Args:
            model: Model to use for creating embeddings
            epoch: Current epoch number
            cl_start: Epoch to start clustering
            cl_update: Frequency of clustering updates
            embs: Optional pre-computed embeddings
            force_rebatch: Whether to force rebatching
        """
        if epoch >= cl_start:
            if (epoch - cl_start) % cl_update == 0 or force_rebatch:
                if embs is None:
                    self.accelerator.print(f'Started creating updated query text embeddings at {time.ctime()}')
                    torch.cuda.empty_cache()
                    embs = model.get_dataset_embeddings(
                        self.doc_dl,
                        tqdm_disable=not self.accelerator.is_main_process
                    )
                    self.accelerator.print(f'Query embeddings created at {time.ctime()}')
                
                if self.accelerator.is_main_process:
                    self.cluster_mat = cluster_dense_embs(
                        embs,
                        embs.device,
                        tree_depth=self.tree_depth
                    ).tocsr()
                    sp.save_npz(f'{self.model_dir}/cluster_mat.npz', self.cluster_mat)
                
                embs = embs.detach().cpu().numpy()
                del embs

            self.accelerator.wait_for_everyone()

            if self.accelerator.is_main_process:
                print('Updating clustered train order...\n')
                cmat = self.cluster_mat[np.random.permutation(self.cluster_mat.shape[0])]
                self.train_order[:] = cmat.indices
                self.train_bs[:] = np.array([b.nnz for b in cmat])
        
        elif self.accelerator.is_main_process:
            print('Shuffling train order...\n')
            self.train_order[:] = np.random.permutation(len(self.train_dl.dataset))

        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()

    def process_batch_for_multi_GPU_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process batch for multi-GPU training.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Processed batch dictionary
        """
        def split_batch(input_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], int]:
            # Use ceiling division for more even batch distribution
            samples_per_gpu = math.ceil(input_dict['input_ids'].shape[0] / self.num_proc)
            start_idx = samples_per_gpu * self.proc_id
            end_idx = samples_per_gpu * (self.proc_id + 1)
            
            input_dict['input_ids'] = input_dict['input_ids'][start_idx:end_idx]
            input_dict['attention_mask'] = input_dict['attention_mask'][start_idx:end_idx]
            return input_dict, samples_per_gpu

        # Process labels
        num_labels = torch.ones(1).to(self.DEVICE) * batch['lbls']['input_ids'].shape[0]
        num_labels = broadcast(num_labels)
        num_labels = int(num_labels.item())

        # Process documents
        batch['docs'] = {k: v.to(self.DEVICE) for k, v in batch['docs'].items()}
        batch['docs'], batch['doc_batch_size'] = split_batch(batch['docs'])
        
        # Process labels
        batch['lbls'] = {k: v[:num_labels].to(self.DEVICE) for k, v in batch['lbls'].items()}
        batch['lbls'] = self.accelerator.pad_across_processes(batch['lbls'])
        batch['lbls'] = broadcast(batch['lbls'])
        batch['lbls'], batch['lbl_batch_size'] = split_batch(batch['lbls'])

        # Process targets
        batch['target']['target_cnst'] = send_to_device(batch['target']['target_cnst'], self.DEVICE)
        batch['target']['target_cnst'] = self.accelerator.pad_across_processes(batch['target']['target_cnst'], dim=1)
        batch['target']['target_cnst'] = broadcast(batch['target']['target_cnst'])
        batch['target']['target_cnst'] = batch['target']['target_cnst'][:, :num_labels]

        return batch

    def fit_one_epoch(self, model: nn.Module, epoch: int) -> None:
        """
        Train model for one epoch.
        
        Args:
            model: Model to train
            epoch: Current epoch number
        """
        torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        
        # Initialize metrics
        train_loss_clf, train_loss_cnst = 0.0, 0.0
        self.cosine_scores = torch.zeros(self.top_k, dtype=torch.int32)
        self.dot_product_scores = torch.zeros(self.top_k, dtype=torch.int32)
        labels_per_doc, docs_per_batch = 0., 0.
        contrastive_loss, classification_loss = torch.tensor([0.]), torch.tensor([0.])

        model.train()
        progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)
        
        for step, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()

            # Process batch for multi-GPU training
            if self.num_proc > 1:
                batch = self.process_batch_for_multi_GPU_train(batch)

            # Update progress bar
            if self.accelerator.is_main_process:
                target = batch['target']
                labels_per_doc += torch.mean(torch.sum(target['target_cnst'], axis=1))
                docs_per_batch += target['target_cnst'].shape[1]
                progress_bar.set_postfix({
                    'Docs/Batch': target['target_cnst'].shape[1],
                    'Contrastive Loss': contrastive_loss.item(),
                    'Classification Loss': classification_loss.item()
                })

            # Forward pass
            contrastive_loss, cosine_probs, classification_loss, dot_probs, candidates = model(**batch, step=step)
            
            # Apply loss weights
            if self.loss_lambda != -1:
                contrastive_loss = contrastive_loss * self.loss_lambda
                classification_loss = classification_loss * (1 - self.loss_lambda)
            
            # Backward pass
            total_loss = contrastive_loss + classification_loss            
            self.accelerator.backward(total_loss)
            self.optimizer.step()
            self.scheduler.step()

            # Update metrics
            if self.accelerator.is_main_process:
                train_loss_cnst += contrastive_loss.detach()
                train_loss_clf += classification_loss.detach()

                # Update contrastive scores
                if cosine_probs is not None and cosine_probs.shape[1] >= self.top_k:
                    top_k_cosine_preds = torch.topk(cosine_probs, self.top_k)[1].to('cpu')
                    top_k_cosine_preds = target['cnst_labels'][top_k_cosine_preds].detach().cpu()
                    self.predict(top_k_cosine_preds, target['lbls'], self.cosine_scores)

                # Update classification scores
                if dot_probs is not None:
                    top_k_dot_preds = torch.topk(dot_probs, self.top_k)[1]
                    top_k_dot_preds = candidates[top_k_dot_preds].detach().cpu()
                    self.predict(top_k_dot_preds, target['lbls'], self.dot_product_scores)

            self.accelerator.wait_for_everyone()

        # Log epoch results
        if self.accelerator.is_main_process:
            train_loss_clf /= self.steps_per_epoch
            train_loss_cnst /= self.steps_per_epoch

            # Log losses
            print(f"Epoch: {epoch}, LR: {[round(x, 6) for x in self.scheduler.get_last_lr()]}, "
                  f"Contrastive Loss: {train_loss_cnst:.4f}, Classification Loss: {train_loss_clf:.4f}")
            print(f'Labels/Doc: {(labels_per_doc/self.steps_per_epoch):.2f}, '
                  f'Labels/Batch: {(docs_per_batch//self.steps_per_epoch)}')
            
            self.logger.info(
                f"Epoch: {epoch}, Contrastive Loss: {train_loss_cnst:.4f}, "
                f"Avg. Labels/Doc: {(labels_per_doc/self.steps_per_epoch):.2f}, "
                f"Avg. Labels/batch: {(docs_per_batch/self.steps_per_epoch):.2f}",
                main_process_only=True
            )

            # Log ANNS scores
            precision = self.cosine_scores.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1))
            print(f'ANNS Training Scores: P@1: {precision[0]:.2f}, P@3: {precision[2]:.2f}, P@5: {precision[4]:.2f}')
            self.logger.info(
                f'ANNS Training Scores: P@1: {precision[0]:.2f}, P@3: {precision[2]:.2f}, P@5: {precision[4]:.2f}\n',
                main_process_only=True
            )

            # Log classification scores
            if hasattr(model, 'ext_classif_embed'):
                precision = self.dot_product_scores.detach().cpu().numpy() * 100.0 / (self.num_train * np.arange(1, self.top_k+1))
                print(f'CLFS Training Scores: P@1: {precision[0]:.2f}, P@3: {precision[2]:.2f}, P@5: {precision[4]:.2f}\n')
                self.logger.info(
                    f'CLFS Training Scores: P@1: {precision[0]:.2f}, P@3: {precision[2]:.2f}, P@5: {precision[4]:.2f}\n',
                    main_process_only=True
                )
        
        self.accelerator.wait_for_everyone()

    def initialize_model(self, model: nn.Module, params: Any) -> int:
        """
        Initialize model from checkpoint.
        
        Args:
            model: Model to initialize
            params: Model parameters
            
        Returns:
            Initial epoch number
        """
        model_path = os.path.join(self.model_dir, params.load_model)
        self.accelerator.print(f'loading model from {model_path}')
        self.logger.info(f'loading model from {model_path}', main_process_only=True)

        self.accelerator.load_state(model_path)
        init = math.ceil(self.scheduler.state_dict()['last_epoch']/self.steps_per_epoch)

        if params.test:
            self.evaluate(model.module, params, init)

        if self.update_pos != -1 and init >= params.cl_start:
            self.accelerator.print(f"\nUpdating number of sampled positive to {self.update_pos}.\n")
            self.train_dl.dataset.num_pos = self.update_pos

        self.accelerator.wait_for_everyone()

        doc_embs = None
        if params.re_hnm:
            doc_embs = self.create_hard_negatives(model.module, init)

        self.update_clustered_batches(
            model.module,
            init,
            params.cl_start,
            params.cl_update,
            doc_embs,
            params.rebatch
        )
        del doc_embs

        return init

    def train(self, model: nn.Module, params: Any) -> None:
        """
        Train model.
        
        Args:
            model: Model to train
            params: Training parameters
        """
        # Log model and parameters
        pattern = "%"*100 + '\n'
        self.accelerator.print(model)
        self.logger.info(model, main_process_only=True)
        self.accelerator.print(pattern + str(params) + '\n' + pattern)
        self.logger.info(pattern + str(params) + '\n' + pattern, main_process_only=True)
        
        # Initialize optimizer
        if params.pre_trained_model == 'phi3':
            self.optimizer = bnb.optim.AdamW8bit(
                model.dense_grouped_params,
                lr=params.lr,
                optim_bits=8
            )
        else:
            self.optimizer = AdamW(model.dense_grouped_params, lr=params.lr)
        
        # Initialize scheduler
        self.steps_per_epoch = len(self.train_dl)
        init, last_batch = 0, -1
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            last_epoch=last_batch,
            num_training_steps=params.num_epochs*self.steps_per_epoch,
            num_warmup_steps=3*self.steps_per_epoch
        )

        # Prepare model and dataloaders
        model, self.optimizer, self.scheduler = self.accelerator.prepare(
            model,
            self.optimizer,
            self.scheduler
        )
        self.train_dl, self.test_dl, self.doc_dl, self.label_dl = self.accelerator.prepare(
            self.train_dl,
            self.test_dl,
            self.doc_dl,
            self.label_dl
        )

        # Load model if specified
        if len(params.load_model):
            init = self.initialize_model(model, params)
        
        cl_start, cl_update = params.cl_start, params.cl_update

        # Handle score fusion
        if params.score_fusion:
            self.evaluate(model.module, params)
            return

        # Training loop
        doc_embs = None
        for epoch in range(init + 1, params.num_epochs + 1):    
            gc.collect()
            torch.cuda.empty_cache()

            # Train epoch
            self.fit_one_epoch(model.module, epoch)

            # Save checkpoint
            self.accelerator.save_state(f'{params.model_dir}/model_latest.pth')

            # Handle clustering start
            if epoch == cl_start:
                self.accelerator.save_state(f'{params.model_dir}/model_{cl_start}.pth')
                if self.update_pos != -1:
                    self.accelerator.print(f"\nUpdating number of sampled positive to {self.update_pos}.\n")
                    self.train_dl.dataset.num_pos = self.update_pos
            
            # Evaluate and update batches
            if (epoch - cl_start) % params.eval_step == 0:
                self.evaluate(model.module, params, epoch)

                if self.hard_min_negs and epoch >= cl_start and (epoch - cl_start) % cl_update == 0:
                    doc_embs = self.create_hard_negatives(model.module, epoch)
                else:
                    doc_embs = None       
                    if self.accelerator.is_main_process:          
                        del model.module.anns.anns
                
                self.update_clustered_batches(
                    model.module,
                    epoch,
                    cl_start,
                    cl_update,
                    doc_embs
                )                
                del doc_embs            
            else:
                self.update_clustered_batches(
                    model.module,
                    epoch,
                    cl_start,
                    cl_update
                )

    def evaluate(self, model: nn.Module, params: Any, epoch: Optional[int] = None) -> None:
        """
        Evaluate model.
        
        Args:
            model: Model to evaluate
            params: Evaluation parameters
            epoch: Current epoch number
        """
        torch.cuda.empty_cache()

        # Get test embeddings
        self.accelerator.print("Creating Test Document Embeddings")
        test_enc_embs, test_clf_embs = model.get_dataset_embeddings(
            self.test_dl,
            is_doc=True,
            tqdm_disable=not self.accelerator.is_main_process
        )
        
        # Get label embeddings
        lbl_enc_embs = model.get_dataset_embeddings(
            self.label_dl,
            tqdm_disable=not self.accelerator.is_main_process
        )

        if self.accelerator.is_main_process:
            # Handle classification embeddings
            if test_clf_embs is not None:
                if params.CLF_loss == 'bce':
                    lbl_clf_embs = model.ext_classif_embed.weight[:-1].data.detach()
                else:
                    lbl_clf_embs = F.normalize(model.ext_classif_embed.weight[:-1].data.detach())
            
                # Evaluate combined embeddings
                lbl_embs = torch.hstack((lbl_enc_embs, lbl_clf_embs))
                test_doc_embs = torch.hstack((test_enc_embs, test_clf_embs))
                model.anns.build_index(lbl_embs)
                self.test(model, test_doc_embs, 'COMB')
            
                torch.cuda.empty_cache()

                # Evaluate classification embeddings
                model.anns.build_index(lbl_clf_embs)
                self.test(model, test_clf_embs, 'CLFS')

                del lbl_clf_embs
                torch.cuda.empty_cache()

            # Evaluate contrastive embeddings
            model.anns.build_index(lbl_enc_embs)
            self.test(model, test_enc_embs, 'ANNS')
        
        del test_enc_embs, test_clf_embs, lbl_enc_embs
        torch.cuda.empty_cache()
        self.latest_anns_ep = epoch
        self.accelerator.wait_for_everyone()
        
    def predict(
        self,
        predictions: torch.Tensor,
        true_labels: torch.Tensor,
        extreme_count: torch.Tensor,
        numerator: Optional[torch.Tensor] = None,
        denominator: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update prediction metrics.
        
        Args:
            predictions: Model predictions
            true_labels: Ground truth labels
            extreme_count: Tensor to store extreme count metrics
            numerator: Optional numerator tensor for PSP
            denominator: Optional denominator tensor for PSP
        """
        for pred, true_label in zip(predictions, true_labels):
            matches = torch.isin(pred, true_label.cpu())
            extreme_count += torch.cumsum(matches, dim=0)
            
            if numerator is not None:
                matches = matches.double() 
                matches[matches > 0] = self.inv_prop[pred[matches > 0]]

                numerator += torch.cumsum(matches, dim=0)
                sorted_inv_prop = torch.sort(self.inv_prop[true_label], descending=True)[0]

                match_tensor = torch.zeros(self.top_k, device=sorted_inv_prop.device)
                match_size = min(true_label.shape[0], self.top_k)
                match_tensor[:match_size] = sorted_inv_prop[:match_size]
                denominator += torch.cumsum(match_tensor, dim=0)

    def test(self, model: nn.Module, doc_cache_embs: torch.Tensor, inference: str = 'ANNS') -> None:
        """
        Test model on given embeddings.
        
        Args:
            model: Model to test
            doc_cache_embs: Document embeddings
            inference: Inference type
        """
        model.eval()
        with torch.no_grad():
            # Initialize metrics
            self.extreme_count = torch.zeros(self.top_k, dtype=torch.int32)
            self.num = torch.zeros(self.top_k)
            self.den = torch.zeros(self.top_k)

            # Get predictions
            candidates, probs = model.anns.search(doc_cache_embs.float(), k=100)   
            candidates, probs = candidates.detach().cpu(), probs.detach().cpu()

            # Apply test filtering if available
            if hasattr(self, 'filter_test'):
                for i in self.filter_test.keys():
                    probs[i, torch.isin(candidates[i], self.filter_test[i])] = 0.
            
            # Get top-k predictions
            preds = torch.topk(probs, self.top_k)[1]
            preds = candidates[np.arange(preds.shape[0]).reshape(-1, 1), preds]

            # Update metrics
            test_labels = self.test_dl.dataset.test_labels
            self.predict(preds, test_labels, self.extreme_count, self.num, self.den)

            # Calculate and log scores
            prec = self.extreme_count * 100.0 / (self.num_test * torch.arange(1, self.top_k+1))
            psp = (self.num * 100 / self.den)

            self.accelerator.print(
                f"{inference} scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}, "
                f"PSP@1: {psp[0]:.2f}, PSP@3: {psp[2]:.2f}, PSP@5: {psp[4]:.2f}\n"
            )
            self.logger.info(
                f"{inference} scores: P@1: {prec[0]:.2f}, P@3: {prec[2]:.2f}, P@5: {prec[4]:.2f}, "
                f"PSP@1: {psp[0]:.2f}, PSP@3: {psp[2]:.2f}, PSP@5: {psp[4]:.2f}\n",
                main_process_only=True
            )
