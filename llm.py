"""
Language Model (LLM) based implementation for document embedding and classification.

This module implements a language model based approach using Phi-3-mini-4k-instruct
with LoRA fine-tuning for efficient document embedding and classification.

Example:
    >>> from models import LLM
    >>> model = LLM(params)
    >>> embeddings = model.encode(documents)
"""

import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from models.anns import FaissMIPSIndex
from models.losses import ContrastiveLossMixin
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, Optional, Tuple, Union, Any


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.

    Args:
        model: PyTorch model to analyze
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )


class LLM(ContrastiveLossMixin, nn.Module):
    """
    Language Model based implementation for document embedding and classification.

    This class implements a language model based approach using Phi-3-mini-4k-instruct
    with LoRA fine-tuning for efficient document embedding and classification.

    Args:
        params: Model parameters including dataset, device, etc.

    Example:
        >>> model = LLM(params)
        >>> loss, predictions = model(documents, labels, targets)
    """

    def __init__(self, params: Any):
        """
        Initialize LLM model.

        Args:
            params: Model parameters
        """
        super(LLM, self).__init__()

        # Initialize model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Phi-3-mini-4k-instruct',
            trust_remote_code=True,
            torch_dtype="auto",
            quantization_config=bnb_config,
            attn_implementation="flash_attention_2"
        )
        print('Loaded with optimizations')

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure LoRA
        config = LoraConfig(
            r=24,
            lora_alpha=36,
            lora_dropout=0.1,
            bias="none",
            target_modules=['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'],
            task_type="SEQ_CLS",
        )
        self.model = get_peft_model(model, config)
        print_trainable_parameters(self.model)

        # Initialize model parameters
        self.embed_dim = self.model.config.hidden_size
        self.dataset = params.dataset
        self.num_proc = params.num_proc
        self.device = params.device
        self.num_labels = params.num_labels
        self.shortlist_size = params.shortlist_size
        self.anns = FaissMIPSIndex(torch.cuda.current_device())
        self.add_dual_loss = params.add_dual_loss
        self.loss_crit = params.DE_loss
        self.contrastive_dims = params.contrastive_dims
        self.temp = nn.Parameter(torch.tensor(params.temp))
        self.proc_id = int(str(params.device)[-1])

        print(f"Using temp = {self.temp.item()} for this model training.")

        # Initialize classifiers and loss functions
        self.init_classifiers()
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Configure optimizer parameters
        lr = params.lr
        self.dense_grouped_params = [
            {'params': [*self.model.parameters()], 'lr': lr},
            {'params': [*self.cnst_head[0].parameters()], 'lr': lr*10},
        ]

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get model state dict with only LoRA parameters.

        Returns:
            Dictionary containing model state
        """
        state = self.model.state_dict()
        for name in list(state.keys()):
            if "lora" not in name:
                state.pop(name)
        return state

    def init_classifiers(self) -> None:
        """Initialize contrastive head for embeddings."""
        self.cnst_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.contrastive_dims).bfloat16(),
            nn.Tanh(),
            nn.Dropout(0.1)
        )
        nn.init.xavier_uniform_(self.cnst_head[0].weight)
        nn.init.zeros_(self.cnst_head[0].bias)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = False) -> None:
        """
        Load model state dict.

        Args:
            state_dict: Model state dictionary
            strict: Whether to strictly enforce state dict keys
        """
        self.model.load_state_dict(state_dict, strict=strict)

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
                cnst_embeddings[batch["indices"]] = cnst_emb.detach().float()
        self.train()

        if self.num_proc > 1:
            dist.all_reduce(cnst_embeddings, op=dist.ReduceOp.SUM)

        if is_doc:
            return cnst_embeddings, None
        return cnst_embeddings

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
        input_ids, attn_mask = input_ids[:, -max_len:], attn_mask[:, -max_len:]
        bert_out = self.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True
        )
        cnst_emb = F.normalize(
            self.cnst_head(bert_out.hidden_states[-1][:, -1])
        )
        return cnst_emb

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
