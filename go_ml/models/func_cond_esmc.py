import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable, List, Dict, Any
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    SequenceClassifierOutput,
    TokenClassifierOutput
)

class FuncCondESMC(nn.Module):
    def __init__(self, embeddings, encoder, lm_head, num_labels):
        super(FuncCondESMC, self).__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.lm_head = lm_head
        self.func_emb = nn.Parameter(torch.zeros(num_labels, 128))
        emb_dim = embeddings.weight.shape[1]
        self.func_lin = nn.Linear(128, emb_dim, bias=False)  # Linear layer for function prediction
        
    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        function_labels: torch.Tensor,
    ):
        batch_size, seq_length = input_ids.shape
        token_embedding_output = self.embeddings(input_ids)

        func_embedding = (self.func_emb.unsqueeze(0) * function_labels.unsqueeze(-1)).sum(dim=1) # (batch_size, 128)
        func_embedding = self.func_lin(func_embedding).unsqueeze(1)  # (batch_size, 1, emb_dim)
        token_embedding_output = token_embedding_output + func_embedding

        encoder_outputs = self.encoder(
            token_embedding_output, attention_mask, False, False
        )
        sequence_output = encoder_outputs.last_hidden_state

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output
        )
    
    def forward(self, input_ids, attention_mask, function_labels):
        outputs = self.embed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            function_labels=function_labels
        )
        sequence_output = outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits
    
from typing import Tuple
import torch
import torch.nn as nn
from torch import optim
from sklearn.metrics import f1_score
import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModel, AutoConfig, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    MultiStepLR,
)

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from go_ml.train_utils import CosineAnnealingWarmupRestarts
from transformers import DataCollatorWithPadding
from transformers import AutoModelForMaskedLM

import pandas as pd
import os
import re
from collections import OrderedDict
import logging as log
import numpy as np

class FuncCondESMCFinetune(pl.LightningModule):
    def __init__(self, model_args) -> None:
        # super(FuncCondESMFinetune, self).__init__()
        super().__init__()
        self.save_hyperparameters()
        self.h = model_args
        self.base_model_name = model_args.model_name
        self.base_model = AutoModelForMaskedLM.from_pretrained(self.base_model_name, torch_dtype='auto', trust_remote_code=True).train()
        #check if base_model has a tokenizer
        if(hasattr(self.base_model, 'tokenizer')):
            self.tokenizer = self.base_model.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name, do_lower_case=False)
        self.vocab_size = self.tokenizer.vocab_size
        label_counts = model_args.label_counts
        self.active_labels = label_counts >= 50
        self.num_labels = int(self.active_labels.sum())
        #convert active_labels from numpy to torch
        if isinstance(self.active_labels, np.ndarray):
            self.active_labels = torch.from_numpy(self.active_labels)
        self.model = FuncCondESMC(self.base_model.embed, self.base_model.transformer, self.base_model.sequence_head, 
                                 num_labels=self.num_labels)
        # self.model.half()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        if model_args.freeze_func_encoder:
            self.model.func_emb.requires_grad = False
            self.model.func_lin.requires_grad = False
    
    def forward(self, input_ids, attention_mask, function_labels):
        logits = self.model(input_ids, attention_mask, function_labels)
        return logits[:, :, :self.vocab_size]

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        (masked_seq_ind, masked_seq_labels, 
         mask, labels, seq_ind) = (batch['masked_seq_tensor'], batch['masked_seq_labels'], batch['seq_mask'], 
                             batch['labels'], batch['seq_tensor'])
        labels = labels[:, self.active_labels] # only use active labels for func embedding
        logits = self.forward(masked_seq_ind, mask, labels)
        loss_val = self.loss_fct(logits.view(-1, self.vocab_size), masked_seq_labels.view(-1))
        if(loss_val.isnan().any() or loss_val.isinf().any()):
            print(f"Loss is NaN or Inf: {loss_val}")
            print(f"Inputs: {masked_seq_ind}")
            print(f"batch nb: {batch_nb}")
            raise ValueError("Loss is NaN or Inf")
        lr = self.scheduler.get_last_lr()[0]
        self.log("lr", lr, prog_bar=True, on_step=True)
        self.log("loss/train", loss_val, prog_bar=True, on_step=True)

        tqdm_dict = {"train_loss": loss_val, "lr": lr}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output
    
    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        """ 
        Runs one training step. This usually consists in the forward function followed
            by the loss function.
        
        :param batch: The output of your dataloader. 
        :param batch_nb: Integer displaying which batch this is
        Returns:
            - dictionary containing the loss and the metrics to be added to the lightning logger.
        """
        (masked_seq_ind, masked_seq_labels, 
         mask, labels, seq_ind) = (batch['masked_seq_tensor'], batch['masked_seq_labels'], batch['seq_mask'], 
                             batch['labels'], batch['seq_tensor'])
        labels = labels[:, self.active_labels] # only use active labels for func embedding
        logits = self.forward(masked_seq_ind, mask, labels)
        loss_val = self.loss_fct(logits.view(-1, self.vocab_size), masked_seq_labels.view(-1))
        if(loss_val.isnan().any() or loss_val.isinf().any()):
            print(f"Loss is NaN or Inf: {loss_val}")
            print(f"Inputs: {masked_seq_ind}")
            print(f"batch nb: {batch_nb}")
            raise ValueError("Loss is NaN or Inf")
        self.log("loss/val", loss_val, prog_bar=True, on_step=True)
        # tqdm_dict = {"val_loss": loss_val}
        # output = OrderedDict(
        #     {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        # return output
    
    def configure_optimizers(self):
        """ Sets different Learning rates for different parameter groups. """
        # parameters = [param for param in self.parameters() if param.requires_grad]
        optimizer = optim.AdamW(self.parameters(), lr=self.h.learning_rate, weight_decay=self.h.weight_decay)
        
        # # Use cosine annealing with warmup
        # warmup_steps = int(self.h.num_train_steps * 0.1)
        # lr_scheduler = CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=self.h.num_train_steps - warmup_steps,
        #     T_mult=1,
        #     eta_min=0.0,
        #     last_epoch=-1,
        # )

        scheduler_config = dict(
            first_cycle_steps=int(self.h.num_train_steps * 0.15), 
            cycle_mult=1.0,
            max_lr_mul=1e1,
            warmup_steps=2000,
            gamma=0.7,
        )
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer, **scheduler_config)
        
        self.scheduler = lr_scheduler
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": False,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    @classmethod
    def add_model_specific_args(cls, parser):
        parser.add_argument(
            "--model_name",
            # default='Synthyra/ESM2-650M',
            default='Synthyra/ESMplusplus_small',
            type=str,
            help="Model name",
        )
        parser.add_argument(
            "--max_length",
            default=1024,
            type=int,
            help="Maximum sequence length.",
        )
        parser.add_argument(
            "--learning_rate",
            default=1.5e-5,
            type=float,
            help="Learning rate",
        )
        parser.add_argument("--gradient_checkpointing", default=False, type=bool, 
            help="Enable or disable gradient checkpointing which use the cpu memory \
                with the gpu memory to store the model.",
        )
        parser.add_argument("--freeze_func_encoder", default=False, type=bool, 
            help="Whether to freeze the function-conditioned encoder"
        )
        parser.add_argument(
            "--gradient_clipping", default=1.0, type=float, help="Global norm gradient clipping"
        )
        parser.add_argument(
            "--weight_decay", default=0.01, type=float, help="Weight decay per train step."
        )
        return parser
    



    



    
