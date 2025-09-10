import torch
import torch.nn as nn
import torch.nn.functional as F
    
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
from transformers import AutoModelForTokenClassification
from sklearn.metrics import f1_score, precision_recall_fscore_support

import pandas as pd
import os
import re
from collections import OrderedDict
import logging as log
import numpy as np

class ESMCTokenFinetune(pl.LightningModule):
    def __init__(self, model_args) -> None:
        # super(FuncCondESMFinetune, self).__init__()
        super().__init__()
        self.save_hyperparameters()

        self.h = model_args
        self.model_name = model_args.model_name
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, 
                                                                     torch_dtype='auto', trust_remote_code=True).train()
        self.vocab_size = 33
        if(model_args.freeze_encoder):
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.sequence_head.parameters():
                param.requires_grad = True
            print("Froze ESM model parameters")

    def forward(self, input_ids, attention_mask):
        logits = self.model(input_ids, attention_mask)
        return logits

    def training_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        seq_ind, mask, annot_labels = (batch['seq_ind'], batch['mask'], batch['annot_mask'])
        esm_output = self.forward(seq_ind, mask)
        token_logits = esm_output.logits[:, :annot_labels.shape[1], :]
        annot_labels = annot_labels[:, :token_logits.shape[1]].long()
        annot_labels[~mask.bool()] = -100
        loss_val = F.cross_entropy(token_logits.reshape(-1, 2), annot_labels.reshape(-1), ignore_index=-100)
        if(loss_val.isnan().any() or loss_val.isinf().any()):
            print(f"Loss is NaN or Inf: {loss_val}")
            print(f"Inputs: {seq_ind}")
            print(f"batch nb: {batch_nb}")
            raise ValueError("Loss is NaN or Inf")
        lr = self.scheduler.get_last_lr()[0]
        self.log("lr", lr, prog_bar=True, on_step=True)
        self.log("loss/train", loss_val, prog_bar=True, on_step=True)
        tqdm_dict = {"train_loss": loss_val, "lr": lr}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output
    
    def on_validation_epoch_start(self) -> None:
        self.val_outputs = []
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch: tuple, batch_nb: int, *args, **kwargs) -> dict:
        seq_ind, mask, annot_labels = (batch['seq_ind'], batch['mask'], batch['annot_mask'])
        esm_output = self.forward(seq_ind, mask)
        token_logits = esm_output.logits[:, :annot_labels.shape[1], :]
        annot_labels = annot_labels[:, :token_logits.shape[1]].long()
        annot_labels[~mask.bool()] = -100
        loss_val = F.cross_entropy(token_logits.reshape(-1, 2), annot_labels.reshape(-1), ignore_index=-100)
        if(loss_val.isnan().any() or loss_val.isinf().any()):
            print(f"Loss is NaN or Inf: {loss_val}")
            print(f"Inputs: {seq_ind}")
            print(f"batch nb: {batch_nb}")
            raise ValueError("Loss is NaN or Inf")
        self.log("loss/val", loss_val, prog_bar=True, on_step=True)
        annot_pred = torch.argmax(token_logits, dim=-1)
        annot_labels = annot_labels.cpu().numpy().flatten()
        annot_pred = annot_pred.cpu().numpy().flatten()
        valid_indices = annot_labels != -100
        output = OrderedDict({'annot_pred': annot_pred, 'annot_labels': annot_labels})
        self.val_outputs.append(output)


        tqdm_dict = {"val_loss": loss_val}
        output = OrderedDict(
            {"loss": loss_val, "progress_bar": tqdm_dict, "log": tqdm_dict})
        return output
    
    def on_validation_epoch_end(self) -> dict:
        outputs = self.val_outputs
        annot_labels = np.concatenate([x['annot_labels'] for x in outputs])
        annot_pred = np.concatenate([x['annot_pred'] for x in outputs])
        annot_pred = annot_pred[annot_labels != -100]
        annot_labels = annot_labels[annot_labels != -100]
        f1 = f1_score(annot_labels.flatten(), annot_pred.flatten())
        self.log('F1/val', f1, prog_bar=True)
    
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
            first_cycle_steps=(first_cycle_steps:=int(self.h.num_train_steps * 0.5)), 
            cycle_mult=1.0,
            max_lr_mul=1e1,
            warmup_steps=first_cycle_steps//5,
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
        parser.add_argument("--freeze_encoder", default=False, type=bool, 
            help="Whether to freeze the function-conditioned encoder"
        )
        parser.add_argument(
            "--gradient_clipping", default=1.0, type=float, help="Global norm gradient clipping"
        )
        parser.add_argument(
            "--weight_decay", default=0.01, type=float, help="Weight decay per train step."
        )
        return parser
    



    



    
