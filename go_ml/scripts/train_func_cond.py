import os, json, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_ml.data_utils import *
from argparse import ArgumentParser
import transformers
from go_ml.models.func_cond_esm import FuncCondESM, FuncCondESMFinetune

from go_ml.data_utils import prot_func_collate_bert, ProtFuncDataset, BertFuncDataset

import pickle
data_path = "/home/andrew/GO_interp/data/train_esm_datasets/"
with open(f"{data_path}/train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)
with open(f"{data_path}/val_dataset.pkl", "rb") as f:
    val_dataset = pickle.load(f)

parser = ArgumentParser()
parser.add_argument(
            "--gpu_id",
            default=0,
            type=int,
            help="GPU ID to use for training",
        )
parser.add_argument(
            "--mask_func",
            default='perc',
            type=str,
            help="Mask func to use for input sequences: 'perc' for percentage masking, 'span' for span masking",
        )

parser = FuncCondESMFinetune.add_model_specific_args(parser)
hparams = parser.parse_args()

if(hparams.mask_func == 'perc'):
    mask_func = bert_mask_alias
elif(hparams.mask_func == 'span'):
    mask_func = bert_span_mask_alias
else:
    raise ValueError("Invalid mask_func. Choose 'perc' or 'span'.")

train_dataset = BertFuncDataset.from_prot_func_dataset(train_dataset, mask_func=mask_func)
val_dataset = BertFuncDataset.from_prot_func_dataset(val_dataset, mask_func=mask_func)

dataloader_params = {"shuffle": True, "batch_size": 10, "collate_fn": prot_func_collate_bert}
val_dataloader_params = {"shuffle": False, "batch_size": 12, "collate_fn": prot_func_collate_bert}

train_loader = DataLoader(train_dataset, **dataloader_params)
val_loader = DataLoader(val_dataset, **val_dataloader_params)

hparams.label_counts = train_dataset.labels.sum(axis=0).A1
hparams.num_train_steps = len(train_dataset)*15
hparams.learning_rate = 1e-6

model = FuncCondESMFinetune(hparams)
print('hparams mask func encoder', hparams.freeze_func_encoder)
print('func enc grad', model.model.func_emb.requires_grad)

early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=0.00, patience=2, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(filename="/home/andrew/GO_interp/checkpoints/func_cond_finetune", 
                                        verbose=True, monitor='loss/val', mode='min')
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("cond_logs", name="func_cond_finetune", default_hp_metric=False)
#log hyperparameters except model weights
logger.log_hyperparams(hparams)
trainer = pl.Trainer(devices=[hparams.gpu_id], max_epochs=10, 
                        callbacks=[early_stop_callback, checkpoint_callback], 
                        logger=logger, gradient_clip_val=1.0, accumulate_grad_batches=4, precision='bf16-mixed')
trainer.fit(model, train_loader, val_loader)
