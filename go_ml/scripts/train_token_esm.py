import os, json, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_ml.models.esm_token_finetune import ESMCTokenFinetune
from go_ml.data_utils import *
from argparse import ArgumentParser
import transformers


parser = ArgumentParser()
parser = ESMCTokenFinetune.add_model_specific_args(parser)
hparams = parser.parse_args()
print("got hparams", hparams, type(hparams))

import pickle
import pandas as pd
from go_ml.eval_utils import filter_annot_df
from go_ml.eval_utils import (load_msa_dict, gen_bert_mat, get_bert_entropy, 
                              gen_annot_mat, gen_seq_len_mask, mean_reciprocal_rank, 
                              mean_reciprocal_rank_mat, mean_auc, top_30_score, roc_average)

data_root = '/home/andrew/GO_interp/go_ml/gen_datasets/datasets'
df_f = ['csa_annot', 'llps_dataset', 'elms_dataset'][2]
annot_df = filter_annot_df(pd.read_csv(f'{data_root}/{df_f}.csv', sep='\t'))

hparams.train_perc = 0.1

np.random.seed(42)
train_row_mask = np.random.rand(len(annot_df)) > (1 - hparams.train_perc)
train_df = annot_df[train_row_mask]
val_df = annot_df[~train_row_mask]

import json
train_path = "/home/andrew/cafa5_team/data/"
with open(f"{train_path}/cafa_dataset/go_terms.json", "r") as f:
    go_terms = json.load(f)
# tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.model_name)
dataloader_params = {"shuffle": True, "batch_size": 3, "collate_fn": prot_func_collate}
val_dataloader_params = {"shuffle": False, "batch_size": 12, "collate_fn": prot_func_collate}

train_dataset = ProtFuncDataset.from_annot_df(train_df, go_terms)
val_dataset = ProtFuncDataset.from_annot_df(val_df, go_terms)
train_loader = DataLoader(train_dataset, **dataloader_params)
val_loader = DataLoader(val_dataset, **dataloader_params)

hparams.num_train_steps = 10*len(train_df)

model = ESMCTokenFinetune(hparams)
early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=0.00, patience=3, verbose=True, mode='min')
checkpoint_callback = ModelCheckpoint(filename=f"/home/andrew/GO_interp/checkpoints/annot_esm_warmup-perc{hparams.train_perc}-{df_f.split('_')[0]}",
                                        verbose=True, monitor='loss/val', mode='min')
from pytorch_lightning.loggers import TensorBoardLogger
logger = TensorBoardLogger("annot_logs", name=f"esmc_warmup_{df_f.split('_')[0]}", default_hp_metric=False)
trainer = pl.Trainer(devices=[0], max_epochs=10, 
                        callbacks=[early_stop_callback, checkpoint_callback], 
                        logger=logger, precision="bf16-mixed")
trainer.fit(model, train_loader, val_loader)