import os, json, pickle
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from go_ml.models.warmup_esm_finetune import ESMFinetune
from go_ml.data_utils import *
from argparse import ArgumentParser
import transformers

parser = ArgumentParser()
parser = ESMFinetune.add_model_specific_args(parser)
hparams = parser.parse_args()
print("got hparams", hparams, type(hparams))

import pickle
data_path = "/home/andrew/GO_interp/data/train_esm_datasets/"
with open(f"{data_path}/train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)
with open(f"{data_path}/val_dataset.pkl", "rb") as f:
    val_dataset = pickle.load(f)
tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.model_name)
from go_ml.data_utils import prot_func_collate

hparams.num_classes = train_dataset[0]['labels'].shape[0]
hparams.num_train_steps = 10*len(train_dataset)
hparams.cls_warmup = False
hparams.learning_rate = 3e-6

checkpoint_path = "/home/andrew/GO_interp/checkpoints/esm_finetune-v3.ckpt"
model = ESMFinetune.load_from_checkpoint(checkpoint_path, model_args=hparams)
model.unfreeze_encoder()  # Unfreeze the encoder for fine-tuning

if __name__ == "__main__":
    tokenizer = transformers.AutoTokenizer.from_pretrained(hparams.model_name)
    dataloader_params = {"shuffle": True, "batch_size": 2, "collate_fn": prot_func_collate}
    val_dataloader_params = {"shuffle": False, "batch_size": 6, "collate_fn": prot_func_collate}

    train_loader = DataLoader(train_dataset, **dataloader_params)
    val_loader = DataLoader(val_dataset, **val_dataloader_params)
    # model = ESMFinetune(hparams)
    # model = torch.compile(model)
    
    early_stop_callback = EarlyStopping(monitor='F1/val', min_delta=0.00, patience=3, verbose=True, mode='max')
    # early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=0.00, patience=3, verbose=True, mode='min')
    checkpoint_callback = ModelCheckpoint(filename="/home/andrew/GO_interp/checkpoints/esm_finetune_fin", 
                                          verbose=True, monitor='loss/val', mode='min')
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger("logs", name="esm_finetune_fin", default_hp_metric=False)
    trainer = pl.Trainer(devices=[0], max_epochs=10, 
                         callbacks=[early_stop_callback, checkpoint_callback], 
                         logger=logger, gradient_clip_val=1.0, accumulate_grad_batches=3, precision='bf16-mixed')
    #, 
    #                     accumulate_grad_batches=2)#, precision="bf16-mixed")
    # trainer = pl.Trainer(devices=[0], max_epochs=10, 
    #                      callbacks=[early_stop_callback, checkpoint_callback], logger=logger) precision="bf16-mixed")
    trainer.fit(model, train_loader, val_loader)