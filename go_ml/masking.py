import torch

def mask_range(seq_batch, si, ei, sequence_mask_token, mut_per=0.15):
    device = seq_batch.device
    batch_size, seq_len = seq_batch.shape
    # seq_len = torch.LongTensor([seq_len - 2]) #Discount SOS and EOS tokens at start and end
    range_len = torch.LongTensor([ei-si])
    mut_count = torch.floor(range_len*mut_per).int().item()
    mut_inds = torch.stack([torch.randperm(range_len) for _ in range(batch_size)])[:, :mut_count] + si
    batch_inds = torch.tile(torch.arange(0, mut_inds.shape[0]).reshape((-1, 1)), (1, mut_count))
    mut_inds, batch_inds = mut_inds.to(device), batch_inds.to(device)
    update_batch = seq_batch.clone()
    update_batch[batch_inds, mut_inds] = sequence_mask_token
    return update_batch

#Masking functions for ESM3 only. Requires input to be augmented with SOS and EOS tokens
def mask_perc(seq_ind, mask_token, residue_coverage=6, mut_per=0.15):
    device = seq_ind.device
    seq_len = torch.LongTensor([seq_ind.shape[0] - 2]).to(device) #Discount SOS and EOS tokens at start and end
    mut_count = torch.floor(seq_len*mut_per).int().item()
    total_muts = (torch.floor(seq_len*residue_coverage/mut_count)*mut_count).int().item()
    
    mut_inds = (torch.randperm(total_muts).reshape(-1, mut_count).to(device) % seq_len) + 1
    batch_inds = torch.tile(torch.arange(0, mut_inds.shape[0]).reshape((-1, 1)), (1, mut_count))
    mut_inds, batch_inds = mut_inds.to(device), batch_inds.to(device)

    batch = torch.tile(seq_ind, (mut_inds.shape[0], 1))
    batch[batch_inds, mut_inds] = mask_token
    # print((batch==mask_token).sum())
    return batch, batch_inds, mut_inds

def mask_indiv(seq_ind, mask_token):
    seq_len = seq_ind.shape[0] - 2 #Discount first and last tokens
    batch = torch.tile(seq_ind, (seq_len, 1))
    batch_ind = torch.arange(seq_len) 
    mut_ind = batch_ind + 1
    batch[batch_ind, mut_ind] = mask_token
    return batch, batch_ind, mut_ind

def mask_indiv_neighborhood(seq_ind, mask_token, n_rad=5):
    seq_len = seq_ind.shape[0] - 2 #Discount first and last tokens
    batch = torch.tile(seq_ind, (seq_len, 1))
    batch_ind = torch.arange(seq_len)
    mut_ind = batch_ind + 1
    col_ind = torch.arange(seq_ind.shape[0]).reshape(1, -1)
    col_ind = torch.tile(col_ind, (seq_len, 1))
    mut_delta = mut_ind.reshape(-1, 1) - col_ind
    mut_mask = torch.abs(mut_delta) <= n_rad
    mut_mask[:, 0] = False; mut_mask[:, -1] = False #Don't mess with sos, eos tokens
    batch[mut_mask] = mask_token
    return batch, batch_ind, mut_ind

import math
def mask_span(seq_ind, mask_token, residue_coverage=3, span_rad=55, run_len=4, mask_per=0.3):
    seq_len = seq_ind.shape[0] - 2
    span_rad = min(span_rad, seq_len//2)
    span_center = torch.linspace(1, seq_len+1, steps=int((seq_ind.shape[0] - 2) / (span_rad-1))).int()
    span_rep = math.ceil(residue_coverage / mask_per)

    span_mask_mat = torch.zeros(span_center.shape[0], span_rep, seq_len+2)
    span_mask_mat_ind = torch.arange(seq_len+2).unsqueeze(0).unsqueeze(0).expand(span_center.shape[0], span_rep, seq_len+2)
    span_mask_mat[torch.abs(span_mask_mat_ind - span_center.reshape(-1, 1, 1)) <= span_rad] = 1

    run_start = torch.linspace(-run_len / 2, 2*span_rad+run_len*2, int(2*span_rad * residue_coverage / run_len))
    run_start = torch.tile(run_start.unsqueeze(0), (span_center.shape[0], 1))
    run_row = torch.arange(0, run_start.shape[0]).unsqueeze(1).expand_as(run_start)
    perm_mat = torch.argsort(torch.rand(*run_start.shape), dim=1)
    run_start += span_center.reshape(-1, 1) - span_rad
    run_start = run_start[run_row, perm_mat].int() #span_id x str_id

    span_rep_id = torch.arange(run_start.shape[1]) % span_rep
    span_rep_id = span_rep_id.unsqueeze(0).expand_as(run_start)

    run_ind = run_start.unsqueeze(1).tile((1, run_len, 1))
    run_ind = run_ind + torch.arange(run_len).unsqueeze(0).unsqueeze(2) + 1
    run_ind[run_ind >= seq_len+1] = seq_len+1
    run_ind[run_ind <= 0] = 0

    span_rep_id = span_rep_id.unsqueeze(1).expand((-1, run_len, -1))
    run_row = run_row.unsqueeze(1).expand((-1, run_len, -1))

    mask_mat = torch.zeros(span_center.shape[0], span_rep, seq_len+2)
    mask_mat[run_row, span_rep_id, run_ind] = 1
    mask_mat[:, :, 0] = 0; mask_mat[:, :, -1] = 0

    mask_mat = mask_mat.reshape(-1, seq_len+2).bool()
    span_mask_mat = span_mask_mat.reshape(-1, seq_len+2).bool()

    batch = torch.tile(seq_ind, (mask_mat.shape[0], 1))
    batch[~span_mask_mat] = mask_token
    batch[mask_mat] = mask_token
    batch_inds, mut_inds = torch.nonzero(mask_mat, as_tuple=True)
    return batch, batch_inds, mut_inds

def mask_avg(bert_mask, bert_eval):
    eval_idx = torch.nonzero(bert_mask) ##Nonzero x 2(ij)
    eval_support = torch.sum(bert_mask, dim=0)
    eval_samples = bert_eval[eval_idx[:, 0], eval_idx[:, 1], :]
    eval_avg = torch.zeros_like(bert_eval[0])
    gather_ind = torch.tile(eval_idx[:, 1:2], (1, bert_eval.shape[-1]))
    eval_avg = eval_avg.scatter_reduce(0, gather_ind, eval_samples, 'mean', include_self=False)
    return eval_avg, eval_support

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.constants.esm3 import (
    SEQUENCE_MASK_TOKEN,
)
import torch
def get_logits_esmc(seq, model, batch_size=8, mask_func=mask_indiv):
    seq_ind = model.encode(ESMProtein(sequence=seq)).sequence
    batch, batch_inds, mut_inds = mask_func(seq_ind, SEQUENCE_MASK_TOKEN)
    bert_eval_l = []
    with torch.no_grad():
        for si in range(0, batch.shape[0], batch_size):
            ei = min(batch.shape[0], si+batch_size)
            x = batch[si:ei, :]
            model_eval = model(x)
            bert_eval = model_eval.sequence_logits
            bert_eval_l.append(bert_eval.cpu())
    bert_eval = torch.cat(bert_eval_l)
    # print(bert_eval.mean())
    bert_eval = torch.softmax(bert_eval, dim=2)
    # print(bert_eval.mean(dim=2).mean(dim=1))
    bert_mask = (batch == SEQUENCE_MASK_TOKEN).cpu()
    eval_avg, eval_support = mask_avg(bert_mask, bert_eval)
    return eval_avg#.numpy()


def get_logits_esmfast(seq, model, tokenizer, batch_size=8, mask_func=mask_indiv):
    # tokenizer = model.tokenizer
    device = model.device
    SEQUENCE_MASK_TOKEN = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    seq_ind = tokenizer.batch_encode_plus(
            [seq],
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=1024,
            return_tensors='pt'
        )['input_ids'].to(device)[0]
    batch, batch_inds, mut_inds = mask_func(seq_ind, SEQUENCE_MASK_TOKEN)
    print()
    bert_eval_l = []
    with torch.no_grad():
        for si in range(0, batch.shape[0], batch_size):
            ei = min(batch.shape[0], si+batch_size)
            x = batch[si:ei, :]
            model_eval = model(x)
            bert_eval = model_eval.logits
            bert_eval_l.append(bert_eval.cpu())
    bert_eval = torch.cat(bert_eval_l)
    # print(bert_eval.mean())
    bert_eval = torch.softmax(bert_eval, dim=2)
    # print(bert_eval.mean(dim=2).mean(dim=1))
    bert_mask = (batch == SEQUENCE_MASK_TOKEN).cpu()
    eval_avg, eval_support = mask_avg(bert_mask, bert_eval)
    return eval_avg#.numpy()

def get_logits_cond(seq, func_labels, model, batch_size=8, mask_func=mask_perc):
    batch = model.tokenizer.batch_encode_plus([seq], add_special_tokens=True, max_length=850, truncation=True, return_tensors="pt")
    device = model.device
    seq_ind = batch['input_ids'].squeeze(0)
    mask = batch['attention_mask'].squeeze(0)
    batch, batch_inds, mut_inds = mask_func(seq_ind, SEQUENCE_MASK_TOKEN)
    active_func_labels = func_labels[model.active_labels]
    active_func_labels = torch.tile(active_func_labels.unsqueeze(0), (batch.shape[0], 1))
    tile_mask = torch.tile(mask.unsqueeze(0), (batch.shape[0], 1))
    # print(batch.shape, tile_mask.shape)
    bert_eval_l = []
    with torch.no_grad():
        for si in range(0, batch.shape[0], batch_size):
            ei = min(batch.shape[0], si+batch_size)
            x = batch[si:ei, :].to(device)
            m = tile_mask[si:ei, :].to(device)
            fl = active_func_labels[si:ei, :].to(device)
            model_eval = model.forward(x, m, fl)
            bert_eval = model_eval
            bert_eval_l.append(bert_eval.cpu())
    bert_eval = torch.cat(bert_eval_l)
    bert_eval = torch.softmax(bert_eval, dim=2)
    bert_mask = torch.zeros_like(batch).bool()
    bert_mask[batch_inds, mut_inds] = 1
    eval_avg, eval_support = mask_avg(bert_mask, bert_eval)
    return eval_avg#.numpy()