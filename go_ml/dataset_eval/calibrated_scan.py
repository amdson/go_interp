# eval_df['scan_logits']
import numpy as np
from scipy.special import softmax
eval_df['scan_scores'] = [None for _ in range(len(eval_df))]
for i, row in eval_df.iterrows():
    if row['scan_logits'] is not None:
        # row['scan_scores'] = np.var(row['scan_logits'], axis=2)
        prot_seq_mask = row['seq_len_mask']
        scan_logits = row['scan_logits']
        max_logits = np.max(scan_logits, axis=2, keepdims=True)
        scan_logits = max_logits - scan_logits
        scan_scores = np.max(scan_logits, axis=2)
        # scan_distr = softmax(scan_logits, axis=2)
        # scan_scores = -(np.log(scan_distr + 1e-6) * scan_distr).sum(axis=2)
        # scan_scores[~prot_seq_mask] = 3.0
        # row['scan_scores'] = 3-scan_scores
        eval_df.at[i, 'scan_scores'] = scan_scores
def get_percentile(scores_mat, mask_mat):
    scores_argsort = np.argsort(scores_mat - 1e5*(~mask_mat), axis=1)
    scores_ranks = np.argsort(scores_argsort, axis=1)
    scores_percentile = (scores_ranks / mask_mat.sum(axis=1, keepdims=True))
    scores_percentile[~mask_mat] = 0.0
    return scores_percentile
scan_scores = eval_df.at['llps', 'scan_scores']
bert_scores = eval_df.at['llps', 'bert_entropy']
annot_mat = eval_df.at['llps', 'annot_mat']
seq_len_mask = eval_df.at['llps', 'seq_len_mask']

np.random.seed(42)
calibration_mask = np.random.rand((scan_scores.shape[0])) < 0.2
bert_scores_cal = bert_scores[calibration_mask, :]
scan_scores_cal = scan_scores[calibration_mask, :]
annot_mat_cal = annot_mat[calibration_mask, :]
seq_len_mask_cal = seq_len_mask[calibration_mask, :]
bert_scores = bert_scores[~calibration_mask, :]
scan_scores = scan_scores[~calibration_mask, :]
annot_mat = annot_mat[~calibration_mask, :]
seq_len_mask = seq_len_mask[~calibration_mask, :]

bert_perc_cal = get_percentile(bert_scores_cal, seq_len_mask_cal)
scan_perc_cal = get_percentile(scan_scores_cal, seq_len_mask_cal)
bert_perc = get_percentile(bert_scores, seq_len_mask)
scan_perc = get_percentile(scan_scores, seq_len_mask)
base_features = np.stack([bert_scores_cal.flatten(), scan_scores_cal.flatten()], axis=1)
annot_targets = annot_mat_cal.flatten()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(base_features, annot_targets)
print(log_reg.coef_)

test_features = np.stack([bert_perc.flatten(), scan_perc.flatten()], axis=1)
test_pred = log_reg.predict_proba(test_features)[:, 1]
test_pred = test_pred.reshape(annot_mat.shape)
mrr = mean_reciprocal_rank_mat(test_pred, seq_len_mask, annot_mat)
auc, fpr_l, tpr_l = mean_auc(test_pred, seq_len_mask, annot_mat, return_roc=True)
top30 = top_30_score(test_pred, seq_len_mask, annot_mat)
print(f"Dataset: llps (Scan - Calibrated), MRR: {mrr:.4f}, AUC: {auc:.4f}, Top-30 Score: {top30:.4f}")
import matplotlib.pyplot as plt
fpr, tpr = roc_average(fpr_l, tpr_l)
plt.plot(fpr, tpr, label=f'llps (Scan - Calibrated)')

auc, fpr_l, tpr_l = mean_auc(bert_perc, seq_len_mask, annot_mat, return_roc=True)
fpr, tpr = roc_average(fpr_l, tpr_l)
plt.plot(fpr, tpr, label=f'llps (BERT)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

eval_df.at['llps', 'scan_logits'].shape
eval_df.at['llps', 'annot_mat']

import matplotlib.pyplot as plt

ind=21
plt.plot(eval_df.at['llps', 'scan_scores'][ind, :], label='scan')
plt.plot(-eval_df.at['llps', 'bert_entropy'][ind, :]+3.5, label='bert')
plt.plot(-eval_df.at['llps', 'pssm_entropy'][ind, :]+3.5, label='pssm')
plt.plot(eval_df.at['llps', 'annot_mat'][ind, :], label='annot')
plt.plot(1e6*eval_df.at['csa', 'ig_attribution_scores'][ind, :], label='ig')
plt.legend()
# ind=23
# plt.plot(-eval_df.at['csa', 'bert_entropy'][ind, :]+3, label='bert')
# plt.plot(-eval_df.at['csa', 'pssm_entropy'][ind, :]+6, label='bert')
