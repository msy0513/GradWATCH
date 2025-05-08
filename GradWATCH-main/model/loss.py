import pandas as pd
import torch
import torch.nn as nn
import numpy as np
# from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

def prediction(pred_score, true_l):
    pred = pred_score.clone()
    pred = torch.where(pred > 0.5, 1, 0)
    pred = pred.detach().cpu().numpy()
    pred_score = pred_score.detach().cpu().numpy()
    true = true_l.cpu().numpy()

    recall = recall_score(true, pred)
    precision = precision_score(true, pred)
    f1 = f1_score(true, pred)
    auc = roc_auc_score(true, pred_score)

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return recall, precision, f1, auc, fpr, fnr


def Link_loss_meta(pred, y):
    L = nn.BCELoss()
    pred = pred.float()
    y = y.to(pred)
    loss = L(pred, y)

    return loss
