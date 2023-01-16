# -*- coding: utf-8 -*-

# Copyright 2023 Kuroyanagi Ibuki

"""Utility functions."""

import os
import random
import numpy as np
import torch
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hauc(y_true, y_pred, sections, domains=None, mode="all"):
    score_list = []
    for i in range(3):
        if mode == "all":
            idx = sections == i
        else:
            idx = (sections == i) & (domains == mode)
        score_list.append(roc_auc_score(y_true[idx], y_pred[idx]))
        score_list.append(roc_auc_score(y_true[idx], y_pred[idx], max_fpr=0.1))
    return hmean(score_list), np.array(score_list).std()
