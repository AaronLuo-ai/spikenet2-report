import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.signal import resample_poly
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
import os
import json

def save_curves(hist, path, tag):
    h = np.array(hist); plt.figure(figsize=(5, 3))
    plt.plot(h[:, 0], label='train'); plt.plot(h[:, 1], label='val'); plt.legend(); plt.title(tag)
    plt.savefig(f'{path}/loss_{tag}.png'); plt.close()