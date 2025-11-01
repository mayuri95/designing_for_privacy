import os
import numpy as np
from pac_private_gd import est_L
import utils
import pandas as pd
import random
import string
import os
import data
import pickle
import sys
from models import LinearModel
import torch

num_trials = 1

mus = [1e-3, 1e-2, 1e-1, 1, 10]
T=50
for dataset in ['credit', 'mnist_7_vs_9']:
    for mu in mus:
        X, y, X_test, y_test, num_classes = data.load_dataset(dataset)
        e0 = utils.find_e0(X, y, num_classes, mu)
        n, d = X.shape
        L = est_L(X, mu)
        print(f'L={L}')
        num_clips = 0
        model = LinearModel(d, 1)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        per_sample_grads = utils.get_per_sample_grads(model, loss_fn, X, y, l2_lambda=mu).cpu().numpy()
        for d_i in range(d):

            def grad_i_fn(): # return a torch scalar
                return per_sample_grads[np.random.rand(num_features) < 0.5, d_i].mean().item()
                        
            grad_i_var = utils.exact_var_1d(per_sample_grads[:, d_i])
            eta_i = utils.optimal_eta(mu=mu, T=T, C=0, e0=e0[d_i], var=grad_i_var)
            assert eta_i >= 0
            eta_clip = np.clip(eta_i, 0, L)
            if eta_clip != eta_i:
                num_clips += 1
        print(f'dataset: {dataset} mu: {mu}, % clipped dims:{100*(num_clips/d)}')
        print('-------')
