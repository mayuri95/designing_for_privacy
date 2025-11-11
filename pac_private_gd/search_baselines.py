import os
import numpy as np
from pac_private_gd import pac_private_gd
from utils import find_e0
import pandas as pd
import random
import string
import os
import data
import pickle
import sys

# run as budget ind, e0 ind, dataset ind
budget_list = [4, 16, 64, 256, 1024]
T_list = [50]
num_trials = 500
mu = 1.
T=50
dataset_list = [
    'credit',
    'mnist_7_vs_9',
    'mnist_0_vs_7'
]
priv_oblivious_mi_budgets = [16, 1024]

budget_list = [budget_list[int(sys.argv[1])]]
priv_oblivious_mi_budgets = [priv_oblivious_mi_budgets[int(sys.argv[2])]]

print(budget_list, priv_oblivious_mi_budgets, dataset_list)
for dataset in dataset_list:
    print(dataset)
    X, y, X_test, y_test, num_classes = data.load_dataset(dataset)
    e0 = 0.1 * np.ones(X.shape[1])
    for inv_mi_budget in budget_list:
        for priv_oblivious_mi_budget in priv_oblivious_mi_budgets:
            d = {}
            for privacy_aware in [False]:
                accs = []
                for trial_ind in range(num_trials):
                    train_loss, cla_loss, test_acc = pac_private_gd(
                        X=X,
                        y=y,
                        X_test=X_test,
                        y_test=y_test,
                        num_classes=num_classes,
                        mu=mu,
                        T=T,
                        mi_budget=1/inv_mi_budget if inv_mi_budget is not None else None,
                        priv_oblivious_mi_budget=1./priv_oblivious_mi_budget,
                        privacy_aware=privacy_aware,
                        e0=e0,
                        verbose=False
                    )
                    print(trial_ind, test_acc)
                    accs.append(test_acc)
                d[privacy_aware] = accs
                test_accs = [k[0] for k in accs]
                print(privacy_aware, inv_mi_budget, np.average(test_accs), np.std(test_accs))
            fname = f'results/{dataset}_data_budget={inv_mi_budget}_baseline_priv_obl={priv_oblivious_mi_budget}.pkl'
            pickle.dump(d, open(fname, 'wb'))
