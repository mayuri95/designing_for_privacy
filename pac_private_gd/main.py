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

dataset_list = [
    'credit'
]

budget_list = [None, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
budget_list = [budget_list[int(sys.argv[1])]]
T_list = [50]
num_trials = 1
mu = 1.
T=50
e0_type_list = ['exact', 0.001, 0.01, 0.1]

for dataset in dataset_list:
    X, y, X_test, y_test, num_classes = data.load_dataset(dataset)
    e0 = find_e0(X, y, num_classes, mu)
    for inv_mi_budget in budget_list:
        d = {}
        for e0_type in e0_type_list:
            d[e0_type] = {}
            for privacy_aware in [True, False]:
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
                        privacy_aware=privacy_aware,
                        e0=e0 if e0_type == 'exact' else np.ones_like(e0) * e0_type,
                        verbose=False
                    )
                    print(trial_ind)
                    accs.append(test_acc)
                d[e0_type][privacy_aware] = accs
                print(privacy_aware, inv_mi_budget, np.average(accs), np.std(accs))
        fname = 'results/credit_data_budget={}.pkl'.format(inv_mi_budget)
        pickle.dump(d, open(fname, 'wb'))
