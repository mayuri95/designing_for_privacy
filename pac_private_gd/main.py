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
budget_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
T_list = [50]
num_trials = 100
mu = 1.
T=50
e0_type_list = ['exact', 0.01, 0.1, 1.0]
dataset_list = [
    'mnist_0_vs_7',
    'mnist_7_vs_9',
    'credit'
]
e0_type_list = [e0_type_list[int(sys.argv[1])]]
dataset_list = [dataset_list[int(sys.argv[2])]]
print(e0_type_list)
for dataset in dataset_list:
    print(dataset)
    X, y, X_test, y_test, num_classes = data.load_dataset(dataset)
    e0 = find_e0(X, y, num_classes, mu)
    print(np.linalg.norm(e0))
    for inv_mi_budget in budget_list:
        for e0_type in e0_type_list:
            d = {}
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
                    print(trial_ind, test_acc)
                    accs.append(test_acc)
                d[privacy_aware] = accs
                test_accs = [k[0] for k in accs]
                print(privacy_aware, e0_type, inv_mi_budget, np.average(test_accs), np.std(test_accs))
            fname = f'results/{dataset}_data_budget={inv_mi_budget}_e0={e0_type}.pkl'
            pickle.dump(d, open(fname, 'wb'))
