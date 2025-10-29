import os
import numpy as np
from pac_private_gd import pac_private_gd
from utils import find_e0
import pandas as pd
import random
import string
import os
import data

pwd = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(pwd, 'new_results')
os.makedirs(output_dir, exist_ok=True)

dataset_list = [
    'bank',
    'mnist_0_vs_7',
    'mnist_7_vs_9'
]

random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
filename = os.path.join(output_dir, f'{random_string}.csv')
write_header = not os.path.exists(filename)

budget_list= [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
e0_type_list = ['exact', 0.001, 0.01]
mu_list = [1.0]
T_list = [50]
num_trials = 50

for dataset in dataset_list:
    X, y, X_test, y_test, num_classes = data.load_dataset(dataset)
    for mu in mu_list:
        e0 = find_e0(X, y, num_classes, mu)
        for T in [50]:
            for e0_type in e0_type_list:
                for inv_mi_budget in budget_list:
                    for privacy_aware in [True, False]:
                        for _ in range(num_trials):
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
                            results = {
                                'dataset_name': dataset,
                                'mu': mu,
                                'T': T,
                                'use_e0': e0_type,
                                'inverse_mi_budget': inv_mi_budget,
                                'privacy_aware': privacy_aware,
                                'train_loss_list': train_loss,
                                'cla_loss_list': cla_loss,
                                'final_train_loss': train_loss[-1],
                                'test_acc': test_acc
                            }
                            df = pd.DataFrame([results])
                            df.to_csv(filename, mode='a', index=False, header=write_header)
                            write_header = False
                            del df
                            del results
    del X, y, X_test, y_test
