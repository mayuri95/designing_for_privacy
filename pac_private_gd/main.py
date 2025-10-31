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
output_dir = os.path.join(pwd, 'final_results')
os.makedirs(output_dir, exist_ok=True)

dataset_list = [
    'bank',
]

random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
filename = os.path.join(output_dir, f'{random_string}.csv')
write_header = not os.path.exists(filename)

budget_list= [64]
e0_type_list = ['exact'] # exact for now
T_list = [50]
num_trials = 10

for dataset in dataset_list:
    X, y, X_test, y_test, num_classes = data.load_dataset(dataset)
    e0 = find_e0(X, y, num_classes)
    for T in [50]:
        for e0_type in e0_type_list:
            for inv_mi_budget in budget_list:
                for privacy_aware in [True, False]:
                    accs = []
                    for trial_ind in range(num_trials):
                        train_loss, cla_loss, test_acc = pac_private_gd(
                            X=X,
                            y=y,
                            X_test=X_test,
                            y_test=y_test,
                            num_classes=num_classes,
                            T=T,
                            mi_budget=1/inv_mi_budget if inv_mi_budget is not None else None,
                            privacy_aware=privacy_aware,
                            e0=e0 if e0_type == 'exact' else np.ones_like(e0) * e0_type,
                            verbose=False,
                            seed = 3140931*trial_ind
                        )
                        print(privacy_aware, inv_mi_budget, test_acc)
                        accs.append(test_acc)
                        
                    print(privacy_aware, inv_mi_budget, np.average(accs), np.std(accs))
    del X, y, X_test, y_test

