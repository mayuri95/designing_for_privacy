import numpy as np
from pac_private_gd import pac_private_gd
from utils import find_e0
import pandas as pd

dataset_list = [
    'bank',
    'mnist0_vs_7',
    'mnist7_vs_9'
]

results_df = pd.DataFrame(columns=[
    'dataset_name', 'mu', 'T', 'use_e0', 'inverse_mi_budget', 'privacy_aware',
    'train_loss_list', 'final_train_loss', 'test_acc'
])

budget_list= [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
e0_type_list = ['exact', 0.001, 0.01, 0.1]
mu_list = [1]
T_list = [50]

for dataset in dataset_list:
    for mu in mu_list:
        e0 = find_e0(dataset, mu)
        for T in [50]:
            for e0_type in e0_type_list:
                for inv_mi_budget in budget_list:
                    for privacy_aware in [True, False]:
                        for _ in range(100):
                            train_loss, test_acc = pac_private_gd(
                                dataset_name=dataset,
                                mu=mu,
                                T=T,
                                mi_budget=1/inv_mi_budget if inv_mi_budget is not None else None,
                                privacy_aware=privacy_aware,
                                e0=e0 if e0_type == 'exact' else np.ones_like(e0) * e0_type,
                                verbose=False
                            )
                            new_row = {
                                'dataset_name': dataset,
                                'mu': mu,
                                'T': T,
                                'use_e0': e0_type,
                                'inverse_mi_budget': inv_mi_budget,
                                'privacy_aware': privacy_aware,
                                'train_loss_list': train_loss,
                                'final_train_loss': train_loss[-1],
                                'test_acc': test_acc
                            }
                            results_df = results_df.append(new_row, ignore_index=True)

results_df.to_csv('pac_private_gd_experiments.csv', index=False)