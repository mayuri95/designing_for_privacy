import numpy as np
from pac_private_gd import pac_private_gd
from utils import find_e0
import sys
import pickle

all_datasets = [
    'wine_quality'
    # 'adult', # something weird happens here
    # 'mnist' # this one is very slow
]

# we will write everything into a csv file, of columns:
# dataset_name, mu, T, use_e0, mi_budget, privacy_aware, train_loss (this is a list), final_train_loss, test_acc
budgets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
start_ind = int(sys.argv[1])
budgets = budgets[start_ind: start_ind+2]
for dataset in all_datasets:
    for mu in [0.1]: # different level of regularization
        e0 = find_e0(dataset, mu) # compute e0, which is the global optimum
        for T in [50]: # fixed number of iterations
            for e0_type in ['exact', 0.01, 0.001, 0.1]: # different ways to set e0, exact or prior on initial bias
                results = {}
                for inv_mi_budget in budgets:
                    for privacy_aware in [True, False]:
                        results[(inv_mi_budget, privacy_aware)] = []
                        for trial in range(100):
                            print(trial)
                            train_loss, test_acc = pac_private_gd(
                                dataset_name=dataset,
                                mu=mu,
                                T=T,
                                mi_budget=1/inv_mi_budget if inv_mi_budget is not None else None,
                                privacy_aware=privacy_aware,
                                e0=e0 if e0_type == 'exact' else np.ones_like(e0) * e0_type,
                                verbose=False
                            )
                            results[(inv_mi_budget, privacy_aware)].append((train_loss, test_acc))
                pickle.dump(results, open(f'data/{dataset}_{e0_type}_{budgets}_mses.pkl', 'wb'))
