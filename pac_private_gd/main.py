import numpy as np
from pac_private_gd import pac_private_gd
from utils import find_e0

all_datasets = [
    'mnist_0_vs_7',
    'mnist_7_vs_9',
    'mnist'
]

# we will write everything into a csv file, of columns:
# dataset_name, mu, T, use_e0, mi_budget, privacy_aware, train_loss (this is a list), final_train_loss, test_acc
print("dataset_name, mu, T, e0_type, inv_mi_budget, privacy_aware, train_loss, final_train_loss, test_acc", flush=True)

for dataset in all_datasets:
    for mu in [1]:
        # compute e0, which is the global optimum
        e0 = find_e0(dataset, mu)
        for T in [50]:
            for e0_type in ['exact']:
                for inv_mi_budget in [None, 1, 2, 4, 8, 16, 32, 64, 128, 256]:
                    for privacy_aware in [True, False]:
                        if inv_mi_budget is None and privacy_aware == True:
                            continue # no need to run non-private privacy-aware
                        train_loss, test_acc = pac_private_gd(
                            dataset_name=dataset,
                            mu=mu,
                            T=T,
                            mi_budget=1/inv_mi_budget if inv_mi_budget is not None else None,
                            privacy_aware=privacy_aware,
                            e0=e0 if e0_type == 'exact' else np.ones_like(e0) * e0_type,
                            verbose=False
                        )
                        final_train_loss = train_loss[-1]
                        print(f"{dataset}, {mu}, {T}, {e0_type}, {inv_mi_budget}, {privacy_aware}, {train_loss}, {final_train_loss}, {test_acc}", flush=True)
