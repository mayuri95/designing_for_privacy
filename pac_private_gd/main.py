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
    'bank'
]

random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
filename = os.path.join(output_dir, f'{random_string}.csv')
write_header = not os.path.exists(filename)

budget_list= [1]
e0_type_list = ['exact']
mu_list = [0.0001]
T_list = [50]
num_trials = 3
e0 = 'exact'

import numpy as np
import torch
import random
import os

def stable_training_env(seed: int = 42, precision: str = "float64", verbose: bool = False):
    """
    Enforces deterministic RNG, dtype, and numerical precision across NumPy / Torch / SciPy / sklearn.
    Use this at the start of any training run to get reproducible results across library versions.
    """

    # ---- Seed everything reproducibly ----
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- Deterministic algorithms (Torch) ----
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False

    # ---- Default precision ----
    if precision == "float64":
        torch.set_default_dtype(torch.float64)
        np.set_printoptions(precision=8, suppress=True)
    elif precision == "float32":
        torch.set_default_dtype(torch.float32)
        np.set_printoptions(precision=6, suppress=True)
    else:
        raise ValueError("precision must be 'float64' or 'float32'")

    # ---- NumPy numerical guards ----
    np.seterr(all="raise")

    # ---- Control randomness sources ----
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for deterministic torch.cuda ops

    if verbose:
        print(f"[stable_training_env] seed={seed}, precision={precision}, deterministic=True")
        print(f"NumPy {np.__version__}, Torch {torch.__version__}")

    # ---- Return a stable RNG for NumPy (new API) ----
    rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(seed)
    return rng
rng = stable_training_env(seed=42, precision="float32", verbose=True)


for dataset in dataset_list:
    X, y, X_test, y_test, num_classes = data.load_dataset(dataset)
    for mu in mu_list:
        e0 = find_e0(X, y, num_classes, mu)
        for T in [50]:
            for e0_type in e0_type_list:
                for inv_mi_budget in budget_list:
                    for privacy_aware in [True, False]:
                        test_accs = []
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
                                verbose=False,
                                rng=rng
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
                            test_accs.append(test_acc)
                            print(privacy_aware, test_acc)
                        print(privacy_aware,np.average(test_accs))
    del X, y, X_test, y_test
