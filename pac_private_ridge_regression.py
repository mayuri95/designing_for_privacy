import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import Ridge
from typing import Tuple, Optional, Sequence
import pickle
from utils import *
import sys
import copy
from sklearn.decomposition import PCA, FastICA

TEST_SIZE   = 0.2
RANDOM_SEED = 42
NUM_SUBSETS = 1024
NUM_TRIALS = 1000

inv_mi_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
nonexact_inv_mi_values = [4, 16, 64, 256, 1024]
datasets = ['wine_white', 'wine_red', 'housing']

lams = [('exact', 0), ('exact', 16), ('exact', 1024)]
snr_types = [10, 'opt']
datasets = [datasets[int(sys.argv[1])]]
lams = [lams[int(sys.argv[2])]]
print(datasets, lams)

def create_subsets(X_train, num_subsets):
    subsets = {}
    num_dims = X_train.shape[1]
    for num_subsets in range(NUM_SUBSETS):
        pts = poisson_sample(X_train)
        subsets[num_subsets] = pts
    return subsets

def get_variances(subsets, lams, X_train, y_train_c):
    variances = {}
    n, d = X_train.shape
    for dim_ind in range(d):
        opt_lam0 = lams[dim_ind]
        ws = []
        for subset_ind in range(NUM_SUBSETS):
            pts = subsets[subset_ind]
            X_subset, Y_subset = X_train[pts], y_train_c[pts]
            ws.append(ridge_1d(X_subset[:, dim_ind], Y_subset, opt_lam0))
        variances[dim_ind] = np.var(ws, ddof=1)
    return variances

def get_snr_ratio(X_train, y_train_c):
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train_c
    w_ref = solve(XtX, Xty)
    n, d = X_train.shape

    resid = y_train_c - X_train @ w_ref
    sigma2_hat = float((resid @ resid) / (n-d)) # divide by df
    y_pred_base = ridge_pred(X_test, w_ref) + y_mean
    base_mse = mean_squared_error(y_pred_base, y_test)
    print('non-private baseline mse: ', base_mse)
    snrs = [sigma2_hat/w_ref[ind]**2 for ind in range(len(w_ref))]
    return w_ref, snrs


for lam_val in lams:    
    for data in datasets:
        for snr_type in snr_types:
            if snr_type == 'opt' and lam_val != ('exact', 0.):
                continue
            all_mses = {}
            if data == 'wine_red':
                X, y = load_wine_quality(red=True)
            elif data == 'wine_white':
                X, y = load_wine_quality(red=False)
            elif data == 'housing':
                X, y = load_cali_housing()

            # Train/test split and standardize X (fit on train)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test  = scaler.transform(X_test)

            pca = PCA(whiten=True, random_state=RANDOM_SEED)  # orthonormal columns, unit variance
            X_train = pca.fit_transform(X_train)          # (n, r) where r = rank
            X_test  = pca.transform(X_test)
            n, d = X_train.shape
            print(f'n={n}, d={d}')
            y_mean = y_train.mean()
            y_train_c = y_train - y_mean

            subsets = create_subsets(X_train, NUM_SUBSETS)
            w_ref, snrs = get_snr_ratio(X_train, y_train_c)
            if snr_type != 'opt':
                snrs = np.ones_like(snrs) * snr_type
            
            w_dim = w_ref.shape[0]
            base_C = 0.
            if type(lam_val) == tuple:
                assert lam_val[0] == 'exact'
                if lam_val[1] == 0:
                    base_C = 0.
                else:
                    mi_to_optimize = 1./lam_val[1]
                    base_C = w_dim / (2*mi_to_optimize)
                opt_lams = [(base_C+1)*snrs[ind] for ind in range(len(w_ref))]
            else:
                opt_lams = [lam_val for ind in range(len(w_ref))]
            base_variances = get_variances(subsets, opt_lams, X_train, y_train_c)

            for inv_mi in inv_mi_values:
                to_solve = True
                if inv_mi not in nonexact_inv_mi_values:
                    if lam_val != ('exact', 0.):
                        to_solve = False
                    if snr_type != 'opt':
                        to_solve = False
                # if lam_val != ('exact', 0.) and snr_type != 'opt':
                #     to_solve = False
                if not to_solve:
                    continue
                mi = 1/inv_mi
                C = w_dim / (2*mi)

                priv_obl_mses = pac_private_ridge_regression(
                    X=X_train,
                    y_c=y_train_c,
                    subsets=subsets,
                    lambs=opt_lams,
                    C=C,
                    variances=base_variances,
                    X_test=X_test,
                    y_test=y_test,
                    y_mean=y_mean,
                    num_trials=NUM_TRIALS
                )

                print(f'inv mi={inv_mi}, C={C}, priv oblivious mse: ', np.mean(priv_obl_mses))

                mi_to_opt = None
                correction_factor = (C+1)
                corrected = False
                if type(lam_val) == tuple:
                    assert lam_val[0] == 'exact'
                    if abs(C-base_C) < 1e-12:
                        priv_aware_lam = copy.deepcopy(opt_lams)
                        corrected = True
                    else:
                        correction_factor = (C+1) / (base_C+1)
                        priv_aware_lam = [correction_factor * opt_lams[dim_ind] for dim_ind in range(len(opt_lams))]
                        corrected = True
                if not corrected:
                    priv_aware_lam = [correction_factor * opt_lams[dim_ind] for dim_ind in range(len(opt_lams))]
                variances = get_variances(subsets, priv_aware_lam, X_train, y_train_c)           
     
                priv_aware_mses = pac_private_ridge_regression(
                    X=X_train,
                    y_c=y_train_c,
                    subsets=subsets,
                    lambs=priv_aware_lam,
                    C=C,
                    variances=variances,
                    X_test=X_test,
                    y_test=y_test,
                    y_mean=y_mean,
                    num_trials=NUM_TRIALS
                )
                
                print(f'C={C}, inv budget={inv_mi}, priv aware mse: ', np.mean(priv_aware_mses))
                all_mses[inv_mi] = (priv_obl_mses, priv_aware_mses)

            pickle.dump(all_mses, open(f'results/{data}_{lam_val}_{snr_type}_mses.pkl', 'wb'))
