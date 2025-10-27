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
from sklearn.decomposition import PCA

TEST_SIZE   = 0.3
RANDOM_SEED = 42
NUM_SUBSETS = 128
NUM_TRIALS = 100

C_values = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]
#C_values = [C_values[int(sys.argv[1])]]
print(C_values)
datasets = ['wine_red', 'wine_white']
datasets = [datasets[int(sys.argv[1])]]
all_mses, all_lams = {}, {}
for data in datasets:
    if data == 'wine_red':
        X, y = load_wine_quality(red=True)
    elif data == 'wine_white':
        X, y = load_wine_quality(red=False)
    elif data == 'census':
        X, y = load_adult_census('adult/adult.data')
    elif data == 'bank':
        X, y = load_bank()

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
    y_mean = y_train.mean()
    y_train_c = y_train - y_mean

    rng = np.random.default_rng(RANDOM_SEED)

    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train_c
    w_ref = solve(XtX, Xty)

    resid = y_train_c - X_train @ w_ref
    sigma2_hat = float((resid @ resid) / (n-d)) # divide by df
    y_pred_base = ridge_pred(X_test, w_ref) + y_mean
    base_mse = mean_squared_error(y_pred_base, y_test)
    print('non-private baseline mse: ', base_mse)

    opt_ws = []
    for dim_ind in range(d):
        ridge_model_no_intercept = Ridge(alpha=0, fit_intercept=False)
        ridge_model_no_intercept.fit(np.transpose(
            np.atleast_2d(X_train[:, dim_ind])), y_train_c)
        opt_ws.append(ridge_model_no_intercept.coef_[0])

    opt_lams = [sigma2_hat/opt_ws[ind]**2 for ind in range(len(opt_ws))]
    base_variances = {}
    subsets = []
    all_lams[0] = opt_lams
    for dim_ind in range(d):
        print(f'dim ind {dim_ind} {d}')
        opt_lam0 = opt_lams[dim_ind]
        ws = []
        for num_subsets in range(NUM_SUBSETS):
            pts = poisson_sample(X_train)
            subsets.append(pts)
            X_subset, Y_subset = X_train[pts], y_train_c[pts]
            ws.append(ridge_1d(X_subset[:, dim_ind], Y_subset, opt_lam0))
        base_variances[dim_ind] = np.var(ws)
    pickle.dump(base_variances, open(f'data/baseline_{data}_variances.pkl', 'wb'))

    for C in C_values:
        priv_obl_mses = []
        for trial in range(NUM_TRIALS):
            print(trial)
            unnoised_ws = []
            release = []
            for dim_ind in range(d):
                chosen_pts = subsets[np.random.choice(range(len(subsets)))]
                opt_lam0 = opt_lams[dim_ind]
                w = ridge_1d(
                    X_train[chosen_pts][:, dim_ind],
                    y_train_c[chosen_pts], opt_lam0)
                w += np.random.normal(0, np.sqrt(C*base_variances[dim_ind]))
                release.append(w)
            y_pred_closed = ridge_pred(X_test, release) + y_mean
            priv_obl_mses.append(mean_squared_error(y_pred_closed, y_test))
        print(f'C={C}, priv oblivious mse: ', np.mean(priv_obl_mses))


        variances = {}
        priv_aware_lam = [(C+1) * opt_lams[dim_ind] for dim_ind in range(len(opt_lams))]
        all_lams[C] = priv_aware_lam
        for dim_ind in range(d):
            ws = []
            for subset_ind in range(NUM_SUBSETS):
                pts = subsets[subset_ind]
                X_subset, Y_subset = X_train[pts], y_train_c[pts]
                ws.append(ridge_1d(X_subset[:, dim_ind], Y_subset, priv_aware_lam[dim_ind]))
            variances[dim_ind] = np.var(ws)
        pickle.dump(variances, open(f'data/C={C}_{data}_variances.pkl', 'wb'))

        priv_aware_mses = []
        for trial in range(NUM_TRIALS):
            print(trial)
            chosen_pts = subsets[np.random.choice(range(len(subsets)))]
            unnoised_ws = []
            release = []
            for dim_ind in range(d):
                w = ridge_1d(
                    X_train[chosen_pts][:, dim_ind],
                    y_train_c[chosen_pts], priv_aware_lam[dim_ind])
                w += np.random.normal(0, np.sqrt(C*variances[dim_ind]))
                release.append(w)
            y_pred_closed = ridge_pred(X_test, release) + y_mean
            priv_aware_mses.append(mean_squared_error(y_pred_closed, y_test))
        print(f'C={C}, priv aware mse: ', np.mean(priv_aware_mses))
        all_mses[C] = (priv_obl_mses, priv_aware_mses)

        pickle.dump(all_mses, open(f'data/C={C}_{data}_mses.pkl', 'wb'))
        pickle.dump(all_lams, open(f'data/C={C}_{data}_lams.pkl', 'wb'))
