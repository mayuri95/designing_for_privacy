import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_california_housing
import json

def adassp(X, y, opts):
    BX = np.max(np.linalg.norm(X, axis=1))
    BY = np.max(np.abs(y))

    epsilon = opts['eps']
    delta = opts['delta']

    n, d = X.shape
    varrho = 0.05

    # set the eigenvalue limit
    eta = np.sqrt(d * np.log(4 / delta) * np.log(2 * d**2 / varrho)) * BX**2 / (epsilon / 2)

    XTy = X.T @ y
    XTX = X.T @ X + np.eye(d)

    S = np.linalg.svd(XTX, compute_uv=False)

    lamb_min = S[-1] # + np.random.randn() * BX**2 * np.sqrt(logsod) / (epsilon / 3) - logsod / (epsilon / 3)
    lamb_min = max(lamb_min, 0)

    lamb = max(0, eta - lamb_min)

    XTyhat = XTy + (np.sqrt(np.log(4 / delta)) / (epsilon / 2)) * BX * BY * np.random.randn(d)

    # generate symmetric Gaussian noise
    Z = np.random.randn(d, d)
    Z = 0.5 * (Z + Z.T)
    XTXhat = XTX + (np.sqrt(np.log(4 / delta)) / (epsilon / 2)) * BX**2 * Z

    thetahat = np.linalg.solve(XTXhat + lamb * np.eye(d), XTyhat)

    return thetahat

num_trials = 1000
res_dict = {}
# res_dict[dataset][eps] = mean mse
for dataset in ['wine_quality_red', 'wine_quality_white', 'cali_housing']:
    res_dict[dataset] = {}
    def load_wine_quality(red=True):
        if red:
            url = "winequality-red.csv"
        else:
            url = "winequality-white.csv"
        df= pd.read_csv(url, sep=';')
        X = df.drop("quality", axis=1).to_numpy(dtype=float)
        y = df["quality"].to_numpy(dtype=float)

        return X, y

    def load_cali_housing():
        return fetch_california_housing(return_X_y=True)

    if dataset == 'wine_quality_red':
        X, y = load_wine_quality(red=True)
    elif dataset == 'wine_quality_white':
        X, y = load_wine_quality(red=False)
    else:
        X, y = load_cali_housing()

    # Train/test split and standardize X (fit on train)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    pca = PCA(whiten=True, random_state=42)  # orthonormal columns, unit variance
    X_train = pca.fit_transform(X_train)          # (n, r) where r = rank
    X_test  = pca.transform(X_test)

    y_mean = y_train.mean(axis=0)
    y_train = y_train - y_mean
    y_test  = y_test - y_mean
    for eps in [1.6426117097961406,0.7304317044395013,0.3563228120191924,0.177101450509287,0.08841755656932042]:
        mses = []
        for i in range(num_trials):
            w = adassp(X_train, y_train, {'eps': eps, 'delta': 1e-5})
            mse = w @ X_test.T - y_test
            mse = (mse ** 2).mean()
            mses.append(mse)
        
        res_dict[dataset][eps] = (np.mean(mses), np.std(mses))
    
with open('dp_adassp_results.json', 'w') as f:
    json.dump(res_dict, f)
    