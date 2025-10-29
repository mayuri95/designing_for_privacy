import numpy as np
import pandas as pd
from numpy.linalg import solve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import Ridge
from typing import Tuple, Optional, Sequence
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
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

def ridge_1d(x, y, lam):
    h_s = sum([x[i]**2 for i in range(len(x))])
    xy = sum([x[i]*y[i] for i in range(len(x))])
    return xy / (h_s + lam)
def ridge_pred(x, w):
    return [sum(
        [x[ind][i]*w[i] for i in range(len(x[ind]))
    ]) for ind in range(len(x))]

def poisson_sample(X):
    num_pts = len(X)
    pts = []
    for i in range(num_pts):
        if np.random.rand() < 0.5:
            pts.append(i)
    return pts
