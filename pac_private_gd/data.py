import os
import pandas as pd
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import re
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np, torch, os

def pca_wrapper(X_train, X_test, whiten, tol=1e-6):
    pca = PCA(whiten=False, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    # get the non-zero eigenvalue components
    non_zero_var_indices = pca.explained_variance_ > tol
    X_train_pca = X_train_pca[:, non_zero_var_indices]
    X_test_pca = pca.transform(X_test)
    X_test_pca = X_test_pca[:, non_zero_var_indices]
    if whiten:
        X_train_whitened = X_train_pca / np.sqrt(pca.explained_variance_[non_zero_var_indices])
        X_test_whitened = X_test_pca / np.sqrt(pca.explained_variance_[non_zero_var_indices])
        return X_train_whitened, X_test_whitened
    else:
        return X_train_pca, X_test_pca

def stable_bank_loader(pwd, seed=42):
    # Load data
    df = pd.read_csv(os.path.join(pwd, "bank", "bank-full.csv"), sep=";")
    X = df.drop(columns=["y"])
    y = (df["y"].astype(str).str.lower() == "yes").astype(int).to_numpy()

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    num_cols = X.columns.difference(cat_cols)

    # Freeze category schema (sorted order, stable across versions)
    schema = {}
    for c in cat_cols:
        schema[c] = sorted(X[c].astype("string").fillna("<NA>").unique())

    ohe = OneHotEncoder(
        categories=[schema[c] for c in cat_cols],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float64
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), list(num_cols)),
            ("cat", Pipeline([
                ("onehot", ohe),
                ("scaler", StandardScaler(with_mean=True, with_std=False, copy=True)),
            ]), list(cat_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Deterministic split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Transform with frozen schema
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    # PCA with deterministic solver and dtype
    pca = PCA(whiten=True, random_state=seed, svd_solver="full")
    X_train = pca.fit_transform(X_train.astype(np.float64))
    X_test = pca.transform(X_test.astype(np.float64))

    # Torch tensors (float32 for downstream model)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    num_classes = 2
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    return X_train, y_train, X_test, y_test, num_classes


def load_dataset(dataset_name):
    pwd = os.path.dirname(__file__)  # current directory of this file data.py
    data_dir = os.path.join(pwd, "data")
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True)

        X_train = train_dataset.data.float().view(-1, 28*28) / 255.0
        non_zero_variance_dimensions = torch.std(X_train, dim=0) > 0
        X_train = X_train[:, non_zero_variance_dimensions]
        X_train_mean = X_train.mean(dim=0, keepdim=True)
        X_train_std = X_train.std(dim=0, keepdim=True)
        X_train = (X_train - X_train_mean) / X_train_std
        y_train = train_dataset.targets.view(-1, 1)

        X_test = test_dataset.data.float().view(-1, 28*28) / 255.0
        X_test = X_test[:, non_zero_variance_dimensions]
        X_test = (X_test - X_train_mean) / X_train_std
        y_test = test_dataset.targets.view(-1, 1)

        pca = PCA(random_state=42) # keep all dimensions but make them uncorrelated
        X_train = torch.tensor(pca.fit_transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(pca.transform(X_test), dtype=torch.float32)

        num_classes = 10

    elif re.match(r"^mnist_(\d+)_vs_(\d+)$", dataset_name):
        # it will be mnist_0_vs_1 etc
        class_a = dataset_name.split("_")[1]
        class_b = dataset_name.split("_")[3]
        class_a = int(class_a)
        class_b = int(class_b)
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True)
        train_mask = (train_dataset.targets == class_a) | (
            train_dataset.targets == class_b)
        test_mask = (test_dataset.targets == class_a) | (
            test_dataset.targets == class_b)
        X_train = train_dataset.data[train_mask].float(
        ).view(-1, 28*28) / 255.0
        non_zero_variance_dimensions = torch.std(X_train, dim=0) > 0
        X_train = X_train[:, non_zero_variance_dimensions]
        X_train_mean = X_train.mean(dim=0, keepdim=True)
        X_train_std = X_train.std(dim=0, keepdim=True)
        X_train = (X_train - X_train_mean) / X_train_std
        y_train = train_dataset.targets[train_mask]
        y_train = (y_train == class_b).float().view(-1, 1)
        X_test = test_dataset.data[test_mask].float().view(-1, 28*28) / 255.0
        X_test = X_test[:, non_zero_variance_dimensions]
        X_test = (X_test - X_train_mean) / X_train_std
        y_test = test_dataset.targets[test_mask]
        y_test = (y_test == class_b).float().view(-1, 1)
        pca = PCA(random_state=42) # keep all dimensions but make them uncorrelated
        X_train = torch.tensor(pca.fit_transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(pca.transform(X_test), dtype=torch.float32)
        num_classes = 2

    elif dataset_name == 'bank':
         # fetch dataset
        # bank_marketing = pd.read_csv('bank/bank-full.csv', sep=";")
        bank_marketing = pd.read_csv(os.path.join(pwd, 'bank', 'bank-full.csv'), sep=";")
        # data (as pandas dataframes)
        # DataFrame of features (categorical + numeric)
        X = bank_marketing.drop(columns=["y"])
        y = bank_marketing["y"]                  # Series target: "yes"/"no"
        y = (y.astype(str).str.lower() == "yes").astype(int).to_numpy()

        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        num_cols = X.columns.difference(cat_cols)

        ct = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=True, with_std=True), list(num_cols)),
                ("cat", Pipeline([
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore", sparse_output=False, drop='first')),
                    ("scaler", StandardScaler(with_mean=True, with_std=False)),
                ]),
                    list(cat_cols)),
            ],
            remainder="drop",
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train = ct.fit_transform(X_train)
        X_test = ct.transform(X_test)
        # X_train, X_test = pca_wrapper(X_train, X_test, whiten=True)
        pca = PCA(whiten=True, random_state=42, svd_solver="full")
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        num_classes = 2

    return X_train, y_train, X_test, y_test, num_classes