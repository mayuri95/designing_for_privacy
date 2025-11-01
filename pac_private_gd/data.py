import os
import pandas as pd
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np
from utils import pca

def load_credit():
    df = pd.read_excel('credit_data.xls', header=1)
    target_col = "default payment next month"
    assert target_col in df.columns, f"Target column not found: {target_col}"

    y = df[target_col].to_numpy()
    X = df.drop(columns=["ID"]).copy()

    # ----------------------------
    # Fix codes for categorical
    # ----------------------------
    # Merge unknowns to a valid bucket
    X["EDUCATION"] = X["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    X["MARRIAGE"]  = X["MARRIAGE"].replace({0: 3})

    cat_cols   = ["SEX", "EDUCATION", "MARRIAGE"]
    money_cols = [c for c in X.columns if c.startswith(("BILL_AMT", "PAY_AMT"))]
    num_cols   = [c for c in X.columns if c not in cat_cols]

    # ----------------------------
    # Split BEFORE fitting transforms
    # ----------------------------
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ----------------------------
    # Column-wise transforms
    # ----------------------------
    signed_log = FunctionTransformer(
        lambda A: np.sign(A) * np.log1p(np.abs(A)),
        feature_names_out="one-to-one"
    )
    money_pipe = Pipeline([
        ("slog", signed_log)
    ])


    # OneHotEncoder dense so we can PCA later
    ohe = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False
    )

    # Apply per-group transforms; we’ll standardize + PCA on the concatenated matrix next
    coltf = ColumnTransformer(
        transformers=[
            ("money", money_pipe, money_cols),
            ("num_other", "passthrough", list(set(num_cols) - set(money_cols))),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # ----------------------------
    # Global standardize + PCA whitening
    # ----------------------------
    global_pipe = Pipeline([
        ("cols", coltf),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(whiten=True, random_state=42)),
    ])

    # Fit on train, transform both
    X_train = global_pipe.fit_transform(X_train_df)
    X_test  = global_pipe.transform(X_test_df)

    # Ensure dense np.float64
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test  = np.asarray(X_test, dtype=np.float64)
    num_classes=2

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

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
        X_train_mean = X_train.mean(dim=0, keepdim=True)
        X_train = X_train - X_train_mean
        y_train = train_dataset.targets.view(-1, 1)

        X_test = test_dataset.data.float().view(-1, 28*28) / 255.0
        X_test = X_test - X_train_mean
        y_test = test_dataset.targets.view(-1, 1)

        X_train, X_test = pca(X_train.numpy(), X_test.numpy(), whiten=False)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

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
        X_train_mean = X_train.mean(dim=0, keepdim=True)
        X_train = X_train - X_train_mean
        y_train = train_dataset.targets[train_mask]
        y_train = (y_train == class_b).float().view(-1, 1)
        X_test = test_dataset.data[test_mask].float().view(-1, 28*28) / 255.0
        X_test = X_test - X_train_mean
        y_test = test_dataset.targets[test_mask]
        y_test = (y_test == class_b).float().view(-1, 1)

        X_train, X_test = pca(X_train.numpy(), X_test.numpy(), whiten=False)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        num_classes = 2

    elif dataset_name == 'credit':
        return load_credit()

    return X_train, y_train, X_test, y_test, num_classes
