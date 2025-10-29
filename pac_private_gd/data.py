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
                        handle_unknown="ignore", sparse_output=False)),
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
        pca = PCA(whiten=True, random_state=42) # keep all dimensions but make them uncorrelated
        X_train = torch.tensor(pca.fit_transform(X_train), dtype=torch.float32)
        X_test = torch.tensor(pca.transform(X_test), dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        num_classes = 2

    return X_train, y_train, X_test, y_test, num_classes
