import os
import pandas as pd
import torch
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import re
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import StandardScaler



def load_dataset(dataset_name):
    pwd = os.path.dirname(__file__) # current directory of this file data.py
    data_dir = os.path.join(pwd, "data")
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)

        X_train = train_dataset.data.float().view(-1, 28*28) / 255.0
        non_zero_variance_dimensions = torch.std(X_train, dim=0) > 0
        X_train = X_train[:, non_zero_variance_dimensions]
        y_train = train_dataset.targets.view(-1, 1)

        X_test = test_dataset.data.float().view(-1, 28*28) / 255.0
        X_test = X_test[:, non_zero_variance_dimensions]
        y_test = test_dataset.targets.view(-1, 1)

        num_classes = 10

    elif re.match(r"^mnist_(\d+)_vs_(\d+)$", dataset_name):
        # it will be mnist_0_vs_1 etc
        class_a = dataset_name.split("_")[1]
        class_b = dataset_name.split("_")[3]
        class_a = int(class_a)
        class_b = int(class_b)
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)
        train_mask = (train_dataset.targets == class_a) | (train_dataset.targets == class_b)
        test_mask = (test_dataset.targets == class_a) | (test_dataset.targets == class_b)
        X_train = train_dataset.data[train_mask].float().view(-1, 28*28) / 255.0
        non_zero_variance_dimensions = torch.std(X_train, dim=0) > 0
        X_train = X_train[:, non_zero_variance_dimensions]
        y_train = train_dataset.targets[train_mask]
        y_train = (y_train == class_b).float().view(-1, 1)
        X_test = test_dataset.data[test_mask].float().view(-1, 28*28) / 255.0
        X_test = X_test[:, non_zero_variance_dimensions]
        y_test = test_dataset.targets[test_mask]
        y_test = (y_test == class_b).float().view(-1, 1)
        num_classes = 2
    
    elif dataset_name == "wine_quality":
        red_url = "winequality-red.csv"
        white_url = "winequality-white.csv"
        df_red = pd.read_csv(red_url, sep=';')
        df_white = pd.read_csv(white_url, sep=';')
        df = pd.concat([df_red, df_white], ignore_index=True)
        X = df.drop("quality", axis=1).values
        X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        y = df["quality"].values
        y = (y > 5).astype(float)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        non_zero_variance_dimensions = torch.std(X_train, dim=0) > 0
        X_train = X_train[:, non_zero_variance_dimensions]
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = X_test[:, non_zero_variance_dimensions]
        y_train = torch.tensor(y_train).view(-1, 1)
        y_test = torch.tensor(y_test).view(-1, 1)
        X_mean = X_train.mean(0, keepdim=True)
        X_train = X_train - X_mean
        X_test = X_test - X_mean
        num_classes = 2
    elif dataset_name == 'bank':
          
        # fetch dataset 
        bank_marketing = pd.read_csv('bank/bank-full.csv', sep=";")
        # data (as pandas dataframes) 
        X = bank_marketing.drop(columns=["y"])   # DataFrame of features (categorical + numeric)
        y = bank_marketing["y"]                  # Series target: "yes"/"no"
        y = (y.astype(str).str.lower() == "yes").astype(int).to_numpy()

        cat_cols = X.select_dtypes(include=["object", "category"]).columns
        num_cols = X.columns.difference(cat_cols)

        ct = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(with_mean=True, with_std=True), list(num_cols)),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)),
            ],
            remainder="drop",
        )
        X = ct.fit_transform(X)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_mean = X_train.mean(0, keepdim=True)
        X_train = X_train - X_mean
        X_test = X_test - X_mean

        # shape should be [num_samples, 1]
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        num_classes = 2
        
    elif dataset_name == "adult":
        cols = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]

        train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        test_url  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

        df_train = pd.read_csv(train_url, header=None, names=cols, na_values=' ?', skipinitialspace=True)
        df_test  = pd.read_csv(test_url, header=None, names=cols, na_values=' ?', skipinitialspace=True, skiprows=1)

        df_test['income'] = df_test['income'].str.replace('.', '', regex=False)
        df = pd.concat([df_train, df_test], ignore_index=True)
        df = df.dropna()

        X = df.drop('income', axis=1)
        y = (df['income'] == '>50K').astype(float)

        num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        cat_features = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']
        
        preprocessor = ColumnTransformer([
            ('num', MinMaxScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test  = preprocessor.transform(X_test_raw)

        # everything as torch tensors
        X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
        non_zero_variance_dimensions = torch.std(X_train, dim=0) > 0
        X_train = X_train[:, non_zero_variance_dimensions]
        X_test  = torch.tensor(X_test.toarray(), dtype=torch.float32)
        X_test  = X_test[:, non_zero_variance_dimensions]
        y_train = torch.tensor(y_train.values).view(-1, 1)
        y_test  = torch.tensor(y_test.values).view(-1, 1)
        X_mean = X_train.mean(0, keepdim=True)
        X_train = X_train - X_mean
        X_test = X_test - X_mean
        num_classes = 2

    return X_train, y_train, X_test, y_test, num_classes
