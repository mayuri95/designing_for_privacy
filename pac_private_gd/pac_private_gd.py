import numpy as np
import torch

import data
from models import LinearModel
import utils
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

NUM_SUBSETS=1024
def est_L_diag(X, mu):
    X = X.numpy()
    n = X.shape[0]
    s = np.full((n,), 0.25)   # worst-case curvature
    H_diag = (s[:, None] * (X ** 2)).mean(axis=0)
    L_diag = H_diag + mu
    return L_diag

def poisson_sample(X):
    num_pts = len(X)
    pts = []
    for i in range(num_pts):
        if np.random.rand() < 0.5:
            pts.append(i)
    return pts

def create_subsets(X_train):
    subsets = {}
    num_dims = X_train.shape[1]
    for num_subsets in range(NUM_SUBSETS):
        pts = poisson_sample(X_train)
        subsets[num_subsets] = pts
    return subsets

def get_variance(p, support):
    _, d = support.shape
    mu = np.sum(support.T * p, axis=1) # d
    centered = support - mu # m x d
    Y = centered * np.sqrt(p)[:, None]
    S = np.sum(Y**2, axis=0)
    S = np.sqrt(S)

    variance = S * (S.sum())
    return variance

def update_p(p, support, noisy_result, noise_lambda):
    diff = support - noisy_result  # Shape (m, d)
    mahalanobis = np.sum((diff ** 2) / noise_lambda, axis=1)
    log_likelihoods = -0.5 * mahalanobis
    log_p = np.log(p + 1e-300) + log_likelihoods
    c = log_p.max()
    log_sum = c + np.log(np.sum(np.exp(log_p - c)))
    p = np.exp(log_p - log_sum)
    return p

def pac_private_gd(X, y, X_test, y_test, num_classes, mu, T, mi_budget, privacy_aware, e0,
                   verbose=True, priv_oblivious_mi_budget = 0.):

    # X, y, X_test, y_test, num_classes = data.load_dataset(dataset_name)
    num_features = X.shape[0]

    model = LinearModel(X.shape[1], num_classes if num_classes > 2 else 1)

    L = est_L_diag(X, mu)

    if num_classes == 2:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    d = sum(p.numel() for p in model.parameters())

    # non-private
    if mi_budget is None:
        C = 0 # note even if we are adding no noise, we still do sampling
    else:
        C = d * T / (2.0 * mi_budget) # num_params * T releases in total

    train_loss = []
    cla_loss = [] # classification loss
    y_pred = model(X_test)
    y_pred_probs = torch.sigmoid(y_pred.view(-1)).detach().numpy()
    y_pred_labels = (torch.sigmoid(y_pred) >= 0.5).float().view(-1, 1)
    test_acc = accuracy_score(y_test, y_pred_labels.numpy())
    print(f'starting acc: {test_acc}')

    subsets = create_subsets(X)
    release_ind = np.random.choice(NUM_SUBSETS)
    p = np.array([1./NUM_SUBSETS for _ in range(NUM_SUBSETS)])
    for i in range(T):
        print(f'iteration {i}')
        per_sample_grads = utils.get_per_sample_grads(model, loss_fn, X, y, mu).cpu().numpy()
        model_update = np.zeros(d)

        for d_i in range(d):

            per_sample_grads_dim_i = per_sample_grads[:, d_i]
            subset_grads = np.atleast_2d(
                np.array([per_sample_grads_dim_i[subsets[i]].mean() for i in sorted(subsets)])).T
            grad_i_var = get_variance(p, subset_grads)
            grad_i_var = np.clip(grad_i_var, 1e-15, None)

            noise_lambda = C * grad_i_var
            if privacy_aware:
                eta_i = utils.optimal_eta(mu=mu, T=T, C=C, e0=e0[d_i], var=grad_i_var)
            else:
                if priv_oblivious_mi_budget != 0:
                    oblivious_C = d * T / (2.0 * priv_oblivious_mi_budget)
                else:
                    oblivious_C = 0
                eta_i = utils.optimal_eta(mu=mu, T=T, C=oblivious_C, e0=e0[d_i], var=grad_i_var)
            eta_i = np.clip(eta_i, -L[d_i], L[d_i])

            grad_i = per_sample_grads_dim_i[subsets[release_ind]].mean()
            grad_i +=  np.sqrt(C * grad_i_var) * np.random.randn()
            model_update[d_i] = -eta_i * grad_i
            p = update_p(p, subset_grads, grad_i, noise_lambda)
        utils.apply_update_vec(model, model_update)

        with torch.no_grad():
            loss = loss_fn(model(X), y).item()
            cla_loss.append(loss)
            loss += (mu / 2) * utils.get_param_vec(model).norm().item()**2
            train_loss.append(loss)

        if verbose:
            # print(L, mu, np.linalg.norm(model_update))
            
            y_pred = model(X_test)
            y_pred_probs = torch.sigmoid(y_pred.view(-1)).detach().numpy()
            y_pred_labels = (torch.sigmoid(y_pred) >= 0.5).float().view(-1, 1)
            test_acc = accuracy_score(y_test, y_pred_labels.numpy())
            print(f"Iter {i+1}/{T}, Train Loss: {loss:.4f}, Test Acc: {test_acc}")
        del per_sample_grads

    # now that we have trained the model, calculate the test accuracy
    y_pred = model(X_test)
    y_pred_probs = torch.sigmoid(y_pred.view(-1)).detach().numpy()
    auc = roc_auc_score(y_test, y_pred_probs)
    y_pred_labels = (torch.sigmoid(y_pred) >= 0.5).float().view(-1, 1)
    bal_acc = balanced_accuracy_score(y_test, y_pred_labels.numpy())
    test_acc = accuracy_score(y_test, y_pred_labels.numpy())

    return train_loss, cla_loss, (test_acc, bal_acc, auc)
