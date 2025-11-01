import numpy as np
import torch

import data
from models import LinearModel
import utils

# global_rng.py
import numpy as np, torch


def make_rng(seed=42):
    rng = np.random.Generator(np.random.PCG64(seed))
    torch.manual_seed(seed)
    return rng

@torch.no_grad()
def estimate_mu(X, w):
    """
    Return local strong-convexity (mu) and smoothness (L) for logistic loss at w:
        mu = lambda_min( (1/n) X^T S X ) + reg_lambda
        L   = lambda_max( (1/n) X^T S X ) + reg_lambda
    where S = diag(p*(1-p)), p = sigmoid(X w).

    Args:
        X: [n, d] float tensor
        w: [d] or [d,1] float tensor
        reg_lambda: L2 regularization coefficient (>=0)

    Returns:
        mu, L  (floats)
    """
    if w.ndim == 2 and w.shape[1] == 1:
        w = w.squeeze(1)
    z = X @ w
    p = torch.sigmoid(z)
    s = (p * (1 - p))           # [n]
    # form H = (1/n) X^T S X  by scaling rows of X by sqrt(s)
    Xs = X * s.sqrt().unsqueeze(1)  # [n, d]
    H = (Xs.t() @ Xs) / X.shape[0]  # [d, d], symmetric PSD
    # Use symmetric eigvals for numerical stability
    eigvals = torch.linalg.eigvalsh(H)
    mu = float(eigvals[0].item())
    return mu


def pac_private_gd(X, y, X_test, y_test, num_classes, T, mi_budget, privacy_aware, e0, verbose=True, seed=1):
    rng = make_rng(seed=seed)

    # X, y, X_test, y_test, num_classes = data.load_dataset(dataset_name)
    num_features = X.shape[0]

    mu = estimate_mu(X, torch.zeros(X.shape[1], 1))

    model = LinearModel(X.shape[1], num_classes if num_classes > 2 else 1)
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

    for i in range(T):
        model_update = np.zeros(d)

        per_sample_grads = utils.get_per_sample_grads(model, loss_fn, X, y).cpu().numpy()

    
        for d_i in range(d):
            
            def grad_i_fn(): # return a torch scalar
                return per_sample_grads[rng.random(num_features) < 0.5, d_i].mean().item()
            
            grad_i_var = utils.exact_var_1d(per_sample_grads[:, d_i])
            # grad_i_var = utils.est_var_1d(grad_i_fn)
            if privacy_aware:
                eta_i = utils.optimal_eta(mu=mu, T=T, C=C, e0=e0[d_i], var=grad_i_var)
            else:
                eta_i = utils.optimal_eta(mu=mu, T=T, C=0, e0=e0[d_i], var=grad_i_var)

            grad_i = grad_i_fn()

            grad_i +=  np.sqrt(C * grad_i_var) * rng.standard_normal()

            model_update[d_i] = -eta_i * grad_i

        utils.apply_update_vec(model, model_update)

        with torch.no_grad():
            loss = loss_fn(model(X), y).item()
            cla_loss.append(loss)
            train_loss.append(loss)

        if verbose:
            print(f"Iter {i+1}/{T}, Train Loss: {loss:.4f}")

        del per_sample_grads

    # now that we have trained the model, calculate the test accuracy
    y_pred = model(X_test)
    if num_classes == 2:
        y_pred_labels = (torch.sigmoid(y_pred) >= 0.5).float().view(-1, 1)
    else:
        y_pred_labels = torch.argmax(y_pred, dim=1).view(-1, 1)
    test_acc = (y_pred_labels == y_test).float().mean().item()

    return train_loss, cla_loss, test_acc

