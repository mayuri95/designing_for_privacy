import numpy as np
import torch

import data
from models import LinearModel
import utils
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

def est_L(X, mu):
    """
    Estimate strong convexity μ and smoothness L for logistic loss + ridge.
    Uses sigmoid curvature s_i = σ(z_i)(1-σ(z_i)).
    Args:
        X: [n, d] input
        w: [d, 1] weights; if None, use w=0 (worst-case bound)
        lam: ridge regularization
    Returns:
        (mu, L) estimates
    """
    X = X.numpy()
    n = X.shape[0]
    s = np.full((n,), 0.25)       # worst-case curvature at w=0
    H = (X.T * s) @ X / n
    eigs = np.linalg.eigvalsh(H)
    L  = np.max(eigs) + mu
    return L

def pac_private_gd(X, y, X_test, y_test, num_classes, mu, T, mi_budget, privacy_aware, e0, verbose=True):

    # X, y, X_test, y_test, num_classes = data.load_dataset(dataset_name)
    num_features = X.shape[0]

    model = LinearModel(X.shape[1], num_classes if num_classes > 2 else 1)
    L = est_L(X, mu)
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
        per_sample_grads = utils.get_per_sample_grads(model, loss_fn, X, y, mu).cpu().numpy()

        model_update = np.zeros(d)

        for d_i in range(d):

            def grad_i_fn(): # return a torch scalar
                return per_sample_grads[np.random.rand(num_features) < 0.5, d_i].mean().item()

            grad_i_var = utils.exact_var_1d(per_sample_grads[:, d_i])

            if privacy_aware:
                eta_i = utils.optimal_eta(mu=mu, T=T, C=C, e0=e0[d_i], var=grad_i_var)
            else:
                eta_i = utils.optimal_eta(mu=mu, T=T, C=0, e0=e0[d_i], var=grad_i_var)
            assert eta_i >= 0
            eta_i = np.clip(eta_i, -L, L)
            grad_i = grad_i_fn()

            grad_i +=  np.sqrt(C * grad_i_var) * np.random.randn()

            model_update[d_i] = -eta_i * grad_i
        utils.apply_update_vec(model, model_update)

        with torch.no_grad():
            loss = loss_fn(model(X), y).item()
            cla_loss.append(loss)
            loss += (mu / 2) * utils.get_param_vec(model).norm().item()**2
            train_loss.append(loss)

        if verbose:
            print(L, mu, np.linalg.norm(model_update))
            print(f"Iter {i+1}/{T}, Train Loss: {loss:.4f}")

        del per_sample_grads

    # now that we have trained the model, calculate the test accuracy
    y_pred = model(X_test)
    y_pred_probs = torch.sigmoid(y_pred.view(-1)).detach().numpy()
    auc = roc_auc_score(y_test, y_pred_probs)
    y_pred_labels = (torch.sigmoid(y_pred) >= 0.5).float().view(-1, 1)
    bal_acc = balanced_accuracy_score(y_test, y_pred_labels.numpy())
    test_acc = accuracy_score(y_test, y_pred_labels.numpy())

    return train_loss, cla_loss, (test_acc, bal_acc, auc)
