import numpy as np
import torch

import data
from models import LinearModel
import utils

def pac_private_gd(dataset_name, mu, T, mi_budget, privacy_aware, e0, verbose=True):

    X, y, X_test, y_test, num_classes = data.load_dataset(dataset_name)
    num_features = X.shape[0]

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
        print(C)

    train_loss = []

    for i in range(T):
        per_sample_grads = utils.get_per_sample_grads(model, loss_fn, X, y, mu).cpu().numpy()

        model_update = np.zeros(d)
    
        for d_i in range(d):
            
            def grad_i_fn(): # return a torch scalar
                return per_sample_grads[np.random.rand(num_features) < 0.5, d_i].mean().item()
            
            grad_i_var = utils.exact_var_1d(per_sample_grads[:, d_i])
            # grad_i_var = utils.est_var_1d(grad_i_fn)

            if privacy_aware:
                eta_i = utils.optimal_eta(mu=mu, T=T, C=C, e0=e0[d_i], var=grad_i_var)
            else:
                eta_i = utils.optimal_eta(mu=mu, T=T, C=0, e0=e0[d_i], var=grad_i_var)

            grad_i = grad_i_fn()

            grad_i +=  np.sqrt(C * grad_i_var) * np.random.randn()

            model_update[d_i] = -eta_i * grad_i

        utils.apply_update_vec(model, model_update)

        with torch.no_grad():
            loss = loss_fn(model(X), y).item()
            loss += (mu / 2) * utils.get_param_vec(model).norm().item()**2
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

    return train_loss, test_acc
