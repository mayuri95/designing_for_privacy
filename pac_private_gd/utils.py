import torch
from torch.func import functional_call, vmap, grad
import numpy as np
from scipy.special import lambertw, gammaln
import data
from models import LinearModel

def get_param_vec(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    return torch.cat(params)  # shape (num_params,)

def get_per_sample_grads(model, loss_fn, X, y, l2_lambda=0.0):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def compute_loss(params, buffers, sample, target):
        preds = functional_call(model, (params, buffers), (sample.unsqueeze(0),))
        loss = loss_fn(preds, target.unsqueeze(0))
        return loss

    ft_compute_grad = grad(compute_loss)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    per_sample_grads_struct = ft_compute_sample_grad(params, buffers, X, y)

    grads_per_sample = []
    for name, g_list in per_sample_grads_struct.items():
        grads_per_sample.append(g_list.reshape(g_list.shape[0], -1))
    
    grads_flat = torch.cat(grads_per_sample, dim=1)

    if l2_lambda > 0:
        with torch.no_grad():
            w_vec = torch.cat([p.reshape(-1) for p in model.parameters()])
            reg_grad = l2_lambda * w_vec
        grads_flat = grads_flat + reg_grad.unsqueeze(0)

    return grads_flat

def set_param_vec(model, param_vector):
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = param_vector[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

def est_var_1d(random_func, max_iter=1000, tolerance=1e-5):
    samples = []
    prev_var = None

    for i in range(1, max_iter + 1):
        samples.append(random_func())
        if i < 2:
            continue  # need at least two samples for variance

        current_var = np.var(samples, ddof=1)
        if prev_var is not None:
            if abs(current_var - prev_var) < tolerance:
                return current_var
        prev_var = current_var

    print(f"Warning: Variance estimation did not converge within max_iter={max_iter}")
    return prev_var

def exact_var_1d(c):
    """
    Exact variance of
        Y = (sum_i m_i * c_i) / (sum_i m_i),
    where m_i ~ Bernoulli(0.5) i.i.d., and Y=0 when sum_i m_i = 0.

    Uses log-sum-exp vectorization for numerical stability and speed.
    """
    c = np.asarray(c, dtype=float)
    n = c.size
    a = np.sum(c)
    b = np.sum(c**2)

    # ----- compute E_invK = E[1/K * 1_{K≥1}] -----
    k = np.arange(1, n + 1)
    # log(C(n,k)) = log(n!) - log(k!) - log((n-k)!)
    log_comb = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    log_terms = log_comb - np.log(k) - n * np.log(2)

    # log-sum-exp for numerical stability
    max_log = np.max(log_terms)
    E_invK = np.exp(max_log) * np.sum(np.exp(log_terms - max_log))

    # probability K ≥ 1
    p_nonzero = 1 - 2 ** (-n)

    # ----- compute variance -----
    term1 = (b * n - a**2) / (n**2 * (n - 1))
    var_conditional = term1 * (n * E_invK - p_nonzero)
    var_mean = (a / n)**2 * p_nonzero * (1 - p_nonzero)

    return var_conditional + var_mean

def apply_update_vec(model, update_vector):
    # new weights = old weights + update_vector
    with torch.no_grad():
        update_vector = torch.tensor(update_vector, dtype=torch.float32)
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            # Extract the corresponding slice of the update vector
            update_slice = update_vector[offset : offset + numel].view_as(param)
            # Apply the update (add, since update already includes sign)
            param.add_(update_slice)
            offset += numel

def optimal_eta(mu, T, C, e0, var):
    if e0 == 0:
        return 0.0
    
    if var == 0:
        # minimize (1-eta * mu)^T * e0, this is achieved at eta = 1/mu
        return 1/mu

    alpha = mu * T
    beta = (1+C) * var /  mu
    a = alpha * e0**2 / beta + 1
    
    if a>500:
        eta = 1/alpha * (np.log(a) - np.log(a)/a) # very good approximation when a is large
    else:
        eta = 1/alpha * (a - lambertw(np.exp(a)).real)
        
    return eta
    
def find_e0(dataset_name, mu):
    X, y, _, _, num_classes = data.load_dataset(dataset_name)
    model = LinearModel(X.shape[1], num_classes if num_classes > 2 else 1) # logistic/softmax regression
    if num_classes == 2:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=mu)
    for t in range(1000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

    optimal_w = get_param_vec(model).detach().numpy()

    return optimal_w