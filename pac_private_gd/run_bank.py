import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.special import expit as sigmoid, lambertw
from sklearn.linear_model import LogisticRegression


# -------------------------------------------------
# Load & preprocess UCI Bank Marketing
# -------------------------------------------------
def load_bank():
    url = "bank/bank-full.csv"
    df = pd.read_csv(url, sep=";")
    y = (df["y"].str.lower() == "yes").astype(int).to_numpy()
    X = df.drop(columns=["y"])
    num = X.select_dtypes(exclude="object").columns
    cat = X.select_dtypes(include="object").columns
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              sparse_output=False,
                              drop="first"), cat)
    ])
    Xp = pre.fit_transform(X)
    return Xp, y


def estimate_mu(X, w):
    """Estimate strong-convexity μ for logistic loss at current w."""
    p = sigmoid(X @ w)
    s = p * (1 - p)                     # curvature weights per sample
    mu = np.mean(np.sum((X**2) * s[:, None], axis=1)) / X.shape[1]
    return mu

# -------------------------------------------------
# Lambert-W optimal eta
# -------------------------------------------------
def optimal_eta_lambert(e2, mu, sigma2, T=32.0, clip=None):
    if e2 <= 0 or mu <= 0 or sigma2 <= 0:
        return 0.0
    α = mu * T
    β = sigma2 / mu
    z = (α * e2) / β + 1.0
    if z > 40:
        lnz = np.log(z)
        eta = (lnz - lnz / z) / α        # stable large-z eval
    else:
        eta = (z - lambertw(np.exp(z)).real) / α
    if clip is not None:
        eta = min(eta, clip)
    return float(max(0.0, eta))

def grad_per_example(w, X, y):
    # per-example gradient matrix g_i = x_i * (σ(wᵀx_i) - y_i)
    p = sigmoid(X @ w)
    return X * (p - y)[:, None]            # shape (n, d)

def poisson_half_grad_per_dim(w, X, y, p=0.5):
    """
    Draw independent Bernoulli(p) masks for each coordinate,
    return the average gradient per coordinate (with 1/M_j correction),
    and record how many samples were used per coordinate.
    """
    n, d = X.shape
    g_i = grad_per_example(w, X, y)         # (n, d)
    m = np.random.binomial(1, p, size=(n, d))     # independent masks
    M = m.sum(axis=0)
    # avoid division by 0 → skip or use 1 for empty coords
    M = np.maximum(M, 1)
    g_hat = (m * g_i).sum(axis=0) / M       # per-dimension mean gradient
    return g_hat, M


def minibatch_var_poisson_half_per_dim(w, X, y, p=0.5, num_samples=5000, rng=None):
    """Monte-Carlo variance of Poisson(½) mean gradient per coordinate (independent subsets)."""
    n, d = X.shape
    g_i = grad_per_example(w, X, y)
    gbar = g_i.mean(axis=0)
    per_example_var = np.mean((g_i - gbar)**2, axis=0)
    M = np.random.binomial(n, p, size=(num_samples, d))
    M[M == 0] = 1
    Cplus1 = np.mean((1/M) * (1 - M/n), axis=0)
    sigma2_vec = Cplus1 * per_example_var
    sigma2_scalar = sigma2_vec.mean()
    return sigma2_scalar, sigma2_vec


# -------------------------------------------------
# Logistic GD with “aware” Lambert-W eta
# -------------------------------------------------
def run_bank_logistic_aware(Xtr, ytr, Xte, yte,
                            steps=32, T=32,
                            aware=True, k_noise=0.0, seed=0):
    # rng = np.random.default_rng(seed)
    n, d = Xtr.shape
    w = np.zeros(d)
    clf = LogisticRegression(max_iter=2000, penalty=None, fit_intercept=False)
    clf.fit(Xtr, ytr)
    w_star = clf.coef_.ravel()

    # optional: data-driven μ at w_star
    def estimate_mu(X, w):
        p = 1/(1+np.exp(-(X @ w)))
        s = p*(1-p)
        return np.mean(np.sum((X**2) * s[:,None], axis=1)) / X.shape[1]

    mu = estimate_mu(Xtr, w_star)   # or keep your chosen μ

    e2 = np.zeros(steps + 1)
    acc = np.zeros(steps + 1)
    pred = np.zeros(steps)
    etas = np.zeros(steps)
    e2[0] = np.sum((w - w_star)**2)

    e0_sq = np.sum((w - w_star)**2)
    # print(mu)
    for t in range(steps):
        if t == 0:
            e2_prev = e2[0]

        # true logistic gradient
        grad, M = poisson_half_grad_per_dim(w, Xtr, ytr, p=0.5)


        # empirical variance from Poisson(½) subsets
        sigma2_emp, _ = minibatch_var_poisson_half_per_dim(w, Xtr, ytr, p=0.5)

        # total variance of the noisy gradient estimator (same for both)
        sigma2_inj = (1 + k_noise) * sigma2_emp

        # variance that each mode *assumes* for η
        sigma2_for_eta = sigma2_inj if aware else sigma2_emp

        e2_t = np.sum((w - w_star) ** 2)
        # mu = estimate_mu(Xtr, w)
        eta = optimal_eta_lambert(e2_t, mu, sigma2_for_eta, T=T, clip=None)
        if t < 1 and eta == 0:
            eta = 1e-12  # tiny floor to avoid freezing

        # add Gaussian gradient noise with variance σ²_inj
        noise = np.random.normal(0.0, np.sqrt(sigma2_inj), size=d)

        # update
        w -= eta * (grad + noise)

        # predictor (uses variance used in η calculation)
        pred[t] = (1 - eta * mu)**2 * e2_t + (eta**2) * sigma2_for_eta * d
        etas[t] = eta
        e2_prev = e2_t

        # test accuracy
        yhat = (sigmoid(Xte @ w) > 0.5).astype(int)
        acc[t + 1] = np.mean(yhat == yte)
        e2[t + 1] = np.sum((w - w_star) ** 2)
            # after computing grad, sigma2_emp, sigma2_inj, eta (aware) and eta_un (recompute with sigma2_emp)
        # print(f"mu={mu:.3e}")
        # print(f"σ²_emp={sigma2_emp:.3e}, σ²_inj={sigma2_inj:.3e}, σ²_for_eta={sigma2_for_eta:.3e}")
        # print(f"eta={eta:.5f}, e2_t={e2_t:.4f}, pred_next={pred[t]:.4f}")




    return {"e2": e2, "acc": acc, "pred": pred, "etas": etas}

# -------------------------------------------------
# End-to-end run
# -------------------------------------------------
if __name__ == "__main__":
    X, y = load_bank()
    Xtr, Xte, ytr, yte = train_test_split(X, y,
                                          test_size=0.3,
                                          random_state=42,
                                          stratify=y)
    for aware in [False, True]:
        for C in [8, 32, 128, 512]:
            accs = []
            for t in range(10):
                log = run_bank_logistic_aware(Xtr, ytr, Xte, yte,
                                              aware=aware, k_noise=C, seed = 313*t)
                gap = np.mean(np.abs(log["e2"][1:] - log["pred"]))
                acc = log['acc'][-1]
                accs.append(log['acc'][-1])
                print(acc)
            print(C, aware, np.average(accs))

