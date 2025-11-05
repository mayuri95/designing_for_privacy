import numpy as np

def adassp(X, y, opts):
    BX = np.max(np.linalg.norm(X, axis=1))
    BY = np.max(np.abs(y))

    epsilon = opts['eps']
    delta = opts['delta']

    n, d = X.shape
    varrho = 0.05

    # set the eigenvalue limit
    eta = np.sqrt(d * np.log(6 / delta) * np.log(2 * d**2 / varrho)) * BX**2 / (epsilon / 3)

    XTy = X.T @ y
    XTX = X.T @ X + np.eye(d)

    S = np.linalg.svd(XTX, compute_uv=False)
    logsod = np.log(6 / delta)

    lamb_min = S[-1] + np.random.randn() * BX**2 * np.sqrt(logsod) / (epsilon / 3) - logsod / (epsilon / 3)
    lamb_min = max(lamb_min, 0)

    lamb = max(0, eta - lamb_min)

    XTyhat = XTy + (np.sqrt(np.log(6 / delta)) / (epsilon / 3)) * BX * BY * np.random.randn(d)

    # generate symmetric Gaussian noise
    Z = np.random.randn(d, d)
    Z = 0.5 * (Z + Z.T)
    XTXhat = XTX + (np.sqrt(np.log(6 / delta)) / (epsilon / 3)) * BX**2 * Z

    thetahat = np.linalg.solve(XTXhat + lamb * np.eye(d), XTyhat)

    return thetahat
