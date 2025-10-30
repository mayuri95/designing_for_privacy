import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.special import gammaln
import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import lambertw
from scipy.optimize import minimize
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def exact_var_1d(c, p=0.5):
    """
    Exact variance of
        Y = (sum_i m_i * c_i) / (sum_i m_i),
    where m_i ~ Bernoulli(0.5) i.i.d., and Y=0 when sum_i m_i = 0.

    Uses log-sum-exp vectorization for numerical stability and speed.
    """
    c = np.asarray(c, dtype=float)
    n = c.size
    a = c.sum()
    b = np.sum(c**2)
    s2 = (b - a*a/n) / (n - 1)  # unbiased population variance

    k = np.arange(1, n+1)
    # log P(K=k) for Bin(n,p), restricted to k>=1
    log_pk = (gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
              + k*np.log(p) + (n-k)*np.log1p(-p))
    # normalize over k>=1
    m = np.max(log_pk)
    pk = np.exp(log_pk - m)
    pk /= pk.sum()

    E_invK = np.sum(pk / k)
    return s2 * (E_invK - 1.0/n)

    # return var_conditional + var_mean

def logistic_regression_dataset_synthesis(N, d):
    w = np.random.randn(d)
    X = np.random.randn(N, d)
    logits = X @ w
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(N) < probs).astype(np.float32)
    return X, y, w

def optimal_eta(mu, T, C, e0, var):
    if e0 == 0:
        return 0.0
    
    if var == 0:
        # minimize (1-eta * mu)^T * e0, this is achieved at eta = 1/mu
        return 1/mu

    alpha = mu * T
    beta = (1+C) * var /  mu
    a = alpha * e0**2 / beta
    
    if a>500:
        print("Using approximation for large a")
        # eta = 1/alpha * (np.log(a) - np.log(a)/a) # very good approximation when a is large
        eta = 1/alpha * (np.log1p(a) - np.log1p(a)/(a+1)) # very good approximation when a is large
    else:
        eta = 1/alpha * (a+1 - lambertw(np.exp(a+1)).real)
        
    return eta

def pac_private_logistic_regression(X, y, mu, mi_budget, T=10, privacy_aware=True):

    def objective(w):
        N = X.shape[0]
        z = X @ w
        y_hat = sigmoid(z)
        loss = -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log1p(- y_hat))
        reg_term = (mu / 2) * np.sum(w ** 2)
        return loss + reg_term
    
    e0 = minimize(objective, np.zeros(X.shape[1]), method='L-BFGS-B', tol=1e-10).x

    N, d = X.shape
    w = np.zeros(d)
    losses = []
    if mi_budget == 0:
        C = 0
    else:
        C = d * T / 2.0 / mi_budget
    # C = 0

    for _ in range(T):
        z = X @ w
        y_hat = sigmoid(z)

        # --- Per-sample gradients ---
        per_sample_grads = (y_hat - y)[:, np.newaxis] * X  # shape (N, d)

        for d_i in range(d):
            
            grad_i_var = exact_var_1d(per_sample_grads[:, d_i])
            # grad_i_var = np.var(np.array([per_sample_grads[np.random.rand(N) < 0.5, d_i].mean() for _ in range(128)])) # scalar
            grad = per_sample_grads[np.random.rand(N) < 0.5, d_i].mean() + mu * w[d_i] + np.sqrt(C * grad_i_var) * np.random.randn()

            lr = optimal_eta(mu=mu, T=T, C=C if privacy_aware else 0, e0=e0[d_i], var=grad_i_var)
            # lr = 0.1

            w[d_i] = w[d_i] - lr * grad
        
        z = X @ w
        y_hat = sigmoid(z)
        loss = -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log1p(- y_hat))

        reg_term = (mu / 2) * np.sum(w ** 2)
        losses.append(loss + reg_term)
        print(f'Train loss: {losses[-1]:.4f}')

    return w, losses

def evaluate_model(X, y, w):
    z = X @ w
    y_hat = sigmoid(z)
    y_pred = (y_hat > 0.5).astype(int)
    accuracy = np.mean(y_pred == y)
    return accuracy

def model_loss(X, y, w, mu):
    N = X.shape[0]
    z = X @ w
    y_hat = sigmoid(z)
    loss = -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log1p(- y_hat))
    reg_term = (mu / 2) * np.sum(w ** 2)
    return loss + reg_term

# Load dataset
data = pd.read_csv("./bank/bank-full.csv", sep=';')

# Separate features and target
X = data.drop('y', axis=1)
y = data['y'].map({'yes': 1, 'no': 0})  # Binary target
y = y.values.ravel().astype(np.float64)

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # avoid dummy trap
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ]
)

# Create full pipeline with PCA for feature independence
# Adjust n_components as needed (e.g., 0.95 to keep 95% variance)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', PCA(whiten=False, n_components=0.99)),  # whiten makes components uncorrelated
])

# Apply transformations
X_transformed = pipeline.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42, fit_intercept=False)
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
w, losses = pac_private_logistic_regression(X_train, y_train, mu=0.0001, mi_budget=0, T=10)

print(evaluate_model(X_test, y_test, w))
