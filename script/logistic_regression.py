import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# logistic regression from scratch using only numpy
# no sklearn inside the class, gradient descent only
class LogisticRegressionScratch:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []  # store loss each iter so we can plot later

    def _sigmoid(self, z):
        # converts raw score to probability between 0 and 1
        # clip to avoid overflow when exp gets huge
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_hat):
        # binary cross entropy
        # clip y_hat to prevent log(0) which gives -inf
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        N = len(y)
        return -(1/N) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def fit(self, X, y):
        N, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []

        for _ in range(self.n_iterations):
            # forward pass - compute prediction
            z = X @ self.weights + self.bias
            y_hat = self._sigmoid(z)

            self.losses.append(self._compute_loss(y, y_hat))

            # gradients via chain rule
            # dL/dw = (1/N) * X^T * (y_hat - y)
            # dL/db = (1/N) * sum(y_hat - y)
            error = y_hat - y
            dw = (1/N) * (X.T @ error)
            db = (1/N) * np.sum(error)

            # update step
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# --- quick test on make_moons to verify implmentation ---

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale - LR needs this to converge properly
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

our_model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000)
our_model.fit(X_train_sc, y_train)

# compare against sklearn as sanity check
sk_model = LogisticRegression(max_iter=1000)
sk_model.fit(X_train_sc, y_train)

our_acc = accuracy_score(y_test, our_model.predict(X_test_sc))
sk_acc = accuracy_score(y_test, sk_model.predict(X_test_sc))

print(f"Our model accuracy:  {our_acc:.4f}")
print(f"Sklearn accuracy:    {sk_acc:.4f}")
print(f"Difference:          {abs(our_acc - sk_acc):.4f}")

# sanity check - loss should go down
print(f"\nLoss at iter 0:    {our_model.losses[0]:.4f}")
print(f"Loss at iter 500:  {our_model.losses[499]:.4f}")
print(f"Loss at iter 999:  {our_model.losses[999]:.4f}")

plt.figure(figsize=(8, 4))
plt.plot(our_model.losses)
plt.xlabel("Iteration")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Training Loss Curve")
plt.tight_layout()
plt.savefig("lr_loss_curve.png", dpi=150)
plt.close()
print("\nSaved lr_loss_curve.png")
