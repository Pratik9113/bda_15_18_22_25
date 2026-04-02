import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

X, y = datasets.make_blobs(n_samples=200, centers=2, random_state=42)
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset shape:", X.shape)

scratch_model = SVM()
scratch_model.fit(X_train, y_train)

scratch_preds = scratch_model.predict(X_test)
scratch_accuracy = np.mean(scratch_preds == y_test)

sklearn_model = SVC(kernel='linear')
sklearn_model.fit(X_train, y_train)

sklearn_preds = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_preds)

print("\n===== RESULTS =====")
print("Scratch SVM Accuracy:", scratch_accuracy)
print("Sklearn SVM Accuracy:", sklearn_accuracy)

print("\nSample Predictions (First 10):")
print("Actual   :", y_test[:10])
print("Scratch  :", scratch_preds[:10])
print("Sklearn  :", sklearn_preds[:10])