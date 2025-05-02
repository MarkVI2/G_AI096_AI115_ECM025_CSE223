import numpy as np
from typing import List
from sklearn.base import BaseEstimator, ClassifierMixin
import math

class DecisionTreeRegressorScratch:
    """
    Simple decision tree regressor implementing CART for regression tasks.
    """
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1, n_random_splits: int = None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_random_splits = n_random_splits  # number of random thresholds to try per feature
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1:
            return self.Node(value=y.mean())
        # Find best split
        best_mse = float('inf')
        best_idx, best_thr = None, None
        for feature in range(n_features):
            unique_vals = sorted(set(X[:, feature]))
            # Randomized threshold selection for faster splits
            if self.n_random_splits and len(unique_vals) > self.n_random_splits:
                thresholds = np.random.choice(unique_vals, size=self.n_random_splits, replace=False)
            else:
                thresholds = unique_vals
            for thr in thresholds:
                left_mask = X[:, feature] <= thr
                if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
                    continue
                y_left, y_right = y[left_mask], y[~left_mask]
                mse = (len(y_left) * np.var(y_left) + len(y_right) * np.var(y_right)) / n_samples
                if mse < best_mse:
                    best_mse, best_idx, best_thr = mse, feature, thr
        if best_idx is None:
            return self.Node(value=y.mean())
        # Split
        left_mask = X[:, best_idx] <= best_thr
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return self.Node(feature_index=best_idx, threshold=best_thr, left=left, right=right)

    def predict(self, X):
        return np.array([self._predict_row(x, self.tree) for x in X])

    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

class BasicGBDTClassifier:
    """
    Simplified gradient boosting classifier (binary) implemented from scratch.
    """
    def __init__(self, n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 ccp_alpha: float = 0.0,
                 subsample: float = 1.0,
                 max_features=None,
                 early_stopping_rounds: int = None,
                 tol: float = 1e-4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha  # complexity pruning parameter
        self.subsample = subsample  # fraction of samples per tree
        self.trees: List[DecisionTreeRegressorScratch] = []
        self.features_subsets: List[np.ndarray] = []
        self.init_score: float = 0.0
        self.max_features = max_features
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None):
        # Flatten sequences to 2D feature matrix
        n_samples = X.shape[0]
        if X.ndim > 2:
            X_flat = X.reshape(n_samples, -1)
        else:
            X_flat = X
        n_features = X_flat.shape[1]
        # y should be binary {0,1}
        # Initialize model score as log odds
        p = np.clip(np.mean(y), 1e-5, 1 - 1e-5)
        self.init_score = np.log(p / (1 - p))
        F = np.full(shape=y.shape, fill_value=self.init_score)
        # Setup early stopping
        best_loss = float('inf')
        no_improve_rounds = 0
        # Gradient boosting iterations
        for m in range(self.n_estimators):
            prob = 1 / (1 + np.exp(-F))
            residual = y - prob
            # Optionally subsample rows
            if 0 < self.subsample < 1.0:
                idx = np.random.choice(n_samples, int(self.subsample * n_samples), replace=False)
                X_train_full, y_train = X_flat[idx], residual[idx]
            else:
                X_train_full, y_train = X_flat, residual
            # Feature bagging if requested
            if self.max_features:
                if isinstance(self.max_features, float) and self.max_features < 1:
                    kf = int(self.max_features * n_features)
                else:
                    kf = int(self.max_features)
                feat_idx = np.random.choice(n_features, kf, replace=False)
            else:
                feat_idx = np.arange(n_features)
            X_train = X_train_full[:, feat_idx]
            self.features_subsets.append(feat_idx)
            # Fit regression tree on residuals
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_train, y_train)
            self.trees.append(tree)
            # Update F for all samples
            F += self.learning_rate * tree.predict(X_flat[:, feat_idx])
            # early stopping on validation set
            if X_val is not None and y_val is not None and self.early_stopping_rounds:
                # compute val F and loss
                n_val = X_val.shape[0]
                Xv_flat = X_val.reshape(n_val, -1) if X_val.ndim > 2 else X_val
                Fv = np.full(shape=y_val.shape, fill_value=self.init_score)
                for t, t_feat in zip(self.trees, self.features_subsets):
                    Fv += self.learning_rate * t.predict(Xv_flat[:, t_feat])
                pv = 1 / (1 + np.exp(-Fv))
                loss = -np.mean(y_val * np.log(pv+1e-15) + (1-y_val)*np.log(1-pv+1e-15))
                # check improvement
                if best_loss - loss > self.tol:
                    best_loss = loss
                    no_improve_rounds = 0
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.early_stopping_rounds:
                        break
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Flatten sequences to 2D
        n_samples = X.shape[0]
        if X.ndim > 2:
            X_flat = X.reshape(n_samples, -1)
        else:
            X_flat = X
        # Aggregate predictions
        F = np.full(shape=(n_samples,), fill_value=self.init_score)
        for tree, feat_idx in zip(self.trees, self.features_subsets):
            F += self.learning_rate * tree.predict(X_flat[:, feat_idx])
        prob = 1 / (1 + np.exp(-F))
        # Return 2-column probability for binary classes
        return np.vstack([1 - prob, prob]).T

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

class MultiClassGBDTClassifier(BaseEstimator, ClassifierMixin):
    """
    One-vs-rest multi-class gradient boosting using BasicGBDTClassifier.
    """
    def __init__(self,
                 n_classes: int,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 3,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 ccp_alpha: float = 0.0,
                 subsample: float = 1.0):
        self.n_classes = n_classes
        # Create one binary GBDT per class
        self.models = [
            BasicGBDTClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                subsample=subsample
            ) for _ in range(n_classes)
        ]

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Flatten input
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1) if X.ndim > 2 else X
        # Train one-vs-rest binary classifiers
        for k, model in enumerate(self.models):
            y_bin = (y == k).astype(int)
            model.fit(X_flat, y_bin)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1) if X.ndim > 2 else X
        # Gather probabilities for positive class from each binary model
        probas = [model.predict_proba(X_flat)[:, 1] for model in self.models]
        proba_matrix = np.vstack(probas).T
        # Normalize rows to sum to 1 (softmax-like)
        row_sums = proba_matrix.sum(axis=1, keepdims=True)
        return proba_matrix / np.clip(row_sums, a_min=1e-8, a_max=None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

class LogisticRegressionScratch:
    """
    Multi-class logistic regression implemented from scratch using softmax.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, reg_lambda: float = 0.0):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.reg_lambda = reg_lambda
        self.W = None  # weights
        self.b = None  # biases

    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / expZ.sum(axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X shape: (n_samples, n_features)
        # y shape: (n_samples,), values in {0,...,C-1}
        n_samples, n_features = X.shape
        classes = np.unique(y)
        C = len(classes)
        # Initialize parameters
        self.W = np.zeros((n_features, C))
        self.b = np.zeros((1, C))
        # One-hot encode labels
        Y = np.zeros((n_samples, C))
        Y[np.arange(n_samples), y] = 1
        # Gradient descent
        for _ in range(self.n_iter):
            Z = X.dot(self.W) + self.b  # shape (n_samples, C)
            A = self._softmax(Z)
            # gradients
            dW = (1 / n_samples) * X.T.dot(A - Y) + (self.reg_lambda * self.W)
            db = (1 / n_samples) * np.sum(A - Y, axis=0, keepdims=True)
            # update
            self.W -= self.lr * dW
            self.b -= self.lr * db
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = X.dot(self.W) + self.b
        return self._softmax(Z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
