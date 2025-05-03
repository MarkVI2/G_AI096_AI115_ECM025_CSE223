import numpy as np
from typing import List, Optional, Dict
from joblib import Parallel, delayed

class DecisionTreeClassifierScratch:
    """CART decision tree classifier based on Gini impurity."""
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)
        return self
    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / m
        return 1 - np.sum(probs ** 2)
    def _best_split(self, X, y):
        m, n = X.shape
        best_impurity = float('inf')
        best_idx, best_thr = None, None
        parent_imp = self._gini(y)
        for idx in range(n):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left_mask = X[:, idx] <= thr
                if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
                    continue
                y_left, y_right = y[left_mask], y[~left_mask]
                imp = (len(y_left)*self._gini(y_left) + len(y_right)*self._gini(y_right)) / m
                if imp < best_impurity:
                    best_impurity, best_idx, best_thr = imp, idx, thr
        return best_idx, best_thr
    def _build_tree(self, X, y, depth):
        m = len(y)
        if depth >= self.max_depth or m < self.min_samples_split or len(np.unique(y)) == 1:
            # leaf predicts majority class
            vals, counts = np.unique(y, return_counts=True)
            return self.Node(value=vals[np.argmax(counts)])
        idx, thr = self._best_split(X, y)
        if idx is None:
            vals, counts = np.unique(y, return_counts=True)
            return self.Node(value=vals[np.argmax(counts)])
        left_mask = X[:, idx] <= thr
        left = self._build_tree(X[left_mask], y[left_mask], depth+1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth+1)
        return self.Node(feature_index=idx, threshold=thr, left=left, right=right)
    def _predict_row(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_row(x, self.tree) for x in X])
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # estimate proba by subtree majority frequencies
        # not exact but simple: for each leaf count class distribution
        def proba_row(x):
            node = self.tree
            while node.value is None:
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # one-hot
            probs = np.zeros(self.n_classes)
            probs[node.value] = 1
            return probs
        return np.vstack([proba_row(x) for x in X])

class RandomForestScratch:
    """Ensemble of DecisionTreeClassifierScratch via bootstrap and majority vote."""
    def __init__(self, n_estimators: int = 10, max_depth: int = 3, min_samples_split: int = 2, min_samples_leaf: int = 1, max_features: Optional[int] = None, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees: List[DecisionTreeClassifierScratch] = []
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
    def fit(self, X: np.ndarray, y: np.ndarray):
        np.random.seed(self.random_state)
        m, n = X.shape
        # build trees in parallel
        def _build_tree(_):
            idx = np.random.choice(m, m, replace=True)
            X_sample, y_sample = X[idx], y[idx]
            if self.max_features is None:
                feat_idx = np.arange(n)
            else:
                feat_idx = np.random.choice(n, self.max_features, replace=False)
            tree = DecisionTreeClassifierScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.feat_idx = feat_idx
            tree.fit(X_sample[:, feat_idx], y_sample)
            return tree
        self.trees = Parallel(n_jobs=-1)(delayed(_build_tree)(i) for i in range(self.n_estimators))
        self.n_classes = len(np.unique(y))
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        # majority vote
        preds = np.array([tree.predict(X[:, tree.feat_idx]) for tree in self.trees])  # shape n_trees x m
        # majority per column
        out = []
        for col in preds.T:
            vals, counts = np.unique(col, return_counts=True)
            out.append(vals[np.argmax(counts)])
        return np.array(out)
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # average proba
        probas = np.array([tree.predict_proba(X[:, tree.feat_idx]) for tree in self.trees])  # t x m x c
        return np.mean(probas, axis=0)

class SVMScratch:
    """Linear SVM using gradient descent and hinge loss (OVR)."""
    def __init__(self, C: float = 1.0, learning_rate: float = 0.001, n_iter: int = 1000, class_weight: Optional[Dict[int,float]] = None):
        self.C = C
        self.lr = learning_rate
        self.n_iter = n_iter
        self.class_weight = class_weight
    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        k = len(classes)
        # OVR weights and bias
        self.W = np.zeros((k, n))
        self.b = np.zeros(k)
        # map class to index
        cls_to_idx = {c:i for i,c in enumerate(classes)}
        y_idx = np.array([cls_to_idx[yy] for yy in y])
        # class weights
        if self.class_weight is None:
            weights = np.ones(m)
        else:
            weights = np.array([self.class_weight[yy] for yy in y])
        # train each binary svm
        for i, cls in enumerate(classes):
            # binary labels
            y_bin = np.where(y==cls, 1, -1)
            w = np.zeros(n)
            b = 0.0
            for it in range(self.n_iter):
                # compute hinge loss gradient
                margin = y_bin * (X.dot(w) + b)
                idx_err = np.where(margin < 1)[0]
                grad_w = w - self.C * np.dot((weights[idx_err]*y_bin[idx_err]), X[idx_err])
                grad_b = -self.C * np.sum(weights[idx_err]*y_bin[idx_err])
                # update
                w -= self.lr * grad_w
                b -= self.lr * grad_b
            self.W[i] = w
            self.b[i] = b
        return self
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.W.T) + self.b  # shape m x k
    def predict(self, X: np.ndarray) -> np.ndarray:
        df = self.decision_function(X)
        idx = np.argmax(df, axis=1)
        return self.classes_[idx]
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # map decision to prob via softmax
        df = self.decision_function(X)
        exp = np.exp(df - np.max(df, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
