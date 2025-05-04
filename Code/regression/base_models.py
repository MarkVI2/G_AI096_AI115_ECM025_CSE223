import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class DecisionTreeRegressorScratch(BaseEstimator, RegressorMixin):
    """
    CART regression tree implementation from scratch.
    """
    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1, n_random_splits=None):
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

class RandomForestRegressorScratch(BaseEstimator, RegressorMixin):
    """
    Random Forest Regressor implemented from scratch.
    """
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split=2, 
                 min_samples_leaf=1, subsample=1.0, max_features=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        
        # Determine max_features if not explicitly set
        if self.max_features is None:
            self.max_features_ = int(np.sqrt(n_features))
        elif isinstance(self.max_features, float) and self.max_features < 1.0:
            self.max_features_ = int(self.max_features * n_features)
        else:
            self.max_features_ = int(self.max_features)
        
        # Calculate subsample size
        subsample_size = int(n_samples * self.subsample)
        
        # Train individual trees
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=subsample_size, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Feature bagging - select random subset of features for this tree
            feature_indices = np.random.choice(n_features, size=self.max_features_, replace=False)
            X_feature_bagged = X_sample[:, feature_indices]
            
            # Train decision tree
            tree = DecisionTreeRegressorScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                n_random_splits=16  # Use a fixed number of random splits for efficiency
            )
            tree.fit(X_feature_bagged, y_sample)
            
            # Store tree and its selected features
            self.trees.append((tree, feature_indices))
            
            # Report progress periodically
            if (i+1) % 10 == 0:
                print(f"  Trained {i+1}/{self.n_estimators} trees")
                
        return self

    def predict(self, X):
        # Get predictions from each tree
        n_samples = X.shape[0]
        predictions = np.zeros((self.n_estimators, n_samples))
        
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_tree = X[:, feature_indices]
            predictions[i] = tree.predict(X_tree)
        
        # Return average prediction
        return np.mean(predictions, axis=0)

class RidgeRegressionScratch(BaseEstimator, RegressorMixin):
    """
    Ridge regression implemented from scratch via closed-form solution.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Add intercept column to X
        X_aug = np.column_stack((np.ones(n_samples), X))
        
        # Compute closed-form solution: (X^T X + αI)^-1 X^T y
        # Use regularization on all coefficients except intercept
        reg_matrix = np.eye(n_features + 1)
        reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Compute coefficients
        XTX = X_aug.T @ X_aug
        reg_term = self.alpha * reg_matrix
        XTX_reg = XTX + reg_term
        
        try:
            # Try direct inversion for small matrices
            inv_XTX_reg = np.linalg.inv(XTX_reg)
            coeffs = inv_XTX_reg @ X_aug.T @ y
        except np.linalg.LinAlgError:
            # For stability issues, use pseudo-inverse instead
            coeffs = np.linalg.lstsq(XTX_reg, X_aug.T @ y, rcond=None)[0]
        
        # Extract intercept and coefficients
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_

class MultiLinearRegression(BaseEstimator,RegressorMixin):
    """Multi-linear regression placeholder."""
    pass
