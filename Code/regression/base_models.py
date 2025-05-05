import numpy as np
from abc import ABC, abstractmethod

class BaseRegressor(ABC):
    """Abstract base class for all regressors"""
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        pass


class RidgeRegression(BaseRegressor):
    """
    Ridge Regression implementation from scratch.
    
    Ridge regression adds L2 regularization to linear regression to prevent overfitting.
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Initialize Ridge Regression model.
        
        Args:
            alpha: Regularization strength
            max_iter: Maximum number of iterations
            tol: Tolerance for stopping criteria
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Fit Ridge Regression model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Add a column of ones for the bias term
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]
        
        # Initialize weights
        weights = np.zeros(n_features + 1)
        
        # Analytical solution using normal equation with regularization
        # (X^T X + alpha I)^-1 X^T y
        # Note: we don't regularize the bias term (first coefficient)
        identity = np.eye(n_features + 1)
        identity[0, 0] = 0  # Don't regularize bias term
        
        try:
            weights = np.linalg.solve(
                X_with_bias.T.dot(X_with_bias) + self.alpha * identity,
                X_with_bias.T.dot(y)
            )
        except np.linalg.LinAlgError:
            # If matrix is singular, use gradient descent
            print("Matrix is singular, using gradient descent")
            weights = self._fit_gradient_descent(X_with_bias, y)
        
        self.bias = weights[0]
        self.weights = weights[1:]
        
        return self
    
    def _fit_gradient_descent(self, X, y):
        """
        Fit using gradient descent when normal equation fails.
        
        Args:
            X: Features array with bias column
            y: Target values
            
        Returns:
            Optimal weights
        """
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        
        # Learning rate
        learning_rate = 0.01
        
        for i in range(self.max_iter):
            # Predictions
            y_pred = X.dot(weights)
            
            # Compute gradients
            error = y_pred - y
            
            # Gradient of loss with respect to weights
            gradient = (1/n_samples) * X.T.dot(error)
            
            # Add regularization gradient (L2)
            reg_gradient = np.copy(gradient)
            reg_gradient[1:] += (self.alpha / n_samples) * weights[1:]  # Skip bias
            
            # Update weights
            weights -= learning_rate * reg_gradient
            
            # Check convergence
            if np.linalg.norm(learning_rate * reg_gradient) < self.tol:
                break
                
        return weights
    
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return X.dot(self.weights) + self.bias


class DecisionTreeRegressor:
    """
    Decision Tree Regressor implementation from scratch.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2):
        """
        Initialize Decision Tree Regressor.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def fit(self, X, y):
        """
        Fit Decision Tree Regressor.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        self.tree = self._build_tree(X, y, depth=0)
        return self
        
    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.
        
        Args:
            X: Features
            y: Targets
            depth: Current depth
            
        Returns:
            Tree node
        """
        n_samples, n_features = X.shape
        
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            leaf_value = np.mean(y)
            return {'value': leaf_value}
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        # If no good split found, create leaf node
        if best_feature is None:
            leaf_value = np.mean(y)
            return {'value': leaf_value}
        
        # Split the data
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Check if split produces empty node
        if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
            leaf_value = np.mean(y)
            return {'value': leaf_value}
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
        
    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold for splitting.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Tuple of (best_feature, best_threshold)
        """
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        # Calculate parent variance (MSE before split)
        parent_var = np.var(y) * n_samples
        
        # Initialize variables to track best split
        best_feature, best_threshold = None, None
        best_variance_reduction = 0
        
        # For each feature
        for feature in range(n_features):
            # Get unique threshold values
            thresholds = np.unique(X[:, feature])
            
            # For each potential threshold
            for threshold in thresholds:
                # Split data
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                # Skip if split is degenerate
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # Calculate weighted variance after split
                left_var = np.var(y[left_indices]) * np.sum(left_indices)
                right_var = np.var(y[right_indices]) * np.sum(right_indices)
                
                # Calculate variance reduction
                variance_reduction = parent_var - (left_var + right_var)
                
                # Update best split if this is better
                if variance_reduction > best_variance_reduction:
                    best_variance_reduction = variance_reduction
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def predict(self, X):
        """
        Make predictions using the fitted tree.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        if self.tree is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
    
    def _predict_sample(self, sample, node):
        """
        Predict value for a single sample by traversing the tree.
        
        Args:
            sample: Single sample features
            node: Current tree node
            
        Returns:
            Predicted value
        """
        # If leaf node, return value
        if 'value' in node:
            return node['value']
        
        # Determine which branch to follow
        if sample[node['feature']] <= node['threshold']:
            return self._predict_sample(sample, node['left'])
        else:
            return self._predict_sample(sample, node['right'])


class RandomForestRegressor(BaseRegressor):
    """
    Random Forest Regressor implementation from scratch.
    
    Ensemble of decision trees using bootstrap aggregation (bagging).
    """
    
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, max_features='sqrt', 
                 bootstrap=True, random_state=None):
        """
        Initialize Random Forest Regressor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of each tree
            min_samples_split: Minimum samples required to split a node
            max_features: Number of features to consider for best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        """
        Fit Random Forest Regressor.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Set random seed if specified
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Determine max_features
        if self.max_features == 'sqrt':
            self.max_features_ = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            self.max_features_ = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            self.max_features_ = int(self.max_features * n_features)
        elif isinstance(self.max_features, int):
            self.max_features_ = self.max_features
        else:
            self.max_features_ = n_features
            
        # Ensure max_features is within valid range
        self.max_features_ = max(1, min(self.max_features_, n_features))
        
        # Build forest
        self.trees = []
        for _ in range(self.n_estimators):
            # Create bootstrap sample if enabled
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
            else:
                X_bootstrap = X
                y_bootstrap = y
            
            # Build tree with feature subsampling
            tree = self._build_tree_with_feature_subsampling(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            
        return self
    
    def _build_tree_with_feature_subsampling(self, X, y):
        """
        Build a decision tree with feature subsampling.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Fitted decision tree
        """
        # Initialize tree
        tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split
        )
        
        # Subsample features for this tree
        n_features = X.shape[1]
        feature_indices = np.random.choice(n_features, self.max_features_, replace=False)
        
        # Fit tree on subset of features
        tree.fit(X[:, feature_indices], y)
        
        # Store feature indices with the tree for prediction
        tree.feature_indices = feature_indices
        
        return tree
    
    def predict(self, X):
        """
        Make predictions using the random forest.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        if len(self.trees) == 0:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Collect predictions from all trees
        predictions = np.zeros((X.shape[0], self.n_estimators))
        
        for i, tree in enumerate(self.trees):
            # Select features used by this tree
            X_subset = X[:, tree.feature_indices]
            predictions[:, i] = tree.predict(X_subset)
            
        # Average predictions from all trees
        return np.mean(predictions, axis=1)