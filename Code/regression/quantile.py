import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class QuantileRegressorWrapper:
    """
    Wrapper for fitting multiple QuantileRegressor models to estimate prediction intervals.
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9], alpha=0.0, solver='highs'):
        self.quantiles = quantiles
        self.alpha = alpha
        self.solver = solver
        self.models = {}

    def fit(self, X, y):
        # Fit one QuantileRegressor per quantile
        for q in self.quantiles:
            qr = QuantileRegressor(alpha=self.alpha, quantile=q, solver=self.solver)
            qr.fit(X, y)
            self.models[q] = qr
        return self

    def predict(self, X):
        # Return array of shape (n_samples, n_quantiles)
        preds = []
        for q in self.quantiles:
            preds.append(self.models[q].predict(X))
        return np.vstack(preds).T

    def get_prediction_intervals(self, X):
        """
        Returns lower, median, upper predictions if quantiles=[low, mid, high]
        """
        pred_matrix = self.predict(X)
        # assume quantiles sorted in init
        return pred_matrix[:, 0], pred_matrix[:, 1], pred_matrix[:, 2]

class QuantileRegressorScratch(BaseEstimator, RegressorMixin):
    """
    Quantile Regression implemented from scratch.
    Minimizes the tilted loss function to estimate conditional quantiles.
    """
    def __init__(self, quantile=0.5, alpha=0.0, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.quantile = quantile  # q value (0 to 1)
        self.alpha = alpha  # regularization strength
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        
    def _tilted_loss(self, y_true, y_pred):
        """
        Compute the tilted (quantile) loss function
        L_q(y, f(x)) = q(y - f(x)) if y - f(x) >= 0 else (q-1)(y - f(x))
        """
        error = y_true - y_pred
        positive_error = np.maximum(error, 0)
        negative_error = np.maximum(-error, 0)
        return np.sum(self.quantile * positive_error + (1 - self.quantile) * negative_error)
    
    def _tilted_loss_grad(self, X, y_true, y_pred):
        """
        Compute gradient of the tilted loss with respect to coefficients
        """
        error = y_true - y_pred
        # For each data point, determine the gradient factor
        # q-1 if error < 0, q if error >= 0
        grad_factor = np.where(error >= 0, -self.quantile, -(self.quantile - 1))
        
        # Gradient for coefficients: sum of X_i * grad_factor_i
        grad_coef = np.dot(X.T, grad_factor) + self.alpha * self.coef_
        
        # Gradient for intercept: sum of grad_factor_i
        grad_intercept = np.sum(grad_factor)
        
        return grad_coef, grad_intercept
        
    def fit(self, X, y):
        """
        Fit the quantile regression model using gradient descent
        """
        n_samples, n_features = X.shape
        
        # Initialize coefficients
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        # Normalize input features
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-10  # avoid division by zero
        X_norm = (X - X_mean) / X_std
        
        # Store normalization parameters for prediction
        self.X_mean_ = X_mean
        self.X_std_ = X_std
        
        # Track loss for convergence check
        prev_loss = float('inf')
        
        # Gradient descent iterations
        for iteration in range(self.max_iter):
            # Compute predictions
            y_pred = X_norm @ self.coef_ + self.intercept_
            
            # Compute current loss
            loss = self._tilted_loss(y, y_pred) / n_samples
            if self.alpha > 0:
                # Add regularization term (L2 penalty)
                loss += 0.5 * self.alpha * np.sum(self.coef_ ** 2)
            
            # Check convergence
            if np.abs(prev_loss - loss) < self.tol:
                print(f"  Converged at iteration {iteration} with loss {loss:.6f}")
                break
                
            # Compute gradients
            grad_coef, grad_intercept = self._tilted_loss_grad(X_norm, y, y_pred)
            
            # Update parameters
            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept
            
            prev_loss = loss
            
            # Print progress occasionally
            if (iteration + 1) % 100 == 0:
                print(f"  Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.6f}")
        
        # Adjust coefficients for original scale
        self.orig_coef_ = self.coef_ / self.X_std_
        self.orig_intercept_ = self.intercept_ + np.sum(self.X_mean_ * (-self.orig_coef_))
        
        return self
    
    def predict(self, X):
        """
        Predict quantile values for X
        """
        # Normalize inputs using stored mean and std
        X_norm = (X - self.X_mean_) / self.X_std_
        
        # Compute prediction
        y_pred = X_norm @ self.coef_ + self.intercept_
        
        return y_pred

class QuantileRegressionForest:
    """
    Quantile Regression Forest - uses RandomForestRegressorScratch for 
    conditional quantile estimation.
    """
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=2, 
                 min_samples_leaf=1, quantiles=[0.1, 0.5, 0.9], random_state=None):
        from .base_models import RandomForestRegressorScratch
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.quantiles = quantiles
        self.random_state = random_state
        
        # Initialize forest model
        self.forest = RandomForestRegressorScratch(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        
    def fit(self, X, y):
        """Train the forest on data"""
        print(f"Training Quantile Regression Forest with {self.n_estimators} trees...")
        self.forest.fit(X, y)
        return self
        
    def predict(self, X, return_all_quantiles=False):
        """
        Get median (q=0.5) predictions by default or all specified quantiles.
        For QRF, we collect all leaf node predictions from trees, then compute
        the empirical quantiles of those predictions.
        """
        # For proper quantile forest, we need individual tree predictions
        # But as an approximation, we'll use the leaf values directly
        
        # First get all individual tree predictions
        all_preds = []
        
        for tree, indices in self.forest.trees:
            # Get feature-subset compatible input
            X_tree = X[:, indices]
            
            # Get predictions from this tree
            preds = tree.predict(X_tree)
            all_preds.append(preds)
            
        # Stack predictions - shape: (n_trees, n_samples)
        all_preds = np.vstack(all_preds)
        
        # For each sample, compute the quantiles across tree predictions
        n_samples = X.shape[0]
        
        if return_all_quantiles:
            # Return all requested quantiles
            quantile_preds = np.zeros((n_samples, len(self.quantiles)))
            
            for i in range(n_samples):
                tree_preds_for_sample = all_preds[:, i]
                quantile_preds[i] = np.quantile(tree_preds_for_sample, self.quantiles)
                
            return quantile_preds
        else:
            # Return just median (q=0.5) prediction
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else -1
            
            if median_idx >= 0:
                # Use the 0.5 quantile
                medians = np.zeros(n_samples)
                
                for i in range(n_samples):
                    tree_preds_for_sample = all_preds[:, i]
                    medians[i] = np.quantile(tree_preds_for_sample, 0.5)
                    
                return medians
            else:
                # Fallback to mean prediction
                return np.mean(all_preds, axis=0)
    
    def predict_quantiles(self, X):
        """
        Get all specified quantile predictions.
        Returns both the quantile values and predictions
        """
        quantile_preds = self.predict(X, return_all_quantiles=True)
        return np.array(self.quantiles), quantile_preds