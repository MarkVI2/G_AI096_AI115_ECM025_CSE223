import numpy as np
from regression.base_models import BaseRegressor

class QuantileRegressor(BaseRegressor):
    """
    Quantile Regression implementation.
    
    This model predicts a specific quantile of the target distribution,
    useful for creating prediction intervals.
    """
    
    def __init__(self, quantile=0.5, alpha=0.1, max_iter=1000, tol=1e-4):
        """
        Initialize Quantile Regressor.
        
        Args:
            quantile: Target quantile (0.5 = median)
            alpha: Regularization strength
            max_iter: Maximum number of iterations
            tol: Tolerance for stopping criteria
        """
        self.quantile = quantile
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Fit Quantile Regression model.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Learning rate
        learning_rate = 0.01
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Predictions
            y_pred = X.dot(self.weights) + self.bias
            
            # Compute quantile loss gradients
            errors = y - y_pred
            
            # Quantile loss gradient
            grad_weights = np.zeros_like(self.weights)
            grad_bias = 0
            
            for i in range(n_samples):
                # For each sample, compute the gradient
                if errors[i] > 0:
                    # If actual > predicted, gradient is -quantile * X[i]
                    grad_weights += -self.quantile * X[i]
                    grad_bias += -self.quantile
                else:
                    # If actual <= predicted, gradient is (1-quantile) * X[i]
                    grad_weights += (1 - self.quantile) * X[i]
                    grad_bias += (1 - self.quantile)
            
            # Average gradient
            grad_weights /= n_samples
            grad_bias /= n_samples
            
            # Add L2 regularization for weights
            grad_weights += self.alpha * self.weights
            
            # Update weights and bias
            self.weights -= learning_rate * grad_weights
            self.bias -= learning_rate * grad_bias
            
            # Check convergence
            if np.linalg.norm(learning_rate * grad_weights) < self.tol and \
               abs(learning_rate * grad_bias) < self.tol:
                break
                
        return self
    
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


class QuantilePredictionInterval:
    """
    Creates prediction intervals using quantile regression.
    """
    
    def __init__(self, lower_quantile=0.1, upper_quantile=0.9, alpha=0.1):
        """
        Initialize prediction interval generator.
        
        Args:
            lower_quantile: Lower bound quantile
            upper_quantile: Upper bound quantile
            alpha: Regularization strength
        """
        self.lower_model = QuantileRegressor(quantile=lower_quantile, alpha=alpha)
        self.upper_model = QuantileRegressor(quantile=upper_quantile, alpha=alpha)
        
    def fit(self, X, y):
        """
        Fit both lower and upper quantile models.
        
        Args:
            X: Features array
            y: Target values
        """
        self.lower_model.fit(X, y)
        self.upper_model.fit(X, y)
        return self
        
    def predict_interval(self, X):
        """
        Predict lower and upper bounds.
        
        Args:
            X: Features array
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        lower_bounds = self.lower_model.predict(X)
        upper_bounds = self.upper_model.predict(X)
        return lower_bounds, upper_bounds