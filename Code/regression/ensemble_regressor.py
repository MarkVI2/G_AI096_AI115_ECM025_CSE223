import numpy as np
from regression.base_models import BaseRegressor

class EnsembleRegressor(BaseRegressor):
    """
    Ensemble regressor that combines multiple base models.
    
    This implementation uses a weighted average of base model predictions.
    """
    
    def __init__(self, base_models=None, weights=None):
        """
        Initialize Ensemble Regressor.
        
        Args:
            base_models: List of regressor objects implementing fit and predict
            weights: List of weights for each model (default: equal weights)
        """
        self.base_models = base_models if base_models is not None else []
        self.weights = weights
        self.fitted_models = []
        
    def add_model(self, model):
        """
        Add a base model to the ensemble.
        
        Args:
            model: Regressor object implementing fit and predict
        """
        self.base_models.append(model)
        
    def fit(self, X, y):
        """
        Fit all base models.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        if len(self.base_models) == 0:
            raise ValueError("No base models to fit. Add models using add_model().")
            
        self.fitted_models = []
        
        for model in self.base_models:
            # Fit each base model
            model.fit(X, y)
            self.fitted_models.append(model)
            
        # If weights not provided, use equal weights
        if self.weights is None:
            self.weights = np.ones(len(self.fitted_models)) / len(self.fitted_models)
        else:
            # Normalize weights to sum to 1
            self.weights = np.array(self.weights) / np.sum(self.weights)
            
        # Find optimal weights if enough data
        if X.shape[0] > 100:  # Only optimize with sufficient data
            self._optimize_weights(X, y)
            
        return self
        
    def _optimize_weights(self, X, y):
        """
        Optimize weights based on model performance.
        
        This method updates weights based on inverse mean squared error.
        
        Args:
            X: Features
            y: Target values
        """
        n_models = len(self.fitted_models)
        
        # Get predictions from each model
        predictions = np.zeros((X.shape[0], n_models))
        
        for i, model in enumerate(self.fitted_models):
            predictions[:, i] = model.predict(X)
            
        # Calculate mean squared error for each model
        mse = np.mean((predictions - y.reshape(-1, 1))**2, axis=0)
        
        # Set weights inversely proportional to MSE (better models get higher weights)
        # Add small constant to avoid division by zero
        inverse_mse = 1 / (mse + 1e-10)
        self.weights = inverse_mse / np.sum(inverse_mse)
        
    def predict(self, X):
        """
        Make predictions using weighted average of base models.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        if len(self.fitted_models) == 0:
            raise ValueError("No fitted models. Call fit() first.")
            
        # Get predictions from each model
        predictions = np.zeros((X.shape[0], len(self.fitted_models)))
        
        for i, model in enumerate(self.fitted_models):
            predictions[:, i] = model.predict(X)
            
        # Weighted average of predictions
        return np.dot(predictions, self.weights)
    
    def predict_with_individual_models(self, X):
        """
        Make predictions with each individual model.
        
        Useful for analyzing performance of base models.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Dictionary mapping model name to predictions
        """
        if len(self.fitted_models) == 0:
            raise ValueError("No fitted models. Call fit() first.")
            
        individual_predictions = {}
        
        for i, model in enumerate(self.fitted_models):
            model_name = model.__class__.__name__
            individual_predictions[model_name] = model.predict(X)
            
        # Also include ensemble prediction
        individual_predictions['Ensemble'] = self.predict(X)
        
        return individual_predictions