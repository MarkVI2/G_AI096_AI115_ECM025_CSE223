import numpy as np
from .base_models import DecisionTreeRegressorScratch, RandomForestRegressorScratch, RidgeRegressionScratch, SVRScratch
from .quantile import QuantileRegressorScratch

class EnsembleRegressor:
    """
    Ensemble regressor that averages predictions from multiple base regressors
    and can fit quantile regressors for uncertainty.
    """
    def __init__(self, regressors=None, weights=None, n_jobs=1):
        # default base regressors
        if regressors is None:
            self.regressors = [
                RandomForestRegressorScratch(n_estimators=50, max_depth=5, random_state=42),
                RidgeRegressionScratch(alpha=1.0),
                SVRScratch(C=1.0, epsilon=0.1, kernel='rbf', max_iter=500)
            ]
        else:
            self.regressors = regressors
        
        self.weights = weights
        self.n_jobs = n_jobs
        self.quantile_models = None

    def fit(self, X, y):
        """Train all base regressors on the same data"""
        print(f"Training ensemble with {len(self.regressors)} models...")
        
        # Train each model
        for i, model in enumerate(self.regressors):
            print(f"Training model {i+1}/{len(self.regressors)}: {model.__class__.__name__}")
            model.fit(X, y)
            
        return self

    def predict(self, X):
        """Make predictions by averaging all regressor outputs"""
        # Collect predictions from each model
        all_preds = np.zeros((len(self.regressors), X.shape[0]))
        
        for i, model in enumerate(self.regressors):
            all_preds[i] = model.predict(X)
        
        # Weighted average if weights provided, otherwise simple mean
        if self.weights is not None:
            ensemble_pred = np.average(all_preds, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(all_preds, axis=0)
            
        return ensemble_pred

    def fit_quantiles(self, X, y, quantiles=[0.1, 0.5, 0.9]):
        """Train quantile regression models for uncertainty estimation"""
        self.quantile_models = []
        
        for q in quantiles:
            print(f"Training quantile regressor for {q} quantile")
            qr = QuantileRegressorScratch(quantile=q)
            qr.fit(X, y)
            self.quantile_models.append((q, qr))
            
        return self

    def predict_quantiles(self, X):
        """Get quantile predictions for uncertainty bounds"""
        if self.quantile_models is None:
            raise RuntimeError("Quantile models not trained. Call fit_quantiles first.")
        
        # Predict with each quantile model
        quantile_values = []
        quantile_preds = []
        
        for q, model in self.quantile_models:
            quantile_values.append(q)
            preds = model.predict(X)
            quantile_preds.append(preds)
            
        # Organize predictions as (n_samples, n_quantiles) array
        return np.array(quantile_values), np.array(quantile_preds).T

    def predict_with_intervals(self, X):
        """Predict with the ensemble and provide prediction intervals"""
        # Get mean prediction
        mean_pred = self.predict(X)
        
        if self.quantile_models is None:
            return mean_pred, None, None
        
        # Get quantile predictions for bounds
        quantiles, quantile_preds = self.predict_quantiles(X)
        
        # Find indices for lower and upper quantiles (assuming sorted)
        lower_idx = 0  # typically 0.1 quantile
        upper_idx = len(quantiles) - 1  # typically 0.9 quantile
        
        lower_bound = quantile_preds[:, lower_idx]
        upper_bound = quantile_preds[:, upper_idx]
        
        return mean_pred, lower_bound, upper_bound

class WeightedEnsembleRegressor(EnsembleRegressor):
    """
    Extension of EnsembleRegressor that learns optimal weights
    for combining models using validation data.
    """
    def __init__(self, regressors=None, n_jobs=1):
        super().__init__(regressors=regressors, weights=None, n_jobs=n_jobs)
        self.initial_weights = None
        
    def fit(self, X, y, X_val=None, y_val=None, val_size=0.2):
        """
        Train base models and learn optimal weights using validation data.
        If X_val/y_val not provided, splits training data.
        """
        # Split for validation if not provided
        if X_val is None or y_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)
        else:
            X_train, y_train = X, y
            
        # First train each base model
        print(f"Training {len(self.regressors)} base models...")
        for i, model in enumerate(self.regressors):
            print(f"  Training model {i+1}: {model.__class__.__name__}")
            model.fit(X_train, y_train)
            
        # Now optimize weights using validation set
        self._optimize_weights(X_val, y_val)
        
        return self
    
    def _optimize_weights(self, X_val, y_val):
        """Find optimal weights that minimize MSE on validation data"""
        # Get predictions from each model
        val_preds = np.zeros((len(self.regressors), len(y_val)))
        
        for i, model in enumerate(self.regressors):
            val_preds[i] = model.predict(X_val)
            
        # Set up optimization to find weights
        from scipy.optimize import minimize
        
        # MSE loss with weights
        def weighted_mse(weights):
            weights = weights / np.sum(weights)  # normalize
            weighted_pred = np.sum(weights.reshape(-1, 1) * val_preds, axis=0)
            return np.mean((weighted_pred - y_val) ** 2)
        
        # Initial equal weights
        initial_weights = np.ones(len(self.regressors)) / len(self.regressors)
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: all weights between 0 and 1
        bounds = [(0, 1) for _ in range(len(self.regressors))]
        
        # Optimize
        result = minimize(
            weighted_mse, 
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Store optimized weights
        self.weights = result.x
        
        # Print weight distribution
        model_names = [model.__class__.__name__ for model in self.regressors]
        print("Optimized ensemble weights:")
        for name, weight in zip(model_names, self.weights):
            print(f"  {name}: {weight:.4f}")
            
        return self
