import numpy as np
# no external meta-learner imports to limit sklearn usage
from classification.base_model import BasicGBDTClassifier, LogisticRegressionScratch
from joblib import Parallel, delayed
 
class MetaClassifier:
    """
    Simple ensemble meta-classifier by averaging base-learner probabilities.
    """
    def __init__(self, n_estimators_list=None, random_state: int = 42, lr_params=None):
        # Default hyperparameters for base and meta learners
        if lr_params is None:
            lr_params = {}
        self.random_state = random_state
        # Base learners configuration
        if n_estimators_list is None:
            n_estimators_list = [50, 50]
        # Initialize base GBDT models
        self.base_models = [
            BasicGBDTClassifier(
                n_estimators=n,
                learning_rate=lr_params.get('learning_rate', 0.1),
                max_depth=lr_params.get('max_depth', 3)
            ) for n in n_estimators_list
        ]
        # Meta-learner (logistic regression scratch)
        self.meta_model = LogisticRegressionScratch(
            learning_rate=lr_params.get('meta_lr', 0.01),
            n_iterations=lr_params.get('meta_iter', 500),
            reg_lambda=lr_params.get('meta_reg', 0.0)
        )
 
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Flatten sequences
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        # Fit base learners in parallel
        Parallel(n_jobs=-1)(delayed(m.fit)(X_flat, y) for m in self.base_models)
        # Compute base learner probabilities in parallel for meta-features
        base_probs = Parallel(n_jobs=-1)(delayed(m.predict_proba)(X_flat) for m in self.base_models)
        # Stack base probabilities as meta-features
        meta_X = np.hstack(base_probs)
        # Fit meta-learner
        self.meta_model.fit(meta_X, y)
        return self
 
    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
 
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        X_flat = X.reshape(n_samples, -1)
        # Generate base probabilities in parallel
        base_probs = Parallel(n_jobs=-1)(delayed(m.predict_proba)(X_flat) for m in self.base_models)
        # Concatenate for meta input
        meta_X = np.hstack(base_probs)
        # Predict probabilities via meta-learner
        return self.meta_model.predict_proba(meta_X)
