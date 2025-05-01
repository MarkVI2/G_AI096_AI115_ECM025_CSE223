import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from classification.base_model import MultiClassGBDTClassifier


def optimize_hyperparameters(X: np.ndarray,
                             y: np.ndarray,
                             n_classes: int,
                             cv: int = 5,
                             scoring: str = 'accuracy') -> (dict, MultiClassGBDTClassifier):
    """
    Perform hyperparameter optimization for the MultiClassGBDTClassifier using GridSearchCV.

    Args:
        X: Feature matrix (samples x features) or sequences flattened
        y: Label array (samples,)
        n_classes: Number of target classes
        cv: Number of cross-validation folds
        scoring: Scoring metric for selecting best parameters

    Returns:
        best_params: Dictionary of best hyperparameters
        best_model: Trained MultiClassGBDTClassifier with best params
    """
    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'ccp_alpha': [0.0, 0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    # Initialize classifier
    clf = MultiClassGBDTClassifier(n_classes=n_classes)
    # Cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # Grid search
    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring=scoring,
        n_jobs=-1,
        verbose=2
    )
    # Run search
    grid.fit(X, y)
    # Extract best
    best_params = grid.best_params_
    best_model = grid.best_estimator_
    return best_params, best_model
