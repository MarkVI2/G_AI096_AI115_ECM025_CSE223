import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from classification.base_model import MultiClassGBDTClassifier
from joblib import Parallel, delayed


def optimize_hyperparameters(X: np.ndarray,
                             y: np.ndarray,
                             n_classes: int,
                             cv: int = 5,
                             scoring: str = 'accuracy',
                             n_jobs: int = -1,
                             apply_smote: bool = False,
                             random_state: int = 42,
                             max_iter: int = 10):
    """
    Perform hyperparameter optimization for the MultiClassGBDTClassifier using RandomizedSearchCV.

    Args:
        X: Feature matrix (samples x features) or sequences flattened
        y: Label array (samples,)
        n_classes: Number of target classes
        cv: Number of cross-validation folds or list of (train, val) indices
        scoring: Scoring metric for selecting best parameters
        n_jobs: Number of parallel jobs (-1 for all cores)
        apply_smote: Whether to apply SMOTE within each fold
        random_state: Random state for reproducibility
        max_iter: Number of parameter samples for RandomizedSearchCV

    Returns:
        best_params: Dictionary of best hyperparameters
        best_model: Trained MultiClassGBDTClassifier with best params
    """
    # Define parameter grid for tuning
    param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'ccp_alpha': [0.0, 0.01],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    }

    # Adjust parameter distribution based on max_iter
    if max_iter < 10:
        # Use smaller parameter space for faster search
        param_dist = {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'ccp_alpha': [0.0],
            'subsample': [0.8, 1.0]
        }

    # Initialize classifier
    # Set the n_jobs parameter inside MultiClassGBDTClassifier
    clf = MultiClassGBDTClassifier(n_classes=n_classes, n_jobs=n_jobs)
    
    # If using SMOTE, create a custom CV function that applies SMOTE within each fold
    if apply_smote:
        from imblearn.over_sampling import SMOTE
        from sklearn.base import clone
        
        print("Setting up SMOTE inside CV folds for proper balancing...")
        
        # Define a function to process a single fold in parallel
        def smote_cv_split(estimator, X, y, train_idx, test_idx):
            # Get train and test data for this fold
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            # Apply SMOTE only to training data
            smote = SMOTE(random_state=random_state, k_neighbors=3)  # Use fewer neighbors for speed
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            
            # Clone and fit estimator on resampled data
            est = clone(estimator)
            est.fit(X_train_res, y_train_res)
            
            # Predict and score on test data
            return est.score(X_test, y_test)
        
        # Create a custom scorer that applies SMOTE within each fold
        def custom_scorer(estimator, X, y):
            if isinstance(cv, int):
                # Use StratifiedKFold
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
                cv_splits = list(skf.split(X, y))
            else:
                # Use pre-defined CV splits (for GroupKFold)
                cv_splits = cv
            
            # Apply smote_cv_split to each fold in parallel
            scores = Parallel(n_jobs=n_jobs)(
                delayed(smote_cv_split)(estimator, X, y, train_idx, test_idx)
                for train_idx, test_idx in cv_splits
            )
            
            return np.mean(scores)
        
        # Use the custom scorer
        scoring_func = custom_scorer
    else:
        # Choose appropriate CV strategy based on input type
        if isinstance(cv, int):
            # Use StratifiedKFold for simple fold count
            cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        else:
            # Use pre-computed fold indices (for GroupKFold)
            cv_strategy = cv
        
        # Use standard scoring
        scoring_func = scoring

    # Use RandomizedSearchCV for faster tuning with limited iterations
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        cv=cv if not apply_smote else 5,  # Use regular CV when apply_smote=True
        scoring=scoring_func,
        n_jobs=n_jobs,
        n_iter=max_iter,
        verbose=1,
        pre_dispatch='2*n_jobs',
        random_state=random_state
    )
    
    # Run search
    search.fit(X, y)
    
    # Extract best
    best_params = search.best_params_
    
    # For final model, apply SMOTE to full dataset if requested
    if apply_smote:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X, y)
        
        # Create a new model with best params and fit on full resampled data
        best_model = MultiClassGBDTClassifier(n_classes=n_classes, n_jobs=n_jobs, **best_params)
        best_model.fit(X_res, y_res)
    else:
        best_model = search.best_estimator_
    
    print(f"Search completed. Best score: {search.best_score_:.4f}")
    
    return best_params, best_model
