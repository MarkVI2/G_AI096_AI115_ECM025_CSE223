# Default hyperparameters for faster training and testing
# Format: BEST_HYPERPARAMS[model_type][dataset_num]
BEST_HYPERPARAMS = {
    'random_forest': {
        1: {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5},
        2: {'n_estimators': 100, 'max_depth': 12, 'min_samples_split': 5},
        3: {'n_estimators': 50, 'max_depth': 12, 'min_samples_split': 5},
        4: {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5},
    },
    'logistic_regression': {
        1: {'learning_rate': 0.01, 'iterations': 500, 'class_weights': 'balanced'},
        2: {'learning_rate': 0.01, 'iterations': 500, 'class_weights': 'balanced'},
        3: {'learning_rate': 0.01, 'iterations': 500, 'class_weights': 'balanced'},
        4: {'learning_rate': 0.01, 'iterations': 500, 'class_weights': 'balanced'},
    }
}
