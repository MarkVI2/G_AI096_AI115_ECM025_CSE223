# Default hyperparameters for faster training and testing
# Format: BEST_HYPERPARAMS[model_type][dataset_num]

# These numbers are purely based on trial and error approach and they are not optimal
# for the datasets and models. They are just a starting point for faster training and testing.
# It is recommended to find the best hyperparameters using grid search or random search
# for each dataset and model type
BEST_HYPERPARAMS = {
    'random_forest': {
        1: {'n_estimators': 50, 'max_depth': 10, 'min_samples_split': 5},
        2: {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 10},
        3: {'n_estimators': 50, 'max_depth': 12, 'min_samples_split': 5},
        4: {'n_estimators': 100, 'max_depth': 20, 'min_samples_split': 5},
    },
    'logistic_regression': {
        1: {'learning_rate': 0.01, 'iterations': 1000, 'class_weights': 'balanced'},
        2: {'learning_rate': 0.05, 'iterations': 500, 'class_weights': 'balanced'},
        3: {'learning_rate': 0.01, 'iterations': 1000, 'class_weights': 'balanced'},
        4: {'learning_rate': 0.05, 'iterations': 500, 'class_weights': 'balanced'},
    }
}
