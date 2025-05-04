import numpy as np
from imblearn.over_sampling import SMOTE

def handle_imbalance(X_train, y_train, strategy='smote'):
    """Handle class imbalance using SMOTE or class weights"""
    if strategy == 'smote':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    elif strategy == 'class_weights':
        # Calculate class weights
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        class_weights = {i: total_samples / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts)}
        return X_train, y_train, class_weights
    else:
        return X_train, y_train