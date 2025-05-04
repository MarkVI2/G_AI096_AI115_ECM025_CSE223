import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Print metrics
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Get detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'],
               yticklabels=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def analyze_feature_importance(model, feature_names):
    """Analyze and plot feature importance for tree-based models"""
    from classification.base_model import CustomRandomForest
    
    if hasattr(model, 'feature_importances_'):
        # For sklearn models like RandomForest
        importances = model.feature_importances_
    elif isinstance(model, CustomRandomForest):
        # For our custom RandomForest, calculate importances manually
        importances = np.zeros(len(feature_names))
        # This is a simplified version, ideally we'd track feature usage in the custom model
        for tree in model.trees:
            # Count feature usage in each tree (simplified)
            used_features = _extract_used_features(tree.root, set())
            for feat in used_features:
                if feat is not None:
                    importances[feat] += 1
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
    else:
        # For other models, we can't easily get feature importance
        print("Feature importance analysis not implemented for this model type")
        return None
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    return importances, indices

def _extract_used_features(node, feature_set):
    """Helper function to extract features used in a tree"""
    if node is None:
        return feature_set
    
    if node.feature is not None:
        feature_set.add(node.feature)
        feature_set = _extract_used_features(node.left, feature_set)
        feature_set = _extract_used_features(node.right, feature_set)
    
    return feature_set