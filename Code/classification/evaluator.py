import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

# add these imports at top
from classification.base_model import CustomRandomForest, CustomLogisticRegression

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
    """Plot and return feature importances for RF or LR."""
    # 1) sklearn RF
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

    # 2) custom RandomForest
    elif isinstance(model, CustomRandomForest):
        importances = np.zeros(len(feature_names))
        def _gather(node, acc):
            if node is None or node.feature is None: 
                return
            acc.append(node.feature)
            _gather(node.left, acc)
            _gather(node.right, acc)

        for tree in model.trees:
            feats = []
            _gather(tree.root, feats)
            for f in feats:
                importances[f] += 1
        importances = importances / importances.sum()

    # 3) custom LogisticRegression
    elif isinstance(model, CustomLogisticRegression):
        # model.weights: shape (n_features, n_classes)
        coefs = model.weights
        # sum absolute across classes
        importances = np.sum(np.abs(coefs), axis=1)
        # normalize
        if importances.sum() > 0:
            importances = importances / importances.sum()

    else:
        print("Feature importance not implemented for this model type")
        return None

    # sort and plot
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12,6))
    plt.title("Feature importance")
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)),
               [feature_names[i] for i in indices],
               rotation=90, fontsize=8)
    plt.tight_layout()
    return importances, indices