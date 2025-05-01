import numpy as np

class ClassificationEvaluator:
    """
    Evaluator for classification models, computing accuracy, precision,
    recall, F1 and log‑loss without any sklearn calls.
    """
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        n = len(y_true)
        # 1) Accuracy
        accuracy = np.sum(y_true == y_pred) / n

        # 2) Precision, Recall, F1 (macro)
        classes = np.unique(y_true)
        precisions, recalls, f1s = [], [], []
        for c in classes:
            tp = np.sum((y_pred == c) & (y_true == c))
            fp = np.sum((y_pred == c) & (y_true != c))
            fn = np.sum((y_pred != c) & (y_true == c))
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec  = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
        precision_macro = np.mean(precisions)
        recall_macro    = np.mean(recalls)
        f1_macro        = np.mean(f1s)

        # 3) Log‑Loss
        eps = 1e-15
        clipped = np.clip(y_prob, eps, 1 - eps)
        # pick the probability assigned to the true class for each sample
        ll = -np.sum(np.log(clipped[np.arange(n), y_true])) / n

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'log_loss': ll
        }

    def report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Simple text report of the above metrics.
        """
        # random dummy y_prob so we can call evaluate
        n = len(y_pred)
        unique = np.unique(y_true)
        # one‑hot of y_pred to simulate a “pseudo‑proba” for report
        dummy_prob = np.eye(len(unique))[y_pred]
        metrics = self.evaluate(y_true, y_pred, dummy_prob)
        return '\n'.join(f"{k}: {v:.4f}" for k, v in metrics.items())
