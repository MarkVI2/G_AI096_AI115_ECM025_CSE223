import numpy as np

class RegressionEvaluator:
    """
    Evaluator for regression models, computing MAE, MSE, RMSE, and R-squared.
    """
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        n = len(y_true)
        # Mean Absolute Error
        mae = np.mean(np.abs(y_true - y_pred))
        # Mean Squared Error
        mse = np.mean((y_true - y_pred) ** 2)
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        # R-squared
        total_var = np.sum((y_true - np.mean(y_true)) ** 2)
        explained_var = np.sum((y_pred - np.mean(y_true)) ** 2)
        r2 = explained_var / total_var if total_var > 0 else 0.0
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }

    @staticmethod
    def report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        metrics = RegressionEvaluator.evaluate(y_true, y_pred)
        lines = [f"{name}: {value:.4f}" for name, value in metrics.items()]
        return "\n".join(lines)
