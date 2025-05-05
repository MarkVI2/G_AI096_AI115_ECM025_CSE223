import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RegressionEvaluator:
    """
    Evaluator for regression models.
    
    Calculates common regression metrics and creates visualizations.
    """
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate regression model performance.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Mean Squared Error
        mse = self.mean_squared_error(y_true, y_pred)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # Mean Absolute Error
        mae = self.mean_absolute_error(y_true, y_pred)
        
        # R-squared
        r2 = self.r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error
        mape = self.mean_absolute_percentage_error(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate Mean Squared Error.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            MSE value
        """
        return np.mean((y_true - y_pred) ** 2)
    
    def mean_absolute_error(self, y_true, y_pred):
        """
        Calculate Mean Absolute Error.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def r2_score(self, y_true, y_pred):
        """
        Calculate R-squared score.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            R-squared value
        """
        # Total sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        # Residual sum of squares
        ss_res = np.sum((y_true - y_pred) ** 2)
        
        # R-squared
        if ss_tot == 0:
            return 0  # Avoid division by zero
        return 1 - (ss_res / ss_tot)
    
    def mean_absolute_percentage_error(self, y_true, y_pred):
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            MAPE value
        """
        # Filter out zero values in y_true to avoid division by zero
        mask = y_true != 0
        return 100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    
    def plot_residuals(self, y_true, y_pred, save_path=None):
        """
        Create residual plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save the plot (if None, plot is displayed)
        """
        residuals = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_prediction_error(self, y_true, y_pred, save_path=None):
        """
        Create actual vs predicted plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(np.max(y_true), np.max(y_pred))
        min_val = min(np.min(y_true), np.min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_distribution(self, y_true, y_pred, save_path=None):
        """
        Create error distribution plot.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            save_path: Path to save the plot (if None, plot is displayed)
        """
        errors = y_true - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()