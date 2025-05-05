import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add Code directory to system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from regression.base_models import RidgeRegression, RandomForestRegressor
from regression.ensemble_regressor import EnsembleRegressor
from regression.evaluator import RegressionEvaluator
from regression.quantile import QuantileRegressor
from regression.gradient_boosting import HistGradientBoostingRegressor

class RegressionPipeline:
    def __init__(self, datasets=['FD001', 'FD002', 'FD003', 'FD004'], 
                 classifier='random_forest', base_path=None):
        """
        Initialize the regression pipeline.
        
        Args:
            datasets: List of dataset names to process
            classifier: The classifier used for stage predictions
            base_path: Base path to the project
        """
        self.datasets = datasets
        self.classifier = classifier
        
        if base_path is None:
            # Try to determine the base path automatically
            current_file = os.path.abspath(__file__)
            # Navigate up to the project root
            base_path = os.path.dirname(os.path.dirname(current_file))
        
        self.base_path = base_path
        self.results_path = os.path.join(base_path, 'results')
        self.output_path = os.path.join(self.results_path, 'regression')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Initialize evaluation metrics storage
        self.metrics = {}
    
    def load_dataset(self, dataset, is_train=True):
        """
        Load a specific dataset with stage predictions.
        
        Args:
            dataset: Dataset name (e.g., 'FD001')
            is_train: Whether to load training or test data
        
        Returns:
            Pandas DataFrame with the loaded data
        """
        file_type = 'train' if is_train else 'test'
        file_path = os.path.join(
            self.results_path, 
            'classification', 
            dataset, 
            self.classifier,
            f'{file_type}_with_preds.csv'
        )
        
        print(f"Loading {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        return pd.read_csv(file_path)
    
    def prepare_labels(self, df):
        """
        Prepare target labels: time (in cycles) until next stage transition.
        
        Args:
            df: DataFrame with time_cycles and predicted_stage columns
            
        Returns:
            Tuple of (X features, y labels)
        """
        # Sort by unit and time for time series processing
        df = df.sort_values(['unit_number', 'time_cycles'])
        
        # Initialize the target column
        df['time_to_next_stage'] = np.nan
        
        # Process each unit separately
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit].copy()
            
            # For each stage, calculate time to next stage
            for stage in range(4):  # stages 0, 1, 2, 3
                stage_data = unit_data[unit_data['predicted_stage'] == stage]
                
                if stage_data.empty:
                    continue
                    
                # If this is the last stage (3), use RUL directly
                if stage == 3:
                    df.loc[stage_data.index, 'time_to_next_stage'] = stage_data['RUL']
                    continue
                
                # Find the next stage transition point
                next_stage_start = unit_data[
                    (unit_data['predicted_stage'] == stage + 1) & 
                    (unit_data['time_cycles'] > stage_data['time_cycles'].min())
                ]['time_cycles'].min()
                
                if np.isnan(next_stage_start):
                    # If no next stage found, use RUL
                    df.loc[stage_data.index, 'time_to_next_stage'] = stage_data['RUL']
                else:
                    # Calculate cycles until next stage transition
                    df.loc[stage_data.index, 'time_to_next_stage'] = next_stage_start - stage_data['time_cycles']
        
        # Drop rows with missing labels (should be rare)
        df = df.dropna(subset=['time_to_next_stage'])
        
        # Select features (exclude unnecessary columns)
        exclude_cols = ['time_to_next_stage', 'RUL_quartile', 'degradation_stage', 
                        'RUL', 'predicted_stage', 'remaining_cycles', 'normalized_RUL',
                        'cycle_ratio']
        
        # Features are all numeric columns except excluded ones
        feature_cols = [col for col in df.columns if col not in exclude_cols 
                        and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].values
        y = df['time_to_next_stage'].values
        
        return X, y, feature_cols, df
    
    def run_pipeline(self):
        """
        Run the complete regression pipeline for all datasets.
        """
        for dataset in self.datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            # Create dataset output directory
            dataset_output_path = os.path.join(self.output_path, dataset)
            os.makedirs(dataset_output_path, exist_ok=True)
            
            # Load data
            train_df = self.load_dataset(dataset, is_train=True)
            test_df = self.load_dataset(dataset, is_train=False)
            
            # Prepare labels
            X_train, y_train, feature_cols, train_df_with_labels = self.prepare_labels(train_df)
            X_test, y_test, _, test_df_with_labels = self.prepare_labels(test_df)
            
            print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
            print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")
            
            # Save the processed data with labels
            train_df_with_labels.to_csv(os.path.join(dataset_output_path, 'train_with_time_labels.csv'), index=False)
            test_df_with_labels.to_csv(os.path.join(dataset_output_path, 'test_with_time_labels.csv'), index=False)
            
            # Initialize base models
            ridge = RidgeRegression(alpha=1.0)
            rf = RandomForestRegressor(n_estimators=100, max_depth=10)
            quantile_reg = QuantileRegressor(quantile=0.5, alpha=0.1)  # Median quantile
            hist_gbm = HistGradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, max_bins=256)
            
            # Train ensemble model with expanded model set
            ensemble = EnsembleRegressor(base_models=[ridge, rf, quantile_reg, hist_gbm])
            ensemble.fit(X_train, y_train)
            
            # Make predictions
            y_pred = ensemble.predict(X_test)
            
            # Evaluate model
            evaluator = RegressionEvaluator()
            metrics = evaluator.evaluate(y_test, y_pred)
            self.metrics[dataset] = metrics
            
            # For additional insights, get individual model predictions
            individual_preds = ensemble.predict_with_individual_models(X_test)
            
            # Save all model predictions for comparison
            for model_name, preds in individual_preds.items():
                test_df_with_labels[f'pred_{model_name}'] = preds
            
            # Save predictions
            test_df_with_labels.to_csv(os.path.join(dataset_output_path, 'test_predictions.csv'), index=False)
            
            # Create visualizations
            self._create_visualizations(test_df_with_labels, dataset_output_path, dataset)
            self._create_model_comparison_visualizations(test_df_with_labels, dataset_output_path, dataset)
            
        # Save overall metrics
        self._save_metrics()
    
    def _create_visualizations(self, df, output_path, dataset_name):
        """Create visualization for model performance"""
        
        # Actual vs Predicted plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['time_to_next_stage'], df['predicted_time_to_next_stage'], alpha=0.5)
        
        # Add diagonal line (perfect predictions)
        max_val = max(df['time_to_next_stage'].max(), df['predicted_time_to_next_stage'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        
        plt.title(f'Actual vs Predicted Time to Next Stage - {dataset_name}')
        plt.xlabel('Actual Time to Next Stage (cycles)')
        plt.ylabel('Predicted Time to Next Stage (cycles)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'actual_vs_predicted.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Error distribution
        errors = df['predicted_time_to_next_stage'] - df['time_to_next_stage']
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f'Error Distribution - {dataset_name}')
        plt.xlabel('Prediction Error (cycles)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'error_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Error by stage
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='predicted_stage', y=errors, data=df)
        plt.title(f'Error Distribution by Stage - {dataset_name}')
        plt.xlabel('Degradation Stage')
        plt.ylabel('Prediction Error (cycles)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'error_by_stage.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_model_comparison_visualizations(self, df, output_path, dataset_name):
        """Create visualizations comparing performance of different models"""
        
        # Get model columns
        model_cols = [col for col in df.columns if col.startswith('pred_')]
        
        if len(model_cols) > 1:  # Only create comparison if we have multiple models
            # Compare RMSE across models
            model_metrics = {}
            
            for col in model_cols:
                model_name = col.replace('pred_', '')
                rmse = np.sqrt(np.mean((df[col] - df['time_to_next_stage'])**2))
                model_metrics[model_name] = {'RMSE': rmse}
            
            # Create bar chart of model performance
            metrics_df = pd.DataFrame(model_metrics).T
            
            plt.figure(figsize=(10, 6))
            metrics_df['RMSE'].plot(kind='bar', color='skyblue')
            plt.title(f'Model Comparison - {dataset_name}')
            plt.ylabel('RMSE (lower is better)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_path, 'model_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create scatter plot matrix for predictions from each model
            # This can be helpful to see how models differ in their predictions
            cols_to_plot = ['time_to_next_stage'] + model_cols
            plot_df = df[cols_to_plot].sample(min(500, len(df)))  # Sample to avoid overcrowding
            
            # Pairplot if we have 5 or fewer models (becomes unreadable otherwise)
            if len(cols_to_plot) <= 6:  
                sns.pairplot(plot_df, height=2.5)
                plt.suptitle(f'Model Predictions Comparison - {dataset_name}', y=1.02)
                plt.savefig(os.path.join(output_path, 'model_predictions_comparison.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def _save_metrics(self):
        """Save all evaluation metrics to a CSV file"""
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.index.name = 'Dataset'
        metrics_df.to_csv(os.path.join(self.output_path, 'regression_metrics.csv'))
        
        # Create summary plot
        plt.figure(figsize=(12, 6))
        metrics_df[['RMSE', 'MAE']].plot(kind='bar')
        plt.title('Regression Performance Metrics')
        plt.ylabel('Error (cycles)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_path, 'performance_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nRegression metrics summary:")
        print(metrics_df)

if __name__ == "__main__":
    # Run the pipeline
    pipeline = RegressionPipeline()
    pipeline.run_pipeline()