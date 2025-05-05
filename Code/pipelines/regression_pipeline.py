import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add Code directory to system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from regression.base_models import RidgeRegression, RandomForestRegressor
from regression.ensemble_regressor import EnsembleRegressor
from regression.evaluator import RegressionEvaluator
from regression.quantile import QuantileRegressor
from regression.gradient_boosting import HistGradientBoostingRegressor

class RegressionPipeline:
    def __init__(self, datasets=['FD001', 'FD002', 'FD003', 'FD004'], 
                 classifier='random_forest', base_path=None, n_jobs=None):
        """
        Initialize the regression pipeline.
        Args:
            datasets: List of dataset names to process
            classifier: The classifier used for stage predictions
            base_path: Base path to the project
            n_jobs: Number of parallel jobs to run
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
        # Number of parallel workers
        self.n_jobs = n_jobs or os.cpu_count()
        
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
        """
        # Sort by unit and time for time series processing
        df = df.sort_values(['unit_number', 'time_cycles'])
        # Initialize the target column
        df['time_to_next_stage'] = np.nan
        # Determine last stage dynamically
        max_stage = int(df['predicted_stage'].max())
        # Process each unit separately
        for unit in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit]
            # Iterate over each observation
            for idx, row in unit_data.iterrows():
                current_stage = int(row['predicted_stage'])
                # If last stage or beyond, use RUL
                if current_stage >= max_stage:
                    time_val = row['RUL']
                else:
                    # Find next stage start time strictly after this cycle
                    future = unit_data[
                        (unit_data['predicted_stage'] == current_stage + 1) &
                        (unit_data['time_cycles'] > row['time_cycles'])
                    ]['time_cycles']
                    if future.empty:
                        time_val = row['RUL']
                    else:
                        next_time = future.min()
                        time_val = next_time - row['time_cycles']
                # Ensure no negative time
                df.at[idx, 'time_to_next_stage'] = max(0, time_val)
        # Drop rows with missing labels (should be rare)
        df = df.dropna(subset=['time_to_next_stage'])
         
        # Select only essential features and drop any rolling-statistics columns
        feature_cols = []
        for col in df.columns:
            if 'roll' in col:
                continue
            if col in ['unit_number', 'time_cycles', 'RUL_quartile', 'degradation_stage']:
                feature_cols.append(col)
            elif col.startswith('setting_'):
                feature_cols.append(col)
            elif col.startswith('sensor_') and not col.endswith('_diff'):
                feature_cols.append(col)

        X = df[feature_cols].values
        y = df['time_to_next_stage'].values
        
        return X, y, feature_cols, df
    
    def run_pipeline(self):
        """
        Run the complete regression pipeline in parallel across datasets.
        """
        # Use process-based parallelism for CPU-bound training
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {executor.submit(self._process_single, ds): ds for ds in self.datasets}
            for future in as_completed(futures):
                dataset, metrics = future.result()
                self.metrics[dataset] = metrics
        # Save all metrics after parallel execution
        self._save_metrics()

    def _process_single(self, dataset):
        """
        Process a single dataset: load data, prepare labels, train models, evaluate, and save results.
        Returns dataset name and metrics dict.
        """
        print(f"\nProcessing dataset: {dataset}")
        # Paths and data loading
        dataset_output_path = os.path.join(self.output_path, dataset)
        os.makedirs(dataset_output_path, exist_ok=True)
        train_df = self.load_dataset(dataset, is_train=True)
        test_df = self.load_dataset(dataset, is_train=False)
        # Label preparation
        X_train, y_train, feature_cols, train_df = self.prepare_labels(train_df)
        X_test, y_test, feature_cols, test_df = self.prepare_labels(test_df)
        # Save processed
        core_feature_cols = [
            col for col in feature_cols if 
            not any(suffix in col for suffix in ['_cummean', '_cumstd', '_diff'])
        ]
        save_cols = core_feature_cols + ['predicted_stage', 'time_to_next_stage']
        train_df[save_cols].to_csv(
            os.path.join(dataset_output_path, 'train_with_time_labels.csv'), index=False)
        test_df[save_cols].to_csv(
            os.path.join(dataset_output_path, 'test_with_time_labels.csv'), index=False)
        # Model setup
        ridge = RidgeRegression(alpha=1.0)
        rf = RandomForestRegressor(n_estimators=100, max_depth=10)
        quantile_reg = QuantileRegressor(quantile=0.5, alpha=0.1)
        hist_gbm = HistGradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, max_bins=256)
        ensemble = EnsembleRegressor(base_models=[ridge, rf, quantile_reg, hist_gbm])
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        # Evaluation
        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred)
        # Save predictions and visualizations
        test_df['predicted_time_to_next_stage'] = y_pred
        # also individual preds
        ind_preds = ensemble.predict_with_individual_models(X_test)
        for name, preds in ind_preds.items(): test_df[f'pred_{name}'] = preds
        # Save only essential columns in final predictions
        pred_cols = core_feature_cols + ['predicted_stage', 'time_to_next_stage', 'predicted_time_to_next_stage']
        pred_cols += [col for col in test_df.columns if col.startswith('pred_') and col not in pred_cols]
        test_df[pred_cols].to_csv(
            os.path.join(dataset_output_path, 'test_predictions.csv'), index=False)
        self._create_visualizations(test_df, dataset_output_path, dataset)
        self._create_model_comparison_visualizations(test_df, dataset_output_path, dataset)
        return dataset, metrics
    
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