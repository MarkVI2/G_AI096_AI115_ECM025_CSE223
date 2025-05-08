import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Add Code directory to system path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import risk modules
from risk.calibration import RiskCalibrator
from risk.thresholds import RiskThresholdManager
from risk.scoring import RiskScorer

# Import visualization module
from visualisation.risk_dashboard import RiskDashboard

class RiskPipeline:
    """
    Pipeline for computing risk scores and applying decision logic for maintenance alerts.
    
    This pipeline:
    1. Loads prediction results from regression phase
    2. Computes risk scores using failure probability and time to failure
    3. Normalizes scores using min-max or urgency-based methods
    4. Determines maintenance alerts based on thresholds
    5. Visualizes risk patterns and trends
    """
    
    def __init__(self, datasets=['FD001', 'FD002', 'FD003', 'FD004'], 
                 base_path=None, n_jobs=None, risk_method='urgency',
                 high_risk_threshold=0.7, medium_risk_threshold=0.4,
                 failure_stage=4):
        """
        Initialize the risk pipeline.
        
        Args:
            datasets: List of dataset names to process
            base_path: Base path to the project
            n_jobs: Number of parallel jobs to run
            risk_method: Method for computing risk ('minmax' or 'urgency')
            high_risk_threshold: Threshold for high risk alerts
            medium_risk_threshold: Threshold for medium risk alerts
            failure_stage: Stage number considered as failure (default: 4)
        """
        self.datasets = datasets
        self.risk_method = risk_method
        self.failure_stage = failure_stage
        
        if base_path is None:
            # Determine the base path automatically
            current_file = os.path.abspath(__file__)
            base_path = os.path.dirname(os.path.dirname(current_file))
        
        self.base_path = base_path
        self.results_path = os.path.join(base_path, 'results')
        self.output_path = os.path.join(self.results_path, 'risk')
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Number of parallel workers
        self.n_jobs = n_jobs or os.cpu_count()
        
        # Initialize components
        if risk_method == 'calibration':
            self.risk_component = RiskCalibrator(method='minmax', failure_stage=failure_stage)
        else:
            self.risk_component = RiskScorer(failure_stage=failure_stage)
            
        self.threshold_manager = RiskThresholdManager(
            high_risk_threshold=high_risk_threshold,
            medium_risk_threshold=medium_risk_threshold
        )
        
        # Dashboard for visualization
        self.dashboard = RiskDashboard()
        
        # Store metrics for comparison
        self.metrics = {}
        
    def load_regression_results(self, dataset):
        """
        Load regression results for a dataset.
        
        Args:
            dataset: Dataset name (e.g., 'FD001')
            
        Returns:
            DataFrame with regression predictions
        """
        results_path = os.path.join(self.results_path, 'regression', dataset, 'test_predictions.csv')
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Regression results not found for {dataset} at {results_path}")
            
        return pd.read_csv(results_path)
    
    def compute_risk_scores(self, df):
        """
        Calculate risk scores for each data point.
        
        Args:
            df: DataFrame with regression predictions
            
        Returns:
            DataFrame with added risk scores
        """
        if self.risk_method == 'calibration':
            # Use the calibration-based risk scorer
            return self.risk_component.fit_transform(df)
        else:
            # Use the simpler risk scorer
            df_with_scores = self.risk_component.calculate_risk_scores(df)
            
            # Apply dynamic thresholds based on the score distribution
            if 'normalized_urgency' in df_with_scores.columns:
                # Use the normalized_urgency column for threshold calculation
                self.threshold_manager.get_dynamic_thresholds(df_with_scores['normalized_urgency'].values)
            elif 'urgency_score' in df_with_scores.columns:
                # Use the urgency_score column for threshold calculation
                self.threshold_manager.get_dynamic_thresholds(df_with_scores['urgency_score'].values)
                
            return df_with_scores
    
    def determine_risk_levels(self, df):
        """
        Determine risk level for each data point.
        
        Args:
            df: DataFrame with risk scores
            
        Returns:
            DataFrame with added risk levels
        """
        result_df = df.copy()
        
        # Determine which score column to use for risk level assignment
        # Prefer normalized_urgency if available, otherwise fall back to other options
        if 'normalized_urgency' in result_df.columns:
            score_col = 'normalized_urgency'
        elif 'normalized_risk_score' in result_df.columns:
            score_col = 'normalized_risk_score'
        elif 'urgency_score' in result_df.columns:
            score_col = 'urgency_score'
        else:
            raise ValueError("No normalized risk score column found in DataFrame")
            
        # Ensure all scores are positive
        result_df[score_col] = result_df[score_col].apply(lambda x: max(0, float(x)))
        
        # Add risk level based on normalized score
        result_df['risk_level'] = result_df[score_col].apply(self.threshold_manager.get_risk_level)
        
        # Add maintenance alert flag
        result_df['maintenance_alert'] = result_df['risk_level'].apply(
            lambda x: x == 'HIGH'
        )
        
        # Log threshold values used for this dataset
        print(f"Risk thresholds used: HIGH={self.threshold_manager.high_risk_threshold:.3f}, "
              f"MEDIUM={self.threshold_manager.medium_risk_threshold:.3f}, "
              f"LOW={self.threshold_manager.low_risk_threshold:.3f}")
        
        # Print summary of risk levels
        risk_counts = result_df['risk_level'].value_counts()
        print("\nRisk level distribution:")
        for level, count in risk_counts.items():
            print(f"  {level}: {count} ({count/len(result_df)*100:.1f}%)")
            
        return result_df
    
    def process_dataset(self, dataset):
        """
        Process a single dataset: load data, compute risk scores, determine risk levels,
        visualize, and save results.
        
        Args:
            dataset: Dataset name
            
        Returns:
            Dataset name and metrics dict
        """
        print(f"\nProcessing risk analysis for dataset: {dataset}")
        
        # Paths
        dataset_output_path = os.path.join(self.output_path, dataset)
        os.makedirs(dataset_output_path, exist_ok=True)
        
        # Load prediction data
        try:
            df = self.load_regression_results(dataset)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return dataset, {}
            
        # Calculate risk scores
        df_with_risk = self.compute_risk_scores(df)
        
        # Determine risk levels and alerts
        df_with_risk = self.determine_risk_levels(df_with_risk)
        
        # Save results
        df_with_risk.to_csv(os.path.join(dataset_output_path, 'risk_assessment.csv'), index=False)
        
        # Create visualizations
        self._create_visualizations(df_with_risk, dataset_output_path, dataset)
        
        # Compute metrics
        metrics = self._calculate_metrics(df_with_risk)
        self.metrics[dataset] = metrics
        
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(dataset_output_path, 'risk_metrics.csv'), index=False)
        
        return dataset, metrics
    
    def run_pipeline(self):
        """
        Run the complete risk pipeline across all datasets.
        
        Returns:
            Dictionary with results for each dataset
        """
        print(f"Running risk analysis pipeline with method: {self.risk_method}")
        
        # Parallel processing for each dataset
        if self.n_jobs > 1 and len(self.datasets) > 1:
            results = {}
            with ProcessPoolExecutor(max_workers=min(self.n_jobs, len(self.datasets))) as executor:
                future_to_dataset = {
                    executor.submit(self.process_dataset, dataset): dataset 
                    for dataset in self.datasets
                }
                
                for future in as_completed(future_to_dataset):
                    dataset = future_to_dataset[future]
                    try:
                        ds, metrics = future.result()
                        results[ds] = metrics
                    except Exception as e:
                        print(f"Error processing {dataset}: {e}")
        else:
            results = {}
            for dataset in self.datasets:
                ds, metrics = self.process_dataset(dataset)
                results[ds] = metrics
                
        # Save aggregated metrics
        self._save_metrics()
        
        return results
    
    def _create_visualizations(self, df, output_path, dataset_name):
        """
        Create visualizations for risk analysis.
        
        Args:
            df: DataFrame with risk data
            output_path: Path to save visualizations
            dataset_name: Name of the dataset
        """
        # Risk score distribution
        plt.figure(figsize=(10, 6))
        if 'normalized_risk_score' in df.columns:
            sns.histplot(df['normalized_risk_score'], kde=True)
            plt.title(f'Risk Score Distribution - {dataset_name}')
            plt.xlabel('Normalized Risk Score')
        elif 'urgency_score' in df.columns:
            sns.histplot(df['urgency_score'], kde=True)
            plt.title(f'Urgency-Based Risk Score Distribution - {dataset_name}')
            plt.xlabel('Urgency Risk Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'risk_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Risk level distribution
        plt.figure(figsize=(10, 6))
        risk_counts = df['risk_level'].value_counts().sort_index()
        sns.barplot(x=risk_counts.index, y=risk_counts.values)
        plt.title(f'Risk Level Distribution - {dataset_name}')
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'risk_level_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Risk score by unit
        plt.figure(figsize=(12, 6))
        unit_risk = df.groupby('unit_number')['normalized_risk_score' if 'normalized_risk_score' in df.columns else 'urgency_score'].mean().sort_values(ascending=False)
        sns.barplot(x=unit_risk.index[:20], y=unit_risk.values[:20])  # Top 20 units by risk
        plt.title(f'Average Risk Score by Unit (Top 20) - {dataset_name}')
        plt.xlabel('Unit Number')
        plt.ylabel('Average Risk Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'risk_by_unit.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Risk score over time for select units
        plt.figure(figsize=(12, 8))
        # Select a few sample units
        sample_units = sorted(df['unit_number'].unique())[:5]
        for unit in sample_units:
            unit_data = df[df['unit_number'] == unit].sort_values('time_cycles')
            score_col = 'normalized_risk_score' if 'normalized_risk_score' in df.columns else 'urgency_score'
            plt.plot(unit_data['time_cycles'], unit_data[score_col], label=f'Unit {unit}')
        
        plt.axhline(y=self.threshold_manager.high_risk_threshold, color='r', linestyle='--', label='High Risk Threshold')
        plt.axhline(y=self.threshold_manager.medium_risk_threshold, color='y', linestyle='--', label='Medium Risk Threshold')
        
        plt.title(f'Risk Score Trend Over Time - {dataset_name}')
        plt.xlabel('Time Cycles')
        plt.ylabel('Risk Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'risk_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Risk score vs. predicted time to next stage
        plt.figure(figsize=(10, 6))
        score_col = 'normalized_risk_score' if 'normalized_risk_score' in df.columns else 'urgency_score'
        plt.scatter(df['predicted_time_to_next_stage'], df[score_col], alpha=0.5)
        plt.title(f'Risk Score vs Predicted Time to Next Stage - {dataset_name}')
        plt.xlabel('Predicted Time to Next Stage (cycles)')
        plt.ylabel('Risk Score')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_path, 'risk_vs_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _calculate_metrics(self, df):
        """
        Calculate metrics to evaluate risk assessment.
        
        Args:
            df: DataFrame with risk data
            
        Returns:
            Dict with calculated metrics
        """
        # Basic statistics
        score_col = 'normalized_risk_score' if 'normalized_risk_score' in df.columns else 'urgency_score'
        
        metrics = {
            'mean_risk_score': float(df[score_col].mean()),
            'median_risk_score': float(df[score_col].median()),
            'std_risk_score': float(df[score_col].std()),
            'min_risk_score': float(df[score_col].min()),
            'max_risk_score': float(df[score_col].max()),
            'high_risk_threshold': float(self.threshold_manager.high_risk_threshold),
            'medium_risk_threshold': float(self.threshold_manager.medium_risk_threshold),
            'units_with_high_risk': int(df.groupby('unit_number')['risk_level'].apply(lambda x: 'HIGH' in x.values).sum()),
            'total_units': int(df['unit_number'].nunique()),
            'total_high_risk_alerts': int((df['risk_level'] == 'HIGH').sum()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Calculate percentage of high risk units
        metrics['percentage_high_risk_units'] = (metrics['units_with_high_risk'] / metrics['total_units']) * 100
        
        return metrics
    
    def _save_metrics(self):
        """
        Save all evaluation metrics to a CSV file.
        """
        all_metrics = []
        for dataset, metrics in self.metrics.items():
            metrics_row = {'dataset': dataset}
            metrics_row.update(metrics)
            all_metrics.append(metrics_row)
            
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df.to_csv(os.path.join(self.output_path, 'all_metrics.csv'), index=False)
            
            # Create comparison visualizations
            self._create_comparison_visualizations()
    
    def _create_comparison_visualizations(self):
        """
        Create visualizations comparing risk metrics across datasets.
        """
        if not self.metrics:
            return
            
        metrics_df = pd.DataFrame([
            {'dataset': dataset, **metrics} 
            for dataset, metrics in self.metrics.items()
        ])
        
        # Mean risk score by dataset
        plt.figure(figsize=(10, 6))
        sns.barplot(x='dataset', y='mean_risk_score', data=metrics_df)
        plt.title('Average Risk Score by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Mean Risk Score')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_path, 'mean_risk_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Percentage of high risk units by dataset
        plt.figure(figsize=(10, 6))
        sns.barplot(x='dataset', y='percentage_high_risk_units', data=metrics_df)
        plt.title('Percentage of High Risk Units by Dataset')
        plt.xlabel('Dataset')
        plt.ylabel('Percentage of Units (%)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.output_path, 'high_risk_units_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run risk analysis pipeline')
    parser.add_argument('--datasets', type=str, default='FD001,FD002,FD003,FD004',
                      help='Comma-separated list of datasets to process')
    parser.add_argument('--method', type=str, default='urgency', choices=['urgency', 'minmax', 'calibration'],
                      help='Risk calculation method')
    parser.add_argument('--high_threshold', type=float, default=0.7,
                      help='Threshold for high risk alerts')
    parser.add_argument('--medium_threshold', type=float, default=0.4,
                      help='Threshold for medium risk alerts')
    parser.add_argument('--n_jobs', type=int, default=None,
                      help='Number of parallel jobs. Default: use all CPUs')
    args = parser.parse_args()
    
    # Run the pipeline
    pipeline = RiskPipeline(
        datasets=args.datasets.split(','),
        risk_method=args.method,
        high_risk_threshold=args.high_threshold,
        medium_risk_threshold=args.medium_threshold,
        n_jobs=args.n_jobs
    )
    results = pipeline.run_pipeline()
    
    print("\nRisk analysis pipeline completed successfully!")

def run_risk_assessment(classification_results=None, regression_results=None, data_path=None, 
                        n_jobs=None, risk_method='urgency', high_threshold=0.7, medium_threshold=0.4):
    """
    Run the risk assessment pipeline and return the results.
    
    Args:
        classification_results: Results from the classification pipeline
        regression_results: Results from the regression pipeline
        data_path: Path to the data directory
        n_jobs: Number of parallel jobs to run
        risk_method: Method for computing risk ('minmax', 'urgency', or 'calibration')
        high_threshold: Threshold for high risk alerts
        medium_threshold: Threshold for medium risk alerts
        
    Returns:
        Dictionary containing risk assessment results
    """
    # Setup datasets based on regression results if provided
    if regression_results:
        datasets = list(regression_results.keys())
    else:
        datasets = ['FD001', 'FD003']  # Default datasets
        
    # Initialize and run pipeline
    pipeline = RiskPipeline(
        datasets=datasets,
        base_path=data_path,
        n_jobs=n_jobs,
        risk_method=risk_method,
        high_risk_threshold=high_threshold,
        medium_risk_threshold=medium_threshold
    )
    
    # Run the pipeline
    results = pipeline.run_pipeline()
    
    return results