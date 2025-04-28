import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from sklearn.manifold import TSNE
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import CMAPSSDataLoader
from data.preprocessor import CMAPSSPreprocessor
from clustering.hierarchical import (
    HierarchicalAgglomerativeClustering,
    map_clusters_to_degradation_stages,
    visualize_clusters_tsne,
    visualize_degradation_stages,
    analyze_stage_characteristics
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import math
from sklearn.preprocessing import MinMaxScaler
from data.splitter import create_kfold_splits
import time
 
class ClusteringPipeline:
    """
    Pipeline for clustering CMAPSS data using Hierarchical Agglomerative Clustering.
    This pipeline performs:
    1. Data loading
    2. Preprocessing
    3. Hierarchical clustering
    4. Mapping clusters to degradation stages
    5. Visualization and analysis
    6. Performance evaluation against RUL-based ranges
    """
    
    def __init__(self, data_dir: str, dataset_id: str = "FD001",
                n_clusters: int = 5, linkage: str = 'ward',
                distance_metric: str = 'euclidean',
                normalization_method: str = 'minmax',
                verbose: bool = False):
        """
        Initialize the clustering pipeline.
        
        Args:
            data_dir: Directory containing CMAPSS dataset files
            dataset_id: Dataset identifier (FD001, FD002, FD003, or FD004)
            n_clusters: Number of clusters to identify (default: 5 for degradation stages)
            linkage: Linkage criterion to use ('single', 'complete', 'average', 'ward')
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'cosine')
            normalization_method: Method for normalizing data ('minmax', 'standard', 'robust', 'none')
            verbose: Whether to print progress information
        """
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.normalization_method = normalization_method
        self.verbose = verbose
        
        # Initialize components
        self.loader = CMAPSSDataLoader(data_dir)
        self.preprocessor = CMAPSSPreprocessor(normalization_method=normalization_method)
        
        # Storage for results
        self.data = None
        self.preprocessed_data = None
        self.engine_clusters = {}
        self.engine_stages = {}
        self.cluster_analysis = {}
        self.cv_results: List[Dict] = []  # storage for cross-validation results
         
    def load_data(self):
        """
        Load and prepare CMAPSS dataset.
        """
        if self.verbose:
            print(f"Loading dataset {self.dataset_id}...")
            
        # Load dataset
        self.data = self.loader.load_dataset(self.dataset_id)
        
        # Calculate RUL for training data
        train_with_rul = self.loader.calculate_rul_for_training(self.dataset_id)
        
        # Prepare test data with RUL
        test_with_rul = self.loader.prepare_test_with_rul(self.dataset_id)
        
        # Store the data with RUL
        self.data['train_with_rul'] = train_with_rul
        self.data['test_with_rul'] = test_with_rul
        
        if self.verbose:
            print(f"Loaded {len(train_with_rul)} training samples and {len(test_with_rul)} test samples")
        
        return self
    
    def preprocess_data(self, add_engineered_features: bool = True):
        """
        Preprocess the data using the CMAPSSPreprocessor.
        
        Args:
            add_engineered_features: Whether to add engineered features
        """
        if self.data is None:
            self.load_data()
            
        if self.verbose:
            print("Preprocessing data...")
            
        # Fit preprocessor on training data and transform
        train_data = self.data['train_with_rul']
        self.preprocessed_data = {
            'train': self.preprocessor.fit_transform(
                train_data,
                add_remaining_features=add_engineered_features,
                add_sensor_diff=add_engineered_features
            )
        }
        
        # Transform test data
        test_data = self.data['test_with_rul']
        self.preprocessed_data['test'] = self.preprocessor.transform(
            test_data,
            add_remaining_features=add_engineered_features,
            add_sensor_diff=add_engineered_features
        )
        
        if self.verbose:
            print("Data preprocessing complete")
        
        return self
    
    def run_clustering(self, engines: Optional[List[int]] = None, data_subset: str = 'train'):
        """
        Run hierarchical clustering on engine data.
        
        Args:
            engines: List of engine unit numbers to cluster (if None, use all engines)
            data_subset: Which data subset to use ('train' or 'test')
        """
        if self.preprocessed_data is None:
            self.preprocess_data()
            
        # Get the appropriate data subset
        if data_subset not in self.preprocessed_data:
            raise ValueError(f"Data subset '{data_subset}' not found in preprocessed data")
            
        data = self.preprocessed_data[data_subset]
        
        # Get list of engines to process
        if engines is None:
            engines = sorted(data['unit_number'].unique())
        
        # Select features for clustering: raw sensors, engineered diffs, rolling stats, cycle features
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
        diff_cols = [col for col in data.columns if col.endswith('_diff')]
        roll_cols = [col for col in data.columns if 'roll_mean' in col or 'roll_std' in col]
        cycle_cols = [col for col in ['cycle_ratio', 'remaining_cycles'] if col in data.columns]
        feature_cols = sensor_cols + diff_cols + roll_cols + cycle_cols
        if self.verbose:
            print(f"Using {len(feature_cols)} features for clustering: {len(sensor_cols)} sensors, "
                  f"{len(diff_cols)} diffs, {len(roll_cols)} rolling, {len(cycle_cols)} cycle features")

        if self.verbose:
            print(f"Running clustering for {len(engines)} engines using {len(feature_cols)} features")
            print(f"Clustering settings: method=HAC, linkage={self.linkage}, distance={self.distance_metric}")
            
        start_time = time.time()
        
        # Process each engine
        for engine_id in engines:
            if self.verbose:
                print(f"  Processing engine {engine_id}...")
                
            # Extract data for this engine
            engine_data = data[data['unit_number'] == engine_id]
            
            # Skip if not enough data
            if len(engine_data) < self.n_clusters:
                print(f"  WARNING: Engine {engine_id} has only {len(engine_data)} samples, skipping")
                continue
                
            # Get sensor data and time cycles
            X = engine_data[feature_cols].values
            time_cycles = engine_data['time_cycles'].values
            
            # Create and fit clustering model
            hac = HierarchicalAgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                distance_metric=self.distance_metric,
                verbose=self.verbose
            )
            
            # Run clustering
            cluster_labels = hac.fit_predict(X)
            
            # Map clusters to degradation stages
            degradation_stages = map_clusters_to_degradation_stages(cluster_labels, time_cycles)
            
            # Store results
            self.engine_clusters[engine_id] = {
                'model': hac,
                'labels': cluster_labels,
                'data': X,
                'time_cycles': time_cycles,
                'engine_data': engine_data
            }
            
            self.engine_stages[engine_id] = degradation_stages
            
            # Analyze stage characteristics
            stage_stats = analyze_stage_characteristics(X, degradation_stages, feature_cols)
            self.cluster_analysis[engine_id] = stage_stats
            
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Clustering completed in {elapsed_time:.2f} seconds")
        
        return self
    
    def visualize_engine_clusters(self, engine_id: int, save_dir: Optional[str] = None):
        """
        Visualize clustering results for a specific engine.
        
        Args:
            engine_id: Engine unit number to visualize
            save_dir: Directory to save visualizations (if None, display instead)
        """
        if engine_id not in self.engine_clusters:
            raise ValueError(f"No clustering results found for engine {engine_id}")
            
        # Get clustering results
        results = self.engine_clusters[engine_id]
        X = results['data']
        labels = results['labels']
        time_cycles = results['time_cycles']
        stages = self.engine_stages[engine_id]
        
        # Determine save path
        save_path = None
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"engine_{engine_id}_clusters.png")
            
        # Visualize clusters
        visualize_clusters_tsne(
            X, labels, 
            title=f'Hierarchical Clustering Results for Engine {engine_id} - t-SNE Visualization',
            save_path=save_path
        )
        
        # Determine save path for degradation visualization
        deg_save_path = None
        if save_dir is not None:
            deg_save_path = os.path.join(save_dir, f"engine_{engine_id}_degradation.png")
            
        # Visualize degradation stages
        visualize_degradation_stages(
            X, stages, time_cycles,
            title=f'Engine {engine_id} Degradation Stages - t-SNE Visualization',
            save_path=deg_save_path
        )
        
        return self
    
    def visualize_all_engines(self, engines: Optional[List[int]] = None, save_dir: Optional[str] = None):
        """
        Visualize clustering results for multiple engines.
        
        Args:
            engines: List of engine unit numbers to visualize (if None, use all engines)
            save_dir: Directory to save visualizations (if None, display instead)
        """
        if not self.engine_clusters:
            raise ValueError("No clustering results found. Run run_clustering() first.")
            
        # Determine which engines to visualize
        if engines is None:
            engines = sorted(self.engine_clusters.keys())
            
        for engine_id in engines:
            self.visualize_engine_clusters(engine_id, save_dir)
            
        return self
    
    def get_stage_profiles(self):
        """
        Get sensor profiles for each degradation stage across all engines.
        
        Returns:
            DataFrame with stage characteristics
        """
        if not self.cluster_analysis:
            raise ValueError("No cluster analysis found. Run run_clustering() first.")
            
        # Combine stage statistics from all engines
        all_stats = []
        for engine_id, stats in self.cluster_analysis.items():
            engine_stats = stats.copy()
            engine_stats['engine_id'] = engine_id
            all_stats.append(engine_stats)
            
        # Combine into one DataFrame
        combined_stats = pd.concat(all_stats, ignore_index=True)
        
        # Group by stage and calculate aggregate statistics
        stage_profiles = combined_stats.groupby('Stage').agg({
            'Count': 'sum',
            **{col: 'mean' for col in combined_stats.columns 
               if col not in ['Stage', 'Count', 'engine_id'] and col.endswith('_mean')}
        }).reset_index()
        
        return stage_profiles
    
    def create_rul_based_stages(self, num_stages: int = 5) -> Dict[int, np.ndarray]:
        """
        Create ground truth degradation stages based on RUL values.
        
        Args:
            num_stages: Number of degradation stages to create
            
        Returns:
            Dictionary mapping engine IDs to arrays of RUL-based stages
        """
        if self.preprocessed_data is None:
            self.preprocess_data()
            
        rul_based_stages = {}
        train_data = self.preprocessed_data['train']
        
        for engine_id in train_data['unit_number'].unique():
            engine_data = train_data[train_data['unit_number'] == engine_id]
            
            # Get RUL values
            rul_values = engine_data['RUL'].values
            
            # Create equal-sized bins based on RUL values
            # Stage 0: Highest RUL (Normal), Stage 4: Lowest RUL (Near Failure)
            rul_min, rul_max = np.min(rul_values), np.max(rul_values)
            
            # Create stage boundaries
            bin_edges = np.linspace(rul_min, rul_max, num_stages + 1)
            
            # Assign stages based on RUL values (reverse order)
            stages = np.zeros_like(rul_values, dtype=int)
            for i in range(num_stages):
                if i < num_stages - 1:
                    mask = (rul_values >= bin_edges[num_stages - i - 1]) & (rul_values <= bin_edges[num_stages - i])
                else:
                    mask = (rul_values >= bin_edges[0]) & (rul_values <= bin_edges[1])
                stages[mask] = i
            
            rul_based_stages[engine_id] = stages
            
        return rul_based_stages
    
    def calculate_performance_metrics(self, ground_truth: Optional[Dict[int, np.ndarray]] = None) -> Dict[int, Dict[str, float]]:
        """
        Calculate accuracy, precision, recall, and F1 scores for clustering results.
        
        Args:
            ground_truth: Dictionary mapping engine IDs to arrays of true stages.
                         If None, create stages based on RUL values.
                         
        Returns:
            Dictionary mapping engine IDs to dictionaries of performance metrics
        """
        if not self.engine_stages:
            raise ValueError("No clustering results found. Run run_clustering() first.")
            
        # Create RUL-based stages if no ground truth provided
        if ground_truth is None:
            ground_truth = self.create_rul_based_stages(self.n_clusters)
            
        metrics = {}
        
        for engine_id, predicted_stages in self.engine_stages.items():
            if engine_id not in ground_truth:
                if self.verbose:
                    print(f"No ground truth found for engine {engine_id}, skipping")
                continue
                
            true_stages = ground_truth[engine_id]
            
            # Ensure same length
            min_len = min(len(true_stages), len(predicted_stages))
            true_stages = true_stages[:min_len]
            predicted_stages = predicted_stages[:min_len]
            
            # Calculate metrics
            accuracy = accuracy_score(true_stages, predicted_stages)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_stages, predicted_stages, average='weighted'
            )
            
            metrics[engine_id] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
        return metrics
    
    def calculate_gini_index(self, clusters: np.ndarray, rul_ranges: np.ndarray) -> float:
        """
        Calculate Gini impurity index for clustering compared to RUL ranges.
        Lower Gini index indicates better clustering alignment with RUL ranges.
        
        Args:
            clusters: Cluster assignments
            rul_ranges: RUL-based stage assignments
            
        Returns:
            Gini impurity index
        """
        # Get unique clusters
        unique_clusters = np.unique(clusters)
        
        # Calculate Gini index
        gini_sum = 0.0
        total_samples = len(clusters)
        
        for cluster in unique_clusters:
            # Get samples in this cluster
            cluster_mask = clusters == cluster
            cluster_samples = clusters[cluster_mask]
            
            if len(cluster_samples) == 0:
                continue
                
            # Calculate proportion of each RUL range in this cluster
            cluster_gini = 1.0
            
            for rul_range in np.unique(rul_ranges):
                proportion = np.sum((rul_ranges[cluster_mask] == rul_range)) / len(cluster_samples)
                cluster_gini -= proportion ** 2
            
            # Weight by cluster size
            gini_sum += cluster_gini * (len(cluster_samples) / total_samples)
        
        return gini_sum
    
    def calculate_entropy(self, clusters: np.ndarray, rul_ranges: np.ndarray) -> float:
        """
        Calculate Shannon entropy for clustering compared to RUL ranges.
        Lower entropy indicates better clustering alignment with RUL ranges.
        
        Args:
            clusters: Cluster assignments
            rul_ranges: RUL-based stage assignments
            
        Returns:
            Shannon entropy
        """
        # Get unique clusters
        unique_clusters = np.unique(clusters)
        
        # Calculate entropy
        entropy_sum = 0.0
        total_samples = len(clusters)
        
        for cluster in unique_clusters:
            # Get samples in this cluster
            cluster_mask = clusters == cluster
            cluster_samples = clusters[cluster_mask]
            
            if len(cluster_samples) == 0:
                continue
                
            # Calculate entropy for this cluster
            cluster_entropy = 0.0
            
            for rul_range in np.unique(rul_ranges):
                count = np.sum((rul_ranges[cluster_mask] == rul_range))
                if count > 0:
                    proportion = count / len(cluster_samples)
                    cluster_entropy -= proportion * math.log2(proportion)
            
            # Weight by cluster size
            entropy_sum += cluster_entropy * (len(cluster_samples) / total_samples)
        
        return entropy_sum
    
    def evaluate_clustering_quality(self) -> Dict[int, Dict[str, float]]:
        """
        Evaluate clustering quality by comparing with RUL-based ranges.
        
        Returns:
            Dictionary mapping engine IDs to dictionaries of quality metrics
        """
        if not self.engine_stages:
            raise ValueError("No clustering results found. Run run_clustering() first.")
            
        # Create RUL-based stages
        rul_stages = self.create_rul_based_stages(self.n_clusters)
        
        quality_metrics = {}
        
        for engine_id, predicted_stages in self.engine_stages.items():
            if engine_id not in rul_stages:
                if self.verbose:
                    print(f"No RUL stages found for engine {engine_id}, skipping")
                continue
                
            true_stages = rul_stages[engine_id]
            
            # Ensure same length
            min_len = min(len(true_stages), len(predicted_stages))
            true_stages = true_stages[:min_len]
            predicted_stages = predicted_stages[:min_len]
            
            # Calculate metrics
            gini = self.calculate_gini_index(predicted_stages, true_stages)
            entropy = self.calculate_entropy(predicted_stages, true_stages)
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_stages, predicted_stages)
            
            # Normalize confusion matrix by row (true labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Calculate purity and inverse purity
            purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
            inverse_purity = np.sum(np.max(cm, axis=1)) / np.sum(cm)
            
            # Calculate F-measure (harmonic mean of purity and inverse purity)
            f_measure = 2 * (purity * inverse_purity) / (purity + inverse_purity) if (purity + inverse_purity) > 0 else 0
            
            quality_metrics[engine_id] = {
                'gini_index': gini,
                'entropy': entropy,
                'purity': purity,
                'inverse_purity': inverse_purity,
                'f_measure': f_measure
            }
            
        return quality_metrics
    
    def visualize_metrics(self, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Visualize performance and quality metrics across engines.
        """
        # Calculate metrics
        perf = self.calculate_performance_metrics()
        quality = self.evaluate_clustering_quality()
        # Plot performance metrics
        perf_df = pd.DataFrame.from_dict(perf, orient='index')
        plt.figure(figsize=(10, 5))
        perf_df.plot(kind='bar')
        plt.title('Performance Metrics by Engine')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            perf_path = os.path.join(save_dir, f'performance_metrics_plot_{self.dataset_id}.png')
            plt.savefig(perf_path, dpi=300)
            if self.verbose:
                print(f"Saved performance metrics plot to {perf_path}")
        else:
            plt.show()
        plt.close()
        # Plot quality metrics
        qual_df = pd.DataFrame.from_dict(quality, orient='index')
        plt.figure(figsize=(10, 5))
        qual_df.plot(kind='bar')
        plt.title('Quality Metrics by Engine')
        plt.tight_layout()
        if save_dir:
            qual_path = os.path.join(save_dir, f'quality_metrics_plot_{self.dataset_id}.png')
            plt.savefig(qual_path, dpi=300)
            if self.verbose:
                print(f"Saved quality metrics plot to {qual_path}")
        else:
            plt.show()
        plt.close()
        return self
    
    def run_cross_validation(self, n_splits: int = 5) -> List[Dict]:
        """
        Perform k-fold cross-validation over engines, storing performance and quality metrics per fold.
        """
        if self.preprocessed_data is None:
            self.preprocess_data()
        train_df = self.preprocessed_data['train']
        splits = create_kfold_splits(train_df, n_splits=n_splits)
        self.cv_results = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            val_engines = train_df.loc[val_idx, 'unit_number'].unique().tolist()
            # clear previous
            self.engine_clusters.clear(); self.engine_stages.clear(); self.cluster_analysis.clear()
            self.run_clustering(engines=val_engines, data_subset='train')
            perf = self.calculate_performance_metrics()
            quality = self.evaluate_clustering_quality()
            # aggregate
            agg_perf = {m: np.mean([v[m] for v in perf.values()]) for m in next(iter(perf.values())).keys()} if perf else {}
            agg_quality = {m: np.mean([v[m] for v in quality.values()]) for m in next(iter(quality.values())).keys()} if quality else {}
            self.cv_results.append({
                'fold': fold,
                'val_engines': val_engines,
                'performance': perf,
                'quality': quality,
                'agg_performance': agg_perf,
                'agg_quality': agg_quality
            })
        return self.cv_results
    
    def visualize_rul_vs_stages(self, engine_id: int, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Visualize relationship between true RUL-based stages and clustered stages for an engine.
        """
        if engine_id not in self.engine_clusters:
            raise ValueError(f"No clustering results found for engine {engine_id}")
        engine_data = self.engine_clusters[engine_id]['engine_data']
        stages = self.engine_stages[engine_id]
        time_cycles = engine_data['time_cycles'].values
        rul_values = engine_data['RUL'].values
        # ground truth stages
        rul_stages = self.create_rul_based_stages(self.n_clusters)[engine_id]
        fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        # RUL vs time
        axes[0].plot(time_cycles, rul_values, 'b-', linewidth=2)
        axes[0].set_ylabel('RUL', fontsize=12); axes[0].grid(True)
        # clustered stages
        colors = plt.cm.viridis_r(np.linspace(0, 1, self.n_clusters))
        for t, s in zip(time_cycles, stages): axes[1].axvline(t, color=colors[int(s)], alpha=0.5, linewidth=2)
        axes[1].set_ylabel('Cluster Stage', fontsize=12); axes[1].grid(True)
        # RUL-based stages
        for t, s in zip(time_cycles, rul_stages): axes[2].axvline(t, color=colors[int(s)], alpha=0.5, linewidth=2)
        axes[2].set_ylabel('RUL Stage', fontsize=12); axes[2].set_xlabel('Time Cycles'); axes[2].grid(True)
        legend = [plt.Line2D([0],[0], color=colors[i], lw=4, label=f'Stage {i}') for i in range(self.n_clusters)]
        axes[1].legend(handles=legend, loc='upper right')
        axes[2].legend(handles=legend, loc='upper right')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'engine_{engine_id}_rul_vs_stages.png')
            plt.savefig(path, dpi=300)
            if self.verbose: print(f"Saved RUL vs stages to {path}")
        else:
            plt.show()
        plt.close()
        return self
    
    def save_results(self, output_dir: str) -> 'ClusteringPipeline':
        """
        Save clustering results, profiles, and metrics to CSV files.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not self.engine_stages:
            raise ValueError("No clustering results found. Run run_clustering() first.")
        # Combine per-engine results
        results = []
        for engine_id, stages in self.engine_stages.items():
            engine_data = self.engine_clusters[engine_id]['engine_data'].copy()
            engine_data['degradation_stage'] = stages
            results.append(engine_data)
        all_results = pd.concat(results)
        # Save degradation stages
        path_rs = os.path.join(output_dir, f"degradation_stages_{self.dataset_id}.csv")
        all_results.to_csv(path_rs, index=False)
        # Save stage profiles
        profiles = self.get_stage_profiles()
        path_sp = os.path.join(output_dir, f"stage_profiles_{self.dataset_id}.csv")
        profiles.to_csv(path_sp, index=False)
        # Save performance metrics
        perf = self.calculate_performance_metrics()
        df_perf = pd.DataFrame.from_dict(perf, orient='index')
        path_pm = os.path.join(output_dir, f"performance_metrics_{self.dataset_id}.csv")
        df_perf.to_csv(path_pm)
        # Save quality metrics
        qual = self.evaluate_clustering_quality()
        df_qual = pd.DataFrame.from_dict(qual, orient='index')
        path_qm = os.path.join(output_dir, f"clustering_quality_{self.dataset_id}.csv")
        df_qual.to_csv(path_qm)
        if self.verbose:
            print(f"Results saved to {output_dir}")
        return self
    

def main():
    """
    Example usage of the ClusteringPipeline class.
    """
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    output_dir = os.path.join(project_dir, 'results', 'clustering')
    
    # Create pipeline
    pipeline = ClusteringPipeline(
        data_dir=data_dir,
        dataset_id="FD001",
        n_clusters=5,
        linkage='ward',
        distance_metric='euclidean',
        normalization_method='minmax',
        verbose=True
    )
    
    # Run pipeline
    pipeline.load_data()
    pipeline.preprocess_data()
    # Run k-fold cross-validation
    cv_results = pipeline.run_cross_validation(n_splits=5)
    # Save CV aggregated results
    cv_df = pd.DataFrame([{**{'fold': r['fold']}, **r['agg_performance'], **r['agg_quality']} for r in cv_results])
    cv_path = os.path.join(output_dir, f"cv_results_{pipeline.dataset_id}.csv")
    cv_df.to_csv(cv_path, index=False)
    if pipeline.verbose:
        print(f"Cross-validation results saved to {cv_path}")
 
    # Run clustering on the first 5 engines
    engines_to_process = list(range(1, 6))
    pipeline.run_clustering(engines=engines_to_process)
    
    # Visualize results
    pipeline.visualize_all_engines(save_dir=output_dir)
    
    # Visualize RUL vs Stages
    for engine_id in engines_to_process:
        pipeline.visualize_rul_vs_stages(engine_id, save_dir=output_dir)
    
    # Calculate and print performance metrics
    performance_metrics = pipeline.calculate_performance_metrics()
    print("\nPerformance Metrics (against RUL-based stages):")
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
    print(metrics_df)
    
    # Calculate and print clustering quality metrics
    quality_metrics = pipeline.evaluate_clustering_quality()
    print("\nClustering Quality Metrics (against RUL-based stages):")
    quality_df = pd.DataFrame.from_dict(quality_metrics, orient='index')
    print(quality_df)
    # Visualize and save performance/quality metric plots
    pipeline.visualize_metrics(save_dir=output_dir)
    
    # Save results
    pipeline.save_results(output_dir)
    
    # Print stage profiles
    stage_profiles = pipeline.get_stage_profiles()
    print("\nDegradation Stage Profiles:")
    print(stage_profiles[['Stage', 'Count']])
    

if __name__ == "__main__":
    main()