import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # heatmap support
from mpl_toolkits.mplot3d import Axes3D  # 3D plotting
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
    
    def compute_average_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute average performance and quality metrics across engines.
        """
        perf = self.calculate_performance_metrics()
        qual = self.evaluate_clustering_quality()
        # average performance
        avg_perf = {}
        if perf:
            keys = next(iter(perf.values())).keys()
            for k in keys:
                avg_perf[k] = np.mean([v[k] for v in perf.values()])
        # average quality
        avg_qual = {}
        if qual:
            keys = next(iter(qual.values())).keys()
            for k in keys:
                avg_qual[k] = np.mean([v[k] for v in qual.values()])
        return {'average_performance': avg_perf, 'average_quality': avg_qual}

    def visualize_average_clusters(self, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Run t-SNE on combined engine data and plot aggregated cluster and RUL stage distributions.
        """
        # Combine data across engines
        X_all = np.vstack([info['data'] for info in self.engine_clusters.values()])
        labels = np.hstack([self.engine_stages[e] for e in self.engine_clusters.keys()])
        # ground truth RUL stages
        rul_all = np.hstack([self.create_rul_based_stages(self.n_clusters)[e] for e in self.engine_clusters.keys()])
        # compute t-SNE embedding
        tsne_embed = TSNE(n_components=2, random_state=0).fit_transform(X_all)
        # plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        # clusters
        scatter1 = ax1.scatter(tsne_embed[:,0], tsne_embed[:,1], c=labels, cmap='viridis', s=5)
        ax1.set_title('Aggregated t-SNE: Cluster Labels')
        # RUL stages
        scatter2 = ax2.scatter(tsne_embed[:,0], tsne_embed[:,1], c=rul_all, cmap='plasma', s=5)
        ax2.set_title('Aggregated t-SNE: RUL-based Stages')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'average_clusters_{self.dataset_id}.png')
            fig.savefig(path, dpi=300)
            if self.verbose:
                print(f"Saved aggregated cluster plot to {path}")
        else:
            plt.show()
        plt.close(fig)
        return self
    
    def visualize_confusion_matrix_heatmap(self, engine_id: int, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Plot a heatmap of the confusion matrix between RUL-based and clustered stages for an engine.
        """
        if engine_id not in self.engine_stages:
            raise ValueError(f"No clustering results for engine {engine_id}")
        # True and predicted
        true = self.create_rul_based_stages(self.n_clusters)[engine_id]
        pred = self.engine_stages[engine_id]
        # align lengths
        n = min(len(true), len(pred))
        cm = confusion_matrix(true[:n], pred[:n])
        # plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Stage')
        plt.ylabel('True RUL Stage')
        plt.title(f'Engine {engine_id} Confusion Matrix Heatmap')
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'engine_{engine_id}_cm_heatmap.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            if self.verbose: print(f"Saved confusion matrix heatmap to {path}")
        else:
            plt.show()
        plt.close()
        return self

    def visualize_3d_embedding(self, engine_id: int, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Generate a 3D t-SNE embedding of sensor features for an engine, colored by cluster label.
        """
        if engine_id not in self.engine_clusters:
            raise ValueError(f"No clustering results for engine {engine_id}")
        info = self.engine_clusters[engine_id]
        X = info['data']
        labels = info['labels']
        # 3D t-SNE
        embed3d = TSNE(n_components=3, random_state=0).fit_transform(X)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embed3d[:,0], embed3d[:,1], embed3d[:,2], c=labels, cmap='viridis', s=5)
        ax.set_title(f'Engine {engine_id} 3D t-SNE Clusters')
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.set_zlabel('Dim 3')
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'engine_{engine_id}_3d_tsne.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            if self.verbose: print(f"Saved 3D t-SNE embedding to {path}")
        else:
            plt.show()
        plt.close(fig)
        return self
    
    def visualize_average_clusters_3d(self, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Create a 3D t-SNE embedding for all engines combined, colored by cluster labels.
        """
        # Combine
        X_all = np.vstack([info['data'] for info in self.engine_clusters.values()])
        labels = np.hstack([self.engine_stages[e] for e in self.engine_clusters.keys()])
        embed3d = TSNE(n_components=3, random_state=0).fit_transform(X_all)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(embed3d[:,0], embed3d[:,1], embed3d[:,2], c=labels, cmap='viridis', s=5)
        ax.set_title(f'Aggregated 3D t-SNE Clusters ({self.dataset_id})')
        ax.set_xlabel('Dim 1'); ax.set_ylabel('Dim 2'); ax.set_zlabel('Dim 3')
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f'average_clusters_3d_{self.dataset_id}.png')
            plt.savefig(path, dpi=300, bbox_inches='tight')
            if self.verbose: print(f"Saved averaged 3D cluster plot to {path}")
        else:
            plt.show()
        plt.close(fig)
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
            # reset per-engine
            self.engine_clusters.clear(); self.engine_stages.clear(); self.cluster_analysis.clear()
            # train on folds
            self.run_clustering(engines=val_engines, data_subset='train')
            perf = self.calculate_performance_metrics()
            qual = self.evaluate_clustering_quality()
            # compute aggregate metrics
            agg_perf = {m: np.mean([v[m] for v in perf.values()]) for m in next(iter(perf.values())).keys()} if perf else {}
            agg_qual = {m: np.mean([v[m] for v in qual.values()]) for m in next(iter(qual.values())).keys()} if qual else {}
            # store
            self.cv_results.append({
                'fold': fold,
                'val_engines': val_engines,
                'performance': perf,
                'quality': qual,
                'agg_performance': agg_perf,
                'agg_quality': agg_qual
            })
        return self.cv_results
    
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
            df = self.engine_clusters[engine_id]['engine_data'].copy()
            df['degradation_stage'] = stages
            results.append(df)
        all_df = pd.concat(results)
        # Save degradation stages
        all_df.to_csv(os.path.join(output_dir, f"degradation_stages_{self.dataset_id}.csv"), index=False)
        # Save stage profiles
        self.get_stage_profiles().to_csv(os.path.join(output_dir, f"stage_profiles_{self.dataset_id}.csv"), index=False)
        # Save performance metrics
        perf = self.calculate_performance_metrics()
        pd.DataFrame.from_dict(perf, orient='index').to_csv(os.path.join(output_dir, f"performance_metrics_{self.dataset_id}.csv"))
        # Save quality metrics
        qual = self.evaluate_clustering_quality()
        pd.DataFrame.from_dict(qual, orient='index').to_csv(os.path.join(output_dir, f"clustering_quality_{self.dataset_id}.csv"))
        if self.verbose:
            print(f"Results saved to {output_dir}")
        return self
    

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    output_dir = os.path.join(project_dir, 'results', 'clustering')
    
    # Process all four CMAPSS datasets
    dataset_ids = ["FD001", "FD002", "FD003", "FD004"]
    for ds in dataset_ids:
        ds_output = os.path.join(output_dir, ds)
        os.makedirs(ds_output, exist_ok=True)
        pipeline = ClusteringPipeline(
            data_dir=data_dir,
            dataset_id=ds,
            n_clusters=5,
            linkage='ward',
            distance_metric='euclidean',
            normalization_method='minmax',
            verbose=True
        )
        # Load and preprocess
        pipeline.load_data().preprocess_data()
        # Cross-validation
        cv_results = pipeline.run_cross_validation(n_splits=5)
        cv_df = pd.DataFrame([{**{'fold': r['fold']}, **r['agg_performance'], **r['agg_quality']} for r in cv_results])
        cv_df.to_csv(os.path.join(ds_output, f"cv_results_{ds}.csv"), index=False)
        # Full engine clustering
        pipeline.run_clustering()
        # Compute and print average metrics
        avg = pipeline.compute_average_metrics()
        print(f"\nDataset {ds} - Average Metrics:")
        print(pd.DataFrame(avg))
        # Save average metrics
        avg_df = pd.DataFrame.from_dict(avg, orient='index')
        avg_df.to_csv(os.path.join(ds_output, f"average_metrics_{ds}.csv"))
        # Visualizations
        pipeline.visualize_all_engines(save_dir=ds_output)
        # Per-engine confusion heatmaps and 3D embeddings
        for engine_id in pipeline.engine_stages.keys():
            try:
                pipeline.visualize_confusion_matrix_heatmap(engine_id, save_dir=ds_output)
            except Exception as e:
                if pipeline.verbose:
                    print(f"Warning: heatmap failed for engine {engine_id}: {e}")
            try:
                pipeline.visualize_3d_embedding(engine_id, save_dir=ds_output)
            except Exception as e:
                if pipeline.verbose:
                    print(f"Warning: 3D embedding failed for engine {engine_id}: {e}")
        pipeline.visualize_average_clusters(save_dir=ds_output)
        # Aggregated 3D t-SNE over all engines
        pipeline.visualize_average_clusters_3d(save_dir=ds_output)
        pipeline.visualize_metrics(save_dir=ds_output)
        # Save full results
        pipeline.save_results(ds_output)
        # Stage profiles
        sp = pipeline.get_stage_profiles()
        sp.to_csv(os.path.join(ds_output, f"stage_profiles_{ds}.csv"), index=False)
        print(f"Completed dataset {ds}, results in {ds_output}")
    
    # End processing of all datasets

if __name__ == "__main__":
    main()