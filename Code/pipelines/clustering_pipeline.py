import os
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional
import sys
from joblib import Parallel, delayed

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import CMAPSSDataLoader
from data.preprocessor import CMAPSSPreprocessor
from clustering.hierarchical import (
    HierarchicalAgglomerativeClustering,
    map_clusters_to_degradation_stages,
    analyze_stage_characteristics
)
from data.splitter import create_kfold_splits
from visualisation.clustering_viz import (
    plot_engine_clusters,
    plot_confusion_matrix_heatmap,
    plot_3d_embedding,
    plot_average_clusters,
    plot_average_clusters_3d,
    plot_metrics
)
from utils.clustering_metrics import (
    calculate_performance_metrics,
    evaluate_clustering_quality,
    compute_average_metrics
)

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
                scale_per_unit: bool = False,
                verbose: bool = False,
                random_state: int = 42):
        """
        Initialize the clustering pipeline.
        
        Args:
            data_dir: Directory containing CMAPSS dataset files
            dataset_id: Dataset identifier (FD001, FD002, FD003, or FD004)
            n_clusters: Number of clusters to identify (default: 5 for degradation stages)
            linkage: Linkage criterion to use ('single', 'complete', 'average', 'ward')
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'cosine')
            normalization_method: Method for normalizing data ('minmax', 'standard', 'robust', 'none')
            scale_per_unit: Whether to apply scaling per engine unit
            verbose: Whether to print progress information
        """
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.normalization_method = normalization_method
        self.verbose = verbose
        # Reproducibility seed
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        # Initialize components
        self.loader = CMAPSSDataLoader(data_dir)
        self.scale_per_unit = scale_per_unit
        # Initialize preprocessor with optional per-unit scaling
        self.preprocessor = CMAPSSPreprocessor(
            normalization_method=normalization_method,
            scale_per_unit=self.scale_per_unit
        )
        
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
        # Fit and transform training data
        transformed_train = self.preprocessor.fit_transform(
            train_data,
            add_remaining_features=add_engineered_features,
            add_sensor_diff=add_engineered_features
        )
        # Sub-sample ~60% from each engine (unit_number) for reproducible training split
        sampled_train = (
            transformed_train
            .groupby('unit_number', group_keys=False)
            .apply(lambda d: d.sample(frac=0.6, random_state=self.random_state))
            .reset_index(drop=True)
        )
        self.preprocessed_data = {'train': sampled_train}
        
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
        
        # Select features for clustering: include time_cycles, operational settings, and raw sensors
        op_cols = [col for col in data.columns if col.startswith('setting_')]
        sensor_cols = [col for col in data.columns if col.startswith('sensor_')]
        feature_cols = ['time_cycles'] + op_cols + sensor_cols
        if self.verbose:
            print(f"Using {len(feature_cols)} features (time_cycles + settings + sensors) for clustering")

        if self.verbose:
            print(f"Running clustering for {len(engines)} engines using {len(feature_cols)} features")
            print(f"Clustering settings: method=HAC, linkage={self.linkage}, distance={self.distance_metric}")
            
        start_time = time.time()
        # Parallel processing of engines
        def _process(eid):
            engine_data = data[data['unit_number'] == eid]
            if len(engine_data) < self.n_clusters:
                return None
            X_e = engine_data[feature_cols].values
            # Debug: Show scaling stats for first engine in verbose mode
            if self.verbose:
                mins = X_e.min(axis=0)
                maxs = X_e.max(axis=0)
                means = X_e.mean(axis=0)
                stds = X_e.std(axis=0)
                print(f"Engine {eid} feature stats: min={mins[:3]}, max={maxs[:3]}, mean={means[:3]}, std={stds[:3]}")
            tc_e = engine_data['time_cycles'].values
            model = HierarchicalAgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage=self.linkage,
                distance_metric=self.distance_metric,
                verbose=self.verbose
            )
            labels_e = model.fit_predict(X_e)
            stages_e = map_clusters_to_degradation_stages(labels_e, tc_e)
            stats_e = analyze_stage_characteristics(X_e, stages_e, feature_cols)
            return eid, model, labels_e, stages_e, stats_e, engine_data, tc_e, X_e
        results = Parallel(n_jobs=-1)(delayed(_process)(eid) for eid in engines)
        for item in results:
            if item is None:
                continue
            eid, model, labels_e, stages_e, stats_e, edata, tc_e, X_e = item
            self.engine_clusters[eid] = {
                'model': model,
                'labels': labels_e,
                'data': X_e,
                'time_cycles': tc_e,
                'engine_data': edata
            }
            self.engine_stages[eid] = stages_e
            self.cluster_analysis[eid] = stats_e
            
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
        stages = self.engine_stages[engine_id]
        
        # Visualize with the extracted visualization function
        plot_engine_clusters(engine_id, results, stages, save_dir)
        
        # Visualize confusion matrix
        true_stages = self.create_rul_based_stages(self.n_clusters)[engine_id]
        plot_confusion_matrix_heatmap(engine_id, true_stages, stages, save_dir)
        
        # Visualize 3D embedding
        plot_3d_embedding(engine_id, results['data'], results['labels'], save_dir)
        
        # Sensor trend plot by degradation stage (e.g., vibration = sensor_3)
        from visualisation.clustering_viz import plot_sensor_trends_by_stage
        plot_sensor_trends_by_stage(
            engine_id,
            results['engine_data'],
            stages,
            sensor_col='sensor_3',
            save_dir=save_dir
        )
        
        return self
    
    def visualize_all_engines(self, engines: Optional[List[int]] = None, save_dir: Optional[str] = None, interval: int = 49):
        """
        Visualize clustering results for multiple engines.
        
        Args:
            engines: List of engine unit numbers to visualize (if None, use all engines)
            save_dir: Directory to save visualizations (if None, display instead)
            interval: Interval step for engines to visualize (e.g., 10 for every 10th engine)
        """
        if not self.engine_clusters:
            raise ValueError("No clustering results found. Run run_clustering() first.")
            
        # Determine which engines to visualize
        if engines is None:
            engines = sorted(self.engine_clusters.keys())
        # Select every nth engine
        engines = engines[::interval]
         
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
            
        # Use the extracted metric function
        return calculate_performance_metrics(self.engine_stages, ground_truth)
    
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
        
        # Use the extracted utility function
        return evaluate_clustering_quality(self.engine_stages, rul_stages)
    
    def visualize_metrics(self, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Visualize performance and quality metrics across engines.
        """
        # Calculate metrics
        perf = self.calculate_performance_metrics()
        qual = self.evaluate_clustering_quality()
        
        # Create DataFrames for visualization
        perf_df = pd.DataFrame.from_dict(perf, orient='index')
        qual_df = pd.DataFrame.from_dict(qual, orient='index')
        
        # Use the extracted plotting function
        plot_metrics(perf_df, qual_df, self.dataset_id, save_dir)
        
        return self
    
    def compute_average_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute average performance and quality metrics across engines.
        """
        perf = self.calculate_performance_metrics()
        qual = self.evaluate_clustering_quality()
        
        # Use the extracted utility function
        return compute_average_metrics(perf, qual)

    def visualize_average_clusters(self, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Run t-SNE on combined engine data and plot aggregated cluster and RUL stage distributions.
        """
        # Combine data across engines
        X_all = np.vstack([info['data'] for info in self.engine_clusters.values()])
        labels = np.hstack([self.engine_stages[e] for e in self.engine_clusters.keys()])
        
        # Get ground truth RUL stages
        rul_all = np.hstack([self.create_rul_based_stages(self.n_clusters)[e] 
                            for e in self.engine_clusters.keys()])
        
        # Use the extracted visualization function
        plot_average_clusters(X_all, labels, rul_all, self.dataset_id, save_dir)
        
        return self
    
    def visualize_confusion_matrix_heatmap(self, engine_id: int, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Plot a heatmap of the confusion matrix between RUL-based and clustered stages for an engine.
        """
        if engine_id not in self.engine_stages:
            raise ValueError(f"No clustering results for engine {engine_id}")
            
        # Get true and predicted stages
        true_stages = self.create_rul_based_stages(self.n_clusters)[engine_id]
        predicted_stages = self.engine_stages[engine_id]
        
        # Use the extracted visualization function
        plot_confusion_matrix_heatmap(engine_id, true_stages, predicted_stages, save_dir)
        
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
        
        # Use the extracted visualization function
        plot_3d_embedding(engine_id, X, labels, save_dir)
        
        return self
    
    def visualize_average_clusters_3d(self, save_dir: Optional[str] = None) -> 'ClusteringPipeline':
        """
        Create a 3D t-SNE embedding for all engines combined, colored by cluster labels.
        """
        # Combine data across engines
        X_all = np.vstack([info['data'] for info in self.engine_clusters.values()])
        labels = np.hstack([self.engine_stages[e] for e in self.engine_clusters.keys()])
        
        # Use the extracted visualization function
        plot_average_clusters_3d(X_all, labels, self.dataset_id, save_dir)
        
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
        # Ensure base output directory exists
        os.makedirs(output_dir, exist_ok=True)
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
        cv_results = pipeline.run_cross_validation(n_splits=10)
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

def run_clustering(data_dir: str):
    """
    Convenience wrapper to run the clustering pipeline.
    """
    pipeline = ClusteringPipeline(
        data_dir=data_dir,
        n_clusters=5,
        linkage='ward',
        distance_metric='euclidean',
        normalization_method='minmax',
        scale_per_unit=False,
        verbose=True
    )
    pipeline.load_data().preprocess_data().run_clustering()
    return pipeline

if __name__ == "__main__":
    main()