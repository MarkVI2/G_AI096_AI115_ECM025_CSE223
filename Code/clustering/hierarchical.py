import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
import time
import sys
import random
from collections import defaultdict
from .visualization import compute_tsne, visualize_clusters_tsne, visualize_degradation_stages

class HierarchicalAgglomerativeClustering:
    """
    A from-scratch implementation of Hierarchical Agglomerative Clustering (HAC)
    for the CMAPSS dataset to identify 5 distinct degradation stages.
    
    This implementation supports various linkage methods:
    - 'single': Minimum distance between points in clusters
    - 'complete': Maximum distance between points in clusters
    - 'average': Average distance between all pairs of points in clusters
    - 'ward': Minimizes variance within clusters (similar to Ward's method)
    """
    
    def __init__(self, n_clusters: int = 5, linkage: str = 'ward', 
                 distance_metric: str = 'euclidean', verbose: bool = False):
        """
        Initialize the HAC clustering algorithm.
        
        Args:
            n_clusters: Number of clusters to identify (default: 5 for degradation stages)
            linkage: Linkage criterion to use ('single', 'complete', 'average', 'ward')
            distance_metric: Distance metric to use ('euclidean', 'manhattan', 'cosine')
            verbose: Whether to print progress information
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.verbose = verbose
        self.clusters = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.dendrogram_ = []
        self.distances_ = None
        
        # Validate parameters
        valid_linkages = ['single', 'complete', 'average', 'ward']
        if linkage not in valid_linkages:
            raise ValueError(f"Invalid linkage: {linkage}. Must be one of {valid_linkages}")
        
        valid_metrics = ['euclidean', 'manhattan', 'cosine']
        if distance_metric not in valid_metrics:
            raise ValueError(f"Invalid distance metric: {distance_metric}. Must be one of {valid_metrics}")
    
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the distance matrix for all data points.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        
        if self.verbose:
            print(f"Computing distance matrix for {n_samples} samples...")
            start_time = time.time()
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                dist = self._calculate_distance(X[i], X[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        if self.verbose:
            print(f"Distance matrix computed in {time.time() - start_time:.2f} seconds")
        
        return dist_matrix
    
    def _calculate_distance(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        Calculate distance between two points using the selected distance metric.
        
        Args:
            point_a: First data point
            point_b: Second data point
            
        Returns:
            Distance between the points
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((point_a - point_b) ** 2))
        
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(point_a - point_b))
        
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(point_a, point_b)
            norm_a = np.linalg.norm(point_a)
            norm_b = np.linalg.norm(point_b)
            
            # Avoid division by zero
            if norm_a == 0 or norm_b == 0:
                return 1.0  # Maximum distance
                
            similarity = dot_product / (norm_a * norm_b)
            # Convert similarity to distance (1 - similarity)
            return 1.0 - similarity
    
    def _calculate_cluster_distance(self, cluster_a: List[int], cluster_b: List[int], 
                                   dist_matrix: np.ndarray) -> float:
        """
        Calculate distance between two clusters using the selected linkage criterion.
        
        Args:
            cluster_a: List of indices for points in first cluster
            cluster_b: List of indices for points in second cluster
            dist_matrix: Distance matrix for all data points
            
        Returns:
            Distance between the clusters
        """
        if self.linkage == 'single':
            # Single linkage: minimum distance between any points
            return min(dist_matrix[i, j] for i in cluster_a for j in cluster_b)
        
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance between any points
            return max(dist_matrix[i, j] for i in cluster_a for j in cluster_b)
        
        elif self.linkage == 'average':
            # Average linkage: average distance between all pairs of points
            total_dist = sum(dist_matrix[i, j] for i in cluster_a for j in cluster_b)
            return total_dist / (len(cluster_a) * len(cluster_b))
        
        elif self.linkage == 'ward':
            # Simplified Ward's method: increase in within-cluster variance
            # We approximate this by using all pairwise distances
            merged = cluster_a + cluster_b
            n_a, n_b, n_merged = len(cluster_a), len(cluster_b), len(merged)
            
            # Calculate centroid-to-centroid distance (squares)
            centroid_dist_squared = 0
            for i in cluster_a:
                for j in cluster_b:
                    centroid_dist_squared += dist_matrix[i, j] ** 2
            
            centroid_dist_squared = centroid_dist_squared / (n_a * n_b)
            
            # Return the increase in sum of squares scaled by cluster sizes
            return centroid_dist_squared * (n_a * n_b) / (n_a + n_b)
    
    def fit(self, X: np.ndarray) -> 'HierarchicalAgglomerativeClustering':
        """
        Fit the HAC model to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        n_samples = X.shape[0]
        
        if self.verbose:
            print(f"Starting HAC clustering for {n_samples} samples with {self.n_clusters} clusters...")
            sys.stdout.flush()
            fit_start_time = time.time()
        
        # Compute distance matrix
        dist_matrix = self._compute_distance_matrix(X)
        self.distances_ = dist_matrix
        
        # Initialize each data point as its own cluster
        clusters = {i: [i] for i in range(n_samples)}
        
        # For storing dendrogram information
        merge_history = []
        heights = []
        
        # Perform agglomerative clustering
        while len(clusters) > self.n_clusters:
            if self.verbose and len(clusters) % 100 == 0:
                print(f"  Current number of clusters: {len(clusters)}")
                sys.stdout.flush()
            
            # Find the two closest clusters
            min_dist = float('inf')
            merge_candidates = (None, None)
            
            cluster_keys = list(clusters.keys())
            for i in range(len(cluster_keys)):
                for j in range(i+1, len(cluster_keys)):
                    cluster_a = clusters[cluster_keys[i]]
                    cluster_b = clusters[cluster_keys[j]]
                    
                    dist = self._calculate_cluster_distance(cluster_a, cluster_b, dist_matrix)
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_candidates = (cluster_keys[i], cluster_keys[j])
            
            # Merge the two closest clusters
            a_idx, b_idx = merge_candidates
            clusters[a_idx].extend(clusters[b_idx])
            
            # Record the merge for dendrogram
            merge_history.append((a_idx, b_idx))
            heights.append(min_dist)
            
            # Remove the merged cluster
            del clusters[b_idx]
            
        # Assign cluster labels to each sample
        labels = np.full(n_samples, -1, dtype=int)
        for cluster_idx, sample_indices in enumerate(clusters.values()):
            for sample_idx in sample_indices:
                labels[sample_idx] = cluster_idx
        
        self.clusters = clusters
        self.labels_ = labels
        self.dendrogram_ = {'merges': merge_history, 'heights': heights}
        
        # Calculate cluster centers
        self.cluster_centers_ = self._calculate_cluster_centers(X)
        
        if self.verbose:
            print(f"HAC clustering completed in {time.time() - fit_start_time:.2f} seconds")
            print(f"Final number of clusters: {len(clusters)}")
        
        return self
    
    def _calculate_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the centroid of each cluster.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Array of cluster centroids
        """
        centers = []
        for cluster_idx in range(self.n_clusters):
            mask = self.labels_ == cluster_idx
            if np.any(mask):
                center = X[mask].mean(axis=0)
                centers.append(center)
            else:
                # Empty cluster, use random point
                centers.append(X[random.randint(0, X.shape[0]-1)])
        
        return np.array(centers)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster for each sample in X.
        
        Args:
            X: New data points of shape (n_samples, n_features)
            
        Returns:
            Cluster labels for each point
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet.")
            
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Find closest cluster center
            min_dist = float('inf')
            min_cluster = -1
            
            for cluster_idx, center in enumerate(self.cluster_centers_):
                dist = self._calculate_distance(X[i], center)
                if dist < min_dist:
                    min_dist = dist
                    min_cluster = cluster_idx
            
            labels[i] = min_cluster
        
        return labels
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster labels in one step.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Cluster labels for each point
        """
        self.fit(X)
        return self.labels_


def map_clusters_to_degradation_stages(clusters: np.ndarray, time_cycles: np.ndarray) -> np.ndarray:
    """
    Map arbitrary cluster labels to degradation stages (0-4) based on average time cycles.
    Stage 0 is normal operation (highest avg cycle), Stage 4 is near failure (lowest avg cycle).
    
    Args:
        clusters: Cluster assignments for each data point
        time_cycles: Time cycles for each data point
        
    Returns:
        Degradation stages (0-4) for each data point
    """
    # Calculate average cycle for each cluster
    unique_clusters = np.unique(clusters)
    cluster_avg_cycles = {}
    
    for cluster in unique_clusters:
        mask = clusters == cluster
        avg_cycle = np.mean(time_cycles[mask])
        cluster_avg_cycles[cluster] = avg_cycle
    
    # Sort clusters by average cycle (ascending) so stage 0=normal (lowest cycles), stage 4=failure (highest cycles)
    sorted_clusters = sorted(cluster_avg_cycles.keys(), 
                           key=lambda x: cluster_avg_cycles[x])
    
    # Create mapping from cluster to stage
    cluster_to_stage = {}
    for stage, cluster in enumerate(sorted_clusters):
        cluster_to_stage[cluster] = stage
    
    # Map each point's cluster to its stage
    stages = np.zeros_like(clusters)
    for i, cluster in enumerate(clusters):
        stages[i] = cluster_to_stage[cluster]
    
    return stages


def analyze_stage_characteristics(X: np.ndarray, stages: np.ndarray, 
                                feature_names: List[str]) -> pd.DataFrame:
    """
    Analyze the characteristics of each degradation stage by calculating
    average sensor values and other statistics.
    
    Args:
        X: Input data of shape (n_samples, n_features)
        stages: Degradation stage for each data point (0-4)
        feature_names: Names of the features
        
    Returns:
        DataFrame with stage characteristics
    """
    stage_stats = []
    
    for stage in range(5):
        mask = stages == stage
        stage_data = X[mask]
        
        # Skip if no data for this stage
        if len(stage_data) == 0:
            continue
            
        # Calculate mean and std for each feature
        means = np.mean(stage_data, axis=0)
        stds = np.std(stage_data, axis=0)
        
        # Compile statistics
        stats = {'Stage': stage}
        stats['Count'] = len(stage_data)
        
        for i, feature in enumerate(feature_names):
            stats[f'{feature}_mean'] = means[i]
            stats[f'{feature}_std'] = stds[i]
        
        stage_stats.append(stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stage_stats)
    
    return stats_df


if __name__ == "__main__":
    from ..data.loader import CMAPSSDataLoader
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Load CMAPSS data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(data_dir), 'data')
    
    loader = CMAPSSDataLoader(data_dir)
    
    # Load FD001 dataset
    print("Loading dataset FD001...")
    data = loader.load_dataset("FD001")
    train_df = data['train']
    
    # Get sensor columns
    sensor_cols = [col for col in train_df.columns if col.startswith('sensor_')]
    
    # Prepare data for one engine unit as example
    engine_id = 1
    engine_data = train_df[train_df['unit_number'] == engine_id]
    X = engine_data[sensor_cols].values
    time_cycles = engine_data['time_cycles'].values
    
    print(f"Clustering data for engine {engine_id} with shape {X.shape}...")
    
    # Create the HAC model
    hac = HierarchicalAgglomerativeClustering(
        n_clusters=5,
        linkage='ward',
        distance_metric='euclidean',
        verbose=True
    )
    
    # Fit the model
    cluster_labels = hac.fit_predict(X)
    
    # Map clusters to degradation stages
    degradation_stages = map_clusters_to_degradation_stages(cluster_labels, time_cycles)
    
    # Visualize results
    visualize_clusters_tsne(X, cluster_labels, 
                          title=f'HAC Clustering Results for Engine {engine_id}')
    
    visualize_degradation_stages(X, degradation_stages, time_cycles,
                               title=f'Degradation Stages for Engine {engine_id}')
    
    # Analyze stage characteristics
    stage_stats = analyze_stage_characteristics(X, degradation_stages, sensor_cols)
    print("\nStage Characteristics:")
    print(stage_stats[['Stage', 'Count']])