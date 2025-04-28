import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calculate_performance_metrics(predicted_stages: Dict[int, np.ndarray], 
                                ground_truth: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
    """
    Calculate accuracy, precision, recall, and F1 scores for clustering results.
    
    Args:
        predicted_stages: Dictionary mapping engine IDs to arrays of predicted stages
        ground_truth: Dictionary mapping engine IDs to arrays of true stages
                         
    Returns:
        Dictionary mapping engine IDs to dictionaries of performance metrics
    """
    metrics = {}
    
    for engine_id, pred_stages in predicted_stages.items():
        if engine_id not in ground_truth:
            continue
            
        true_stages = ground_truth[engine_id]
        
        # Ensure same length
        min_len = min(len(true_stages), len(pred_stages))
        true_stages = true_stages[:min_len]
        pred_stages = pred_stages[:min_len]
        
        # Calculate metrics
        accuracy = accuracy_score(true_stages, pred_stages)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_stages, pred_stages, average='weighted'
        )
        
        metrics[engine_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
    return metrics


def calculate_gini_index(clusters: np.ndarray, rul_ranges: np.ndarray) -> float:
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


def calculate_entropy(clusters: np.ndarray, rul_ranges: np.ndarray) -> float:
    """
    Calculate Shannon entropy for clustering compared to RUL ranges.
    Lower entropy indicates better clustering alignment with RUL ranges.
    
    Args:
        clusters: Cluster assignments
        rul_ranges: RUL-based stage assignments
        
    Returns:
        Shannon entropy
    """
    import math
    
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


def evaluate_clustering_quality(predicted_stages: Dict[int, np.ndarray], 
                              rul_stages: Dict[int, np.ndarray]) -> Dict[int, Dict[str, float]]:
    """
    Evaluate clustering quality by comparing with RUL-based ranges.
    
    Args:
        predicted_stages: Dictionary mapping engine IDs to arrays of predicted stages
        rul_stages: Dictionary mapping engine IDs to arrays of RUL-based stages
        
    Returns:
        Dictionary mapping engine IDs to dictionaries of quality metrics
    """
    quality_metrics = {}
    
    for engine_id, pred_stages in predicted_stages.items():
        if engine_id not in rul_stages:
            continue
            
        true_stages = rul_stages[engine_id]
        
        # Ensure same length
        min_len = min(len(true_stages), len(pred_stages))
        true_stages = true_stages[:min_len]
        pred_stages = pred_stages[:min_len]
        
        # Calculate metrics
        gini = calculate_gini_index(pred_stages, true_stages)
        entropy = calculate_entropy(pred_stages, true_stages)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_stages, pred_stages)
        
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


def compute_average_metrics(performance_metrics: Dict[int, Dict[str, float]],
                          quality_metrics: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute average performance and quality metrics across engines.
    
    Args:
        performance_metrics: Dictionary of performance metrics per engine
        quality_metrics: Dictionary of quality metrics per engine
        
    Returns:
        Dictionary of average metrics
    """
    # Average performance
    avg_perf = {}
    if performance_metrics:
        keys = next(iter(performance_metrics.values())).keys()
        for k in keys:
            avg_perf[k] = np.mean([v[k] for v in performance_metrics.values()])
    
    # Average quality
    avg_qual = {}
    if quality_metrics:
        keys = next(iter(quality_metrics.values())).keys()
        for k in keys:
            avg_qual[k] = np.mean([v[k] for v in quality_metrics.values()])
    
    return {'average_performance': avg_perf, 'average_quality': avg_qual}