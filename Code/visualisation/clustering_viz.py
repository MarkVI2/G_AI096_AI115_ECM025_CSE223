import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import os


def plot_engine_clusters(engine_id: int, results: Dict, stages: np.ndarray, save_dir: Optional[str] = None):
    """
    Visualize clustering results for a specific engine.
    
    Args:
        engine_id: Engine unit number to visualize
        results: Dictionary with clustering results (data, labels, time_cycles)
        stages: Degradation stages array
        save_dir: Directory to save visualizations (if None, display instead)
    """
    from clustering.visualization import visualize_clusters_tsne, visualize_degradation_stages

    X = results['data']
    labels = results['labels']
    time_cycles = results['time_cycles']
    
    # Determine save path for clustering visualization
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


def plot_confusion_matrix_heatmap(engine_id: int, true_stages: np.ndarray, 
                                predicted_stages: np.ndarray, save_dir: Optional[str] = None):
    """
    Plot a heatmap of the confusion matrix between RUL-based and clustered stages for an engine.
    
    Args:
        engine_id: Engine ID number
        true_stages: Array of true RUL-based stages
        predicted_stages: Array of predicted degradation stages
        save_dir: Directory to save the plot (if None, display instead)
    """
    from sklearn.metrics import confusion_matrix
    
    # Align lengths
    n = min(len(true_stages), len(predicted_stages))
    cm = confusion_matrix(true_stages[:n], predicted_stages[:n])
    
    # Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Stage')
    plt.ylabel('True RUL Stage')
    plt.title(f'Engine {engine_id} Confusion Matrix Heatmap')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'engine_{engine_id}_cm_heatmap.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix heatmap to {path}")
    else:
        plt.show()
    plt.close()


def plot_3d_embedding(engine_id: int, data: np.ndarray, labels: np.ndarray, 
                    save_dir: Optional[str] = None, title: Optional[str] = None):
    """
    Generate a 3D t-SNE embedding of sensor features, colored by cluster label.
    
    Args:
        engine_id: Engine ID number
        data: Feature data array
        labels: Cluster labels array
        save_dir: Directory to save the plot (if None, display instead)
        title: Optional custom title for the plot
    """
    # 3D t-SNE
    embed3d = TSNE(n_components=3, random_state=0).fit_transform(data)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embed3d[:,0], embed3d[:,1], embed3d[:,2], 
                        c=labels, cmap='viridis', s=5)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Engine {engine_id} 3D t-SNE Clusters')
    
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'engine_{engine_id}_3d_tsne.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved 3D t-SNE embedding to {path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_average_clusters(X_all: np.ndarray, labels: np.ndarray, rul_stages: np.ndarray, 
                         dataset_id: str, save_dir: Optional[str] = None):
    """
    Plot aggregated cluster and RUL stage distributions using t-SNE.
    
    Args:
        X_all: Combined feature data across engines
        labels: Combined cluster labels across engines
        rul_stages: Combined RUL-based stages across engines
        dataset_id: Dataset identifier
        save_dir: Directory to save the plot (if None, display instead)
    """
    # Compute t-SNE embedding
    tsne_embed = TSNE(n_components=2, random_state=0).fit_transform(X_all)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Clusters
    scatter1 = ax1.scatter(tsne_embed[:,0], tsne_embed[:,1], c=labels, cmap='viridis', s=5)
    ax1.set_title('Aggregated t-SNE: Cluster Labels')
    
    # RUL stages
    scatter2 = ax2.scatter(tsne_embed[:,0], tsne_embed[:,1], c=rul_stages, cmap='plasma', s=5)
    ax2.set_title('Aggregated t-SNE: RUL-based Stages')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'average_clusters_{dataset_id}.png')
        fig.savefig(path, dpi=300)
        print(f"Saved aggregated cluster plot to {path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_average_clusters_3d(X_all: np.ndarray, labels: np.ndarray, 
                           dataset_id: str, save_dir: Optional[str] = None):
    """
    Create a 3D t-SNE embedding for all engines combined, colored by cluster labels.
    
    Args:
        X_all: Combined feature data across engines
        labels: Combined cluster labels across engines
        dataset_id: Dataset identifier
        save_dir: Directory to save the plot (if None, display instead)
    """
    # Compute 3D t-SNE embedding
    embed3d = TSNE(n_components=3, random_state=0).fit_transform(X_all)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(embed3d[:,0], embed3d[:,1], embed3d[:,2], 
                        c=labels, cmap='viridis', s=5)
    
    ax.set_title(f'Aggregated 3D t-SNE Clusters ({dataset_id})')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'average_clusters_3d_{dataset_id}.png')
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved averaged 3D cluster plot to {path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_metrics(perf_df: pd.DataFrame, qual_df: pd.DataFrame, 
                dataset_id: str, save_dir: Optional[str] = None):
    """
    Visualize performance and quality metrics across engines.
    
    Args:
        perf_df: DataFrame of performance metrics
        qual_df: DataFrame of quality metrics
        dataset_id: Dataset identifier
        save_dir: Directory to save the plots (if None, display instead)
    """
    # Plot performance metrics
    plt.figure(figsize=(10, 5))
    perf_df.plot(kind='bar')
    plt.title('Performance Metrics by Engine')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        perf_path = os.path.join(save_dir, f'performance_metrics_plot_{dataset_id}.png')
        plt.savefig(perf_path, dpi=300)
        print(f"Saved performance metrics plot to {perf_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Plot quality metrics
    plt.figure(figsize=(10, 5))
    qual_df.plot(kind='bar')
    plt.title('Quality Metrics by Engine')
    plt.tight_layout()
    
    if save_dir:
        qual_path = os.path.join(save_dir, f'quality_metrics_plot_{dataset_id}.png')
        plt.savefig(qual_path, dpi=300)
        print(f"Saved quality metrics plot to {qual_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sensor_trends_by_stage(
    engine_id: int,
    engine_data: pd.DataFrame,
    stages: np.ndarray,
    sensor_col: str,
    save_dir: Optional[str] = None
):
    """
    Plot time_cycles vs. a single sensor reading, colored by degradation stage.
    """
    df = engine_data.copy()
    df['stage'] = stages
    plt.figure(figsize=(8, 4))
    for stg in sorted(df['stage'].unique()):
        grp = df[df['stage'] == stg]
        avg = grp.groupby('time_cycles')[sensor_col].mean()
        plt.plot(avg.index, avg.values, label=f"Stage {stg}")
    plt.xlabel('Time Cycle')
    plt.ylabel(sensor_col)
    plt.title(f'Engine {engine_id}: {sensor_col} trend by Stage')
    plt.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"engine_{engine_id}_{sensor_col}_by_stage.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved sensor trend plot to {path}")
    else:
        plt.show()
    plt.close()