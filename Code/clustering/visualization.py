import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def compute_tsne(X: np.ndarray, n_components: int = 2, perplexity: float = 30.0, random_state: int = 0) -> np.ndarray:
    """
    Compute a t-SNE embedding of the data.
    """
    return TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state).fit_transform(X)


def visualize_clusters_tsne(X: np.ndarray, labels: np.ndarray, title: str = None, save_path: str = None):
    """
    Plot a 2D t-SNE scatter of the data colored by cluster labels.
    """
    X_tsne = compute_tsne(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=5)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def visualize_degradation_stages(X: np.ndarray, stages: np.ndarray, time_cycles: np.ndarray, title: str = None, save_path: str = None):
    """
    Plot t-SNE colored by degradation stage along time.
    """
    X_tsne = compute_tsne(X)
    plt.figure(figsize=(8, 6))
    for t, s in zip(X_tsne, stages):
        plt.scatter(t[0], t[1], color=plt.cm.viridis(s / np.max(stages)), s=5)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()