import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5, random_state: int = 42):
    """
    Optimized SMOTE implementation with better memory management and vectorization.
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    # Pre-allocate arrays for better memory efficiency
    total_new_samples = sum(max(0, max_count - count) for count in counts)
    X_synthetic = np.zeros((total_new_samples, X.shape[1]))
    y_synthetic = np.zeros(total_new_samples, dtype=y.dtype)
    
    current_idx = 0

    for cls, count in zip(classes, counts):
        if count < max_count:
            X_c = X[y == cls]
            n_c, n_feat = X_c.shape
            n_to_create = max_count - n_c

            # Build a nearest neighbors model only if there are enough samples
            if n_c > 1:
                if n_c > k_neighbors:
                    # Use KNN for more intelligent sampling
                    nn = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X_c)
                    distances, indices = nn.kneighbors(X_c)
                    
                    # For each sample to create:
                    for i in range(n_to_create):
                        # Pick a random minority class sample
                        reference_idx = np.random.randint(0, n_c)
                        # Pick one of its k neighbors (skip the first as it's the same point)
                        neighbor_idx = indices[reference_idx, 1 + np.random.randint(0, k_neighbors)]
                        # Random interpolation factor
                        gap = np.random.rand()
                        # Create synthetic sample
                        X_synthetic[current_idx] = X_c[reference_idx] + gap * (X_c[neighbor_idx] - X_c[reference_idx])
                        y_synthetic[current_idx] = cls
                        current_idx += 1
                else:
                    # If fewer samples than k_neighbors, just do random pairs (as before)
                    for i in range(n_to_create):
                        a, b = np.random.choice(n_c, 2, replace=False)
                        gap = np.random.rand()
                        X_synthetic[current_idx] = X_c[a] + gap * (X_c[b] - X_c[a])
                        y_synthetic[current_idx] = cls
                        current_idx += 1
            else:
                # If only one sample, duplicate it with small random noise
                for i in range(n_to_create):
                    noise = 0.05 * np.random.randn(n_feat)  # 5% noise
                    X_synthetic[current_idx] = X_c[0] + noise
                    y_synthetic[current_idx] = cls
                    current_idx += 1

    # Combine original and synthetic data
    X_out = np.vstack([X, X_synthetic[:current_idx]])
    y_out = np.hstack([y, y_synthetic[:current_idx]])
    
    return X_out, y_out
