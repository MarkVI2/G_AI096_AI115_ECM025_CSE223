import numpy as np

def smote(X: np.ndarray, y: np.ndarray, k_neighbors: int = 5, random_state: int = 42):
    """
    Synthetic Minority Over-sampling Technique (SMOTE) for multi-class datasets.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Label array, shape (n_samples,)
        k_neighbors: Number of nearest neighbors to use for synthetic generation
        random_state: Seed for reproducibility

    Returns:
        X_res: Resampled feature matrix
        y_res: Resampled label array
    """
    np.random.seed(random_state)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_res = [X]
    y_res = [y]
    # For each class that is under-represented, generate synthetic samples
    for cls, count in zip(classes, counts):
        if count < max_count:
            # Samples of this class
            X_c = X[y == cls]
            n_c, n_features = X_c.shape
            # Compute pairwise distances among minority samples
            diff = X_c[:, np.newaxis, :] - X_c[np.newaxis, :, :]
            dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
            # Exclude self by setting high distance
            np.fill_diagonal(dist_matrix, np.inf)
            # Indices of k nearest neighbors for each sample
            neighbors = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]
            # Number of synthetic samples to create
            n_to_create = max_count - n_c
            synthetic = np.zeros((n_to_create, n_features))
            for i in range(n_to_create):
                # Randomly pick a base sample index
                idx = np.random.randint(0, n_c)
                # Randomly pick one neighbor
                nei = np.random.choice(neighbors[idx])
                # Generate synthetic sample with interpolation
                gap = np.random.rand()
                synthetic[i] = X_c[idx] + gap * (X_c[nei] - X_c[idx])
            X_res.append(synthetic)
            y_res.append(np.full(n_to_create, cls))
    # Concatenate original and synthetic samples
    X_out = np.vstack(X_res)
    y_out = np.hstack(y_res)
    return X_out, y_out
