import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def select_k_best_features(X: np.ndarray,
                           y: np.ndarray,
                           feature_names: list,
                           k: int = 10,
                           score_func: str = 'f_classif'
                          ) -> list:
    """
    Return the indices of the top-k features by ANOVA F-test or mutual info.
    score_func: 'f_classif' or 'mutual_info'
    """
    if score_func == 'f_classif':
        func = f_classif
    elif score_func in ('mutual_info', 'mi'):
        func = mutual_info_classif
    else:
        raise ValueError(f"Unknown score_func={score_func}")
    
    selector = SelectKBest(func, k=k)
    selector.fit(X, y)
    return selector.get_support(indices=True).tolist()