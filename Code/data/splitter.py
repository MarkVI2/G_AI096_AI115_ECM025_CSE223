import numpy as np
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit

def create_validation_split(train_data, val_size=0.2, random_state=42):
    """
    Create a validation set from training data using time-based split per engine unit.
    Splits the last val_size fraction of cycles for each engine as validation.
    """
    train_indices = []
    val_indices = []
    # Split each engine's cycles: last fraction as validation
    for unit in train_data['unit_number'].unique():
        unit_df = train_data[train_data['unit_number'] == unit].sort_values('time_cycles')
        n = len(unit_df)
        split_at = int(n * (1 - val_size))
        indices = unit_df.index.tolist()
        train_indices.extend(indices[:split_at])
        val_indices.extend(indices[split_at:])
    train_subset = train_data.loc[train_indices]
    validation_set = train_data.loc[val_indices]
     
    return train_subset, validation_set

def create_kfold_splits(train_data, n_splits=5, random_state=42):
    """
    Creates k-fold cross-validation splits respecting engine boundaries
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data
    n_splits : int
        Number of folds
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    list : List of (train_idx, val_idx) tuples for each fold
    """
    # Get unique engine IDs
    engine_ids = train_data['unit_number'].unique()
    
    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    engine_splits = []
    for train_engine_idx, val_engine_idx in kf.split(engine_ids):
        train_engines = engine_ids[train_engine_idx]
        val_engines = engine_ids[val_engine_idx]
        
        train_idx = train_data[train_data['unit_number'].isin(train_engines)].index
        val_idx = train_data[train_data['unit_number'].isin(val_engines)].index
        
        engine_splits.append((train_idx, val_idx))
    
    return engine_splits

def create_time_series_splits(train_data, n_splits=5):
    """
    Creates time-based splits for time series cross-validation
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data
    n_splits : int
        Number of splits
        
    Returns:
    --------
    list : List of (train_idx, val_idx) tuples for each split
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    splits_by_engine = {}
    
    for engine_id in train_data['unit_number'].unique():
        engine_data = train_data[train_data['unit_number'] == engine_id]

        engine_splits = []
        for train_idx, test_idx in tscv.split(engine_data):
            engine_train = engine_data.iloc[train_idx].index
            engine_test = engine_data.iloc[test_idx].index
            engine_splits.append((engine_train, engine_test))
            
        splits_by_engine[engine_id] = engine_splits
    
    return splits_by_engine