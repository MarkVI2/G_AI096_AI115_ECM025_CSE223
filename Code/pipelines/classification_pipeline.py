from data.loader import CMAPSSDataLoader
from data.preprocessor import CMAPSSPreprocessor, prepare_data_for_classification, create_sequence_data
from data.splitter import create_time_series_splits


def run_classification(data_path: str, clustering_results=None):
    """
    Stub for classification pipeline.
    Args:
        data_path: Path to CMAPSS dataset
        clustering_results: Optional clustering results from phase 1
    """
    # Load data and compute RUL
    loader = CMAPSSDataLoader(data_path)
    train_df = loader.calculate_rul_for_training()
    # Preprocess training data with time-series features
    preprocessor = CMAPSSPreprocessor(normalization_method='minmax', scale_per_unit=False)
    train_proc = preprocessor.fit_transform(train_df, add_remaining_features=True, add_sensor_diff=True)
    # Discretize RUL into degradation stages
    train_labeled = prepare_data_for_classification(train_proc, n_classes=5)
    # Create sequence data across all cycles
    feature_cols = [c for c in train_labeled.columns if c not in ['unit_number','time_cycles','RUL','degradation_stage']]
    X_seq, y_seq = create_sequence_data(train_labeled, sequence_length=30, feature_columns=feature_cols, target_column='degradation_stage')
    print(f"Created sequence data: X={X_seq.shape}, y={y_seq.shape}")
    # Time-series splits per engine
    splits = create_time_series_splits(train_labeled, n_splits=5)
    for eid, engine_splits in splits.items():
        print(f"Engine {eid} has {len(engine_splits)} time-series folds")
    # TODO: train and evaluate classifier on sequence data
    return None
