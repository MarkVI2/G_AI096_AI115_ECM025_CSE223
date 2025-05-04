import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.loader import CMAPSSDataLoader
from data.preprocessor import CMAPSSPreprocessor
from regression.evaluator import RegressionEvaluator
from regression.base_models import RandomForestRegressorScratch, RidgeRegressionScratch
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class RegressionPipeline:
    def __init__(self, data_dir: str, dataset_id: str = "FD001", n_classes: int = 5, n_jobs: int = 1, classification_results=None):
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.n_classes = n_classes
        self.n_jobs = n_jobs
        # classification_results should be a DataFrame with unit_number, time_cycles, pred_stage
        self.classification_results = classification_results
        self.loader = CMAPSSDataLoader(self.data_dir)
        self.preprocessor = CMAPSSPreprocessor(normalization_method='minmax', scale_per_unit=False)
        self.evaluator = RegressionEvaluator()
        
        # Initialize regression models
        self.models = {
            'RandomForest': RandomForestRegressorScratch(
                n_estimators=50,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ),
            'Ridge': RidgeRegressionScratch(alpha=1.0),
        }

    def load_and_preprocess(self):
        # Load and preprocess train/test with RUL and features
        data = self.loader.load_dataset(self.dataset_id)
        train_df = self.loader.calculate_rul_for_training(self.dataset_id)
        test_df = self.loader.prepare_test_with_rul(self.dataset_id)
        
        # Feature engineering
        train_proc = self.preprocessor.fit_transform(train_df, add_remaining_features=True, add_sensor_diff=True)
        test_proc = self.preprocessor.transform(test_df, add_remaining_features=True, add_sensor_diff=True)
        
        # Assign degradation stage from classification if provided, else discretize RUL
        if self.classification_results is not None:
            # merge predicted stage into processed data
            preds = self.classification_results[['unit_number','time_cycles','pred_stage']]
            train_proc = train_proc.merge(preds, on=['unit_number','time_cycles'], how='left')
            test_proc = test_proc.merge(preds, on=['unit_number','time_cycles'], how='left')
            train_proc['stage'] = train_proc['pred_stage']
            test_proc['stage'] = test_proc['pred_stage']
        else:
            train_proc['stage'] = pd.cut(train_proc['RUL'], bins=self.n_classes, labels=False, include_lowest=True)
            test_proc['stage'] = pd.cut(test_proc['RUL'], bins=self.n_classes, labels=False, include_lowest=True)
        
        # Compute time to next stage
        def compute_ttns(df):
            df_sorted = df.sort_values(['unit_number','time_cycles'])
            targets = []
            for unit, grp in df_sorted.groupby('unit_number'):
                cycles = grp['time_cycles'].values
                stages = grp['stage'].values
                n = len(cycles)
                unit_t = np.full(n, np.nan)
                for i in range(n):
                    # Find next higher stage after current point
                    idx = np.where(stages > stages[i])[0]
                    idx = idx[idx > i]
                    if idx.size:
                        unit_t[i] = cycles[idx[0]] - cycles[i]
                    else:
                        # If no higher stage exists, use RUL as target
                        # This handles the last stage points
                        unit_t[i] = grp['RUL'].values[i]
                targets.extend(unit_t)
            return np.array(targets)
            
        print("Computing time-to-next-stage targets...")
        train_proc['target'] = compute_ttns(train_proc)
        test_proc['target'] = compute_ttns(test_proc)
        
        # Feature engineering: compute rolling-window stats and slopes in batch to avoid fragmentation
        sensor_cols = [c for c in train_proc.columns if c.startswith('sensor')]
        train_feats = []
        test_feats = []
        for window in [5, 10]:
            grp_train = train_proc.groupby('unit_number')
            grp_test = test_proc.groupby('unit_number')
            for col in sensor_cols:
                ma = grp_train[col].rolling(window).mean().reset_index(level=0, drop=True)
                test_ma = grp_test[col].rolling(window).mean().reset_index(level=0, drop=True)
                std = grp_train[col].rolling(window).std().reset_index(level=0, drop=True)
                test_std = grp_test[col].rolling(window).std().reset_index(level=0, drop=True)
                slope = grp_train[col].rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).reset_index(level=0, drop=True)
                test_slope = grp_test[col].rolling(window).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).reset_index(level=0, drop=True)
                train_feats.append(pd.DataFrame({f"{col}_ma{window}": ma, f"{col}_std{window}": std, f"{col}_slope{window}": slope}))
                test_feats.append(pd.DataFrame({f"{col}_ma{window}": test_ma, f"{col}_std{window}": test_std, f"{col}_slope{window}": test_slope}))
        train_proc = pd.concat([train_proc] + train_feats, axis=1)
        test_proc = pd.concat([test_proc] + test_feats, axis=1)
        
        # Drop original less useful features
        drop_cols = ['unit_number','time_cycles','RUL','stage','target']
        feature_cols = [c for c in train_proc.columns if c not in drop_cols]
        # Remove constant features
        feature_cols = [c for c in feature_cols if train_proc[c].nunique() > 1]
        # Drop rows with NaN in features or target (from rolling windows)
        train_proc = train_proc.dropna(subset=feature_cols + ['target'])
        test_proc = test_proc.dropna(subset=feature_cols + ['target'])

        print(f"Dataset prepared: {len(train_proc)} train samples, {len(test_proc)} test samples")
        print(f"Using {len(feature_cols)} features for regression")
        
        self.X_train = train_proc[feature_cols].values
        self.y_train = train_proc['target'].values
        self.X_test = test_proc[feature_cols].values
        self.y_test = test_proc['target'].values
        
        # Save metadata for visualization
        self.train_metadata = train_proc[['unit_number', 'time_cycles', 'RUL', 'stage']]
        self.test_metadata = test_proc[['unit_number', 'time_cycles', 'RUL', 'stage']]
        self.feature_names = feature_cols
        
        return self

    def train(self):
        # Train regression models, including XGBoost
        # Prepare DataFrame for XGBoost
        X_df = pd.DataFrame(self.X_train, columns=self.feature_names)
        # Configure and train XGBoost on full training data (no early stopping)
        print("Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=7,
            random_state=42,
            n_jobs=self.n_jobs,
            verbosity=0
        )
        xgb.fit(X_df, self.y_train)
        self.models['XGBoost'] = xgb
        # Train RF and Ridge on full training data
        for name in ['RandomForest', 'Ridge']:
            print(f"Training {name} on full data...")
            self.models[name].fit(self.X_train, self.y_train)
        return self

    def evaluate(self):
        # Evaluate and report each model's performance
        results = {}
        for name, model in self.models.items():
            # Predict using numpy arrays for compatibility with all models
            y_pred = model.predict(self.X_test)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)
            results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        return results

    def run(self):
        self.load_and_preprocess()
        self.train()
        return self.evaluate()


def run_regression(data_path: str, classification_results=None, n_jobs=1):
    """
    Runs the regression pipeline for Phase 3 (Time-to-Next-Failure Prediction)
    
    Args:
        data_path: Path to CMAPSS dataset
        classification_results: Optional classification results from phase 2
        n_jobs: Number of parallel jobs to use
    """
    pipeline = RegressionPipeline(data_dir=data_path, n_jobs=n_jobs, classification_results=classification_results)
    metrics = pipeline.run()
    
    # Print metrics
    print("\n=== Regression Model Results (Time-to-Next-Stage Prediction) ===")
    for model, scores in metrics.items():
        print(f"\n{model} Performance:")
        print(f"  MAE: {scores['MAE']:.2f} cycles")
        print(f"  RMSE: {scores['RMSE']:.2f} cycles")
        print(f"  R²: {scores['R2']:.4f}")
    
    # Generate and save regression plots
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'regression')
    os.makedirs(results_dir, exist_ok=True)
    # Scatter plots: actual vs predicted for each model
    for name, model in pipeline.models.items():
        # Predict with numpy array input
        y_pred = model.predict(pipeline.X_test)
        plt.figure(figsize=(8,6))
        plt.scatter(pipeline.y_test, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
        maxval = max(pipeline.y_test.max(), y_pred.max())
        plt.plot([0, maxval], [0, maxval], 'r--', linewidth=1)
        plt.xlabel('Actual Time to Next Stage')
        plt.ylabel('Predicted Time to Next Stage')
        plt.title(f'{name}: Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'{name}_actual_vs_predicted.png'))
        plt.close()
    # Bar chart of metrics
    # Prepare DataFrame
    import pandas as _pd
    df = _pd.DataFrame(metrics).T
    plt.figure(figsize=(8,6))
    df[['R2', 'RMSE']].plot(kind='bar', ax=plt.gca())
    plt.title('Regression Performance (R2 and RMSE)')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'performance_summary.png'))
    plt.close()
    print(f"\nRegression visualizations saved to {results_dir}")
    # Save detailed predictions for all models
    preds_df = pipeline.test_metadata.copy()
    preds_df['true_time_to_next_stage'] = pipeline.y_test
    for name, model in pipeline.models.items():
        preds_df[f'pred_{name}'] = model.predict(pipeline.X_test)
    preds_df.to_csv(os.path.join(results_dir, 'time_to_next_stage_predictions.csv'), index=False)
    return metrics


def run_regression_fast(data_path: str, classification_results=None, n_jobs=1, sample_size=5000):
    """
    Fast version of regression pipeline that:
    1. Uses a random sample of training data (default 5000 samples)
    2. Reduces SVR training iterations
    3. Skips weighted ensemble which requires retraining
    4. Samples test data for faster evaluation
    
    Args:
        data_path: Path to CMAPSS dataset
        classification_results: Optional classification results from phase 2
        n_jobs: Number of parallel jobs to use
        sample_size: Number of samples to use for training (smaller = faster)
    """
    print(f"Running fast regression pipeline (using {sample_size} samples)...")
    
    # Import necessary modules
    import os, sys
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from data.loader import CMAPSSDataLoader
    from data.preprocessor import CMAPSSPreprocessor
    from regression.evaluator import RegressionEvaluator
    from regression.base_models import RandomForestRegressorScratch, RidgeRegressionScratch
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import time
    
    # Track execution time
    start_time = time.time()
    
    # Set up components
    loader = CMAPSSDataLoader(data_path)
    preprocessor = CMAPSSPreprocessor(normalization_method='minmax', scale_per_unit=False)
    evaluator = RegressionEvaluator()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    loader.load_dataset("FD001")  # Use dataset FD001 for speed
    train_df = loader.calculate_rul_for_training("FD001")
    test_df = loader.prepare_test_with_rul("FD001")
    
    # Feature engineering (same as original pipeline)
    train_proc = preprocessor.fit_transform(train_df, add_remaining_features=True, add_sensor_diff=True)
    test_proc = preprocessor.transform(test_df, add_remaining_features=True, add_sensor_diff=True)
    
    # Discretize stages for target computation
    n_classes = 5  # 5 degradation stages
    train_proc['stage'] = pd.cut(train_proc['RUL'], bins=n_classes, labels=False, include_lowest=True)
    test_proc['stage'] = pd.cut(test_proc['RUL'], bins=n_classes, labels=False, include_lowest=True)
    
    # Compute time to next stage
    print("Computing time-to-next-stage targets...")
    def compute_ttns(df):
        df_sorted = df.sort_values(['unit_number','time_cycles'])
        targets = []
        for unit, grp in df_sorted.groupby('unit_number'):
            cycles = grp['time_cycles'].values
            stages = grp['stage'].values
            n = len(cycles)
            unit_t = np.full(n, np.nan)
            for i in range(n):
                # Find next higher stage after current point
                idx = np.where(stages > stages[i])[0]
                idx = idx[idx > i]
                if idx.size:
                    unit_t[i] = cycles[idx[0]] - cycles[i]
                else:
                    # If no higher stage exists, use RUL as target
                    unit_t[i] = grp['RUL'].values[i]
            targets.extend(unit_t)
        return np.array(targets)
        
    train_proc['target'] = compute_ttns(train_proc)
    test_proc['target'] = compute_ttns(test_proc)
    
    # Select features
    drop_cols = ['unit_number','time_cycles','RUL','stage','target']
    feature_cols = [c for c in train_proc.columns if c not in drop_cols]
    
    # Prepare training and test datasets
    X_train_full = train_proc[feature_cols].values
    y_train_full = train_proc['target'].values
    X_test_full = test_proc[feature_cols].values
    y_test_full = test_proc['target'].values
    
    # Sample training data for faster processing
    if len(X_train_full) > sample_size:
        print(f"Sampling {sample_size} points from {len(X_train_full)} training samples")
        indices = np.random.choice(len(X_train_full), sample_size, replace=False)
        X_train = X_train_full[indices]
        y_train = y_train_full[indices]
    else:
        X_train = X_train_full
        y_train = y_train_full
    
    # Sample test data for faster evaluation
    test_sample_size = min(2000, len(X_test_full))
    test_indices = np.random.choice(len(X_test_full), test_sample_size, replace=False)
    X_test = X_test_full[test_indices]
    y_test = y_test_full[test_indices]
    
    print(f"Training with {len(X_train)} samples, evaluating on {len(X_test)} samples")
    print(f"Using {len(feature_cols)} features")
    
    # Initialize models with faster configurations
    models = {
        'RandomForest': RandomForestRegressorScratch(
            n_estimators=20,  # Reduced from 50
            max_depth=5,
            subsample=0.8,
            random_state=42
        ),
        'Ridge': RidgeRegressionScratch(alpha=1.0),
        'GBR': HistGradientBoostingRegressor(max_iter=200, random_state=42)  # Reduced iterations
    }
    
    # Train individual models
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
    
    # Evaluate on sampled test set
    print("Evaluating models...")
    results = {}
    
    # Evaluate individual models
    for name, model in models.items():
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    # Generate a simple scatter plot of predictions vs actual
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
    
    # Add diagonal reference line
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.xlabel('Actual Time to Next Stage')
    plt.ylabel('Predicted Time to Next Stage')
    plt.title('Actual vs Predicted Time-to-Next-Stage')
    plt.grid(True, alpha=0.3)
    
    # Save figure
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'regression')
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'actual_vs_predicted_fast.png'))
    plt.close()
    
    # Print metrics
    print("\n=== Regression Model Results (Fast Version) ===")
    for model, scores in results.items():
        print(f"\n{model} Performance:")
        print(f"  MAE: {scores['MAE']:.2f} cycles")
        print(f"  RMSE: {scores['RMSE']:.2f} cycles")
        print(f"  R²: {scores['R2']:.4f}")
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    
    return results