from joblib import Parallel, delayed, Memory
import joblib
import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split, ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import argparse

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.hyperparameters import BEST_HYPERPARAMS

# Import custom modules
from classification.base_model import CustomLogisticRegression, CustomRandomForest
from classification.evaluator import evaluate_model, analyze_feature_importance
from classification.smote import handle_imbalance
from features.statistical import add_statistical_features
from features.time_series import add_time_series_features
from features.selector import select_k_best_features

# Create a memory cache for expensive operations
cache_dir = '/tmp/joblib_cache'
memory = Memory(cache_dir, verbose=0)

def load_data(dataset_num):
    """Load data from specified dataset number (1-4)"""
    file_path = f"/Users/garga4/Desktop/MUGithub/Year2/2-2/CS3102/p3-project/G_AI096_AI115_ECM025_CSE223/Code/results/clustering/FD00{dataset_num}/degradation_stages_FD00{dataset_num}.csv"
    data = pd.read_csv(file_path)
    return data

def add_rolling_features(data, window_size=5, n_jobs=-1):
    """Add rolling mean and std features for each sensor with parallel processing"""
    # Group by unit_number to ensure rolling stats are calculated per engine
    units = data['unit_number'].unique()
    
    def process_unit(unit_id):
        unit_data = data[data['unit_number'] == unit_id].copy()
        unit_data = unit_data.sort_values('time_cycles')
        
        # Get sensor columns
        sensor_cols = [col for col in unit_data.columns if col.startswith('sensor_')]
        
        # Calculate rolling statistics
        for col in sensor_cols:
            unit_data[f'{col}_roll_mean'] = unit_data[col].rolling(window=window_size, min_periods=1).mean()
            unit_data[f'{col}_roll_std'] = unit_data[col].rolling(window=window_size, min_periods=1).std().fillna(0)
            
        return unit_data
    
    # Process units in parallel
    processed_data = Parallel(n_jobs=n_jobs)(
        delayed(process_unit)(unit) for unit in units
    )
    
    # Combine results
    result = pd.concat(processed_data)
    
    return result

@memory.cache
def load_and_preprocess(dataset_num, window_size=5):
    """Load and preprocess data with caching"""
    data = load_data(dataset_num)
    data = add_rolling_features(data, window_size)
    return data

def prepare_data(data, test_size=0.2, random_state=42, sample_size=None):
    """Prepare data for modeling by splitting by unit_number with optional unit sampling"""
    # Get unique unit numbers
    units = data['unit_number'].unique()
    
    # ADDED: Sample a subset of units if sample_size is specified
    if sample_size is not None and sample_size < len(units):
        print(f"Sampling {sample_size} units out of {len(units)} total units")
        np.random.seed(random_state)
        units = np.random.choice(units, size=sample_size, replace=False)
    
    # Split units into train and test
    train_units, test_units = train_test_split(units, test_size=test_size, random_state=random_state)
    
    # Filter data by unit number
    train_data = data[data['unit_number'].isin(train_units)].copy().reset_index(drop=True)
    test_data = data[data['unit_number'].isin(test_units)].copy().reset_index(drop=True)
    
    # Extract features and target
    feature_cols = [col for col in data.columns if col.startswith(('setting_', 'sensor_')) or 
                   ('roll_' in col)]
    
    X_train = train_data[feature_cols].values
    y_train = train_data['degradation_stage'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['degradation_stage'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, feature_cols, scaler, train_data, test_data

def check_class_balance(y):
    """Check class balance and display distribution"""
    class_counts = np.bincount(y)
    total_samples = len(y)
    
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  Stage {i}: {count} samples ({count/total_samples:.2%})")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_counts)), class_counts)
    plt.xticks(range(len(class_counts)), [f'Stage {i}' for i in range(len(class_counts))])
    plt.title('Class Distribution')
    plt.xlabel('Degradation Stage')
    plt.ylabel('Count')
    plt.savefig(f'class_distribution.png')
    
    return class_counts

def run_classification(dataset_num=1,
                       model_type='random_forest',
                       handle_imbalance_strategy='smote',
                       n_jobs=-1,
                       sample_units=None,
                       hp_n_iter=10,
                       hp_val_size=1000,
                       hp_checkpoint_every=5,
                       hp_incremental=False):
    """Run the full classification pipeline with parallel processing and caching"""
    # 1. Results folder
    results_dir = f"./Code/results/classification/FD00{dataset_num}/{model_type}"
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load & preprocess
    print(f"Loading and preprocessing dataset FD00{dataset_num}...")
    data = load_and_preprocess(dataset_num)
    
    # 3. Extra feature engineering
    print("Adding extra features...")
    data = add_time_series_features(data,
            group_col='unit_number',
            sensor_cols=[c for c in data.columns if c.startswith('sensor_')],
            windows=[5,10,20])
    data = add_statistical_features(data,
            group_col='unit_number',
            sensor_cols=[c for c in data.columns if c.startswith('sensor_')])
    
    # 4. Prepare with unit sampling
    print("Preparing data for modeling...")
    X_train, y_train, X_test, y_test, feature_cols, scaler, train_df, test_df = prepare_data(
        data, 
        test_size=0.2, 
        random_state=42,
        sample_size=sample_units  # Pass sample size to prepare_data
    )
    # keep a copy of the original train split (before SMOTE) for later annotation
    X_train_orig = X_train.copy()
    
    # 5a. Remove zero‐variance features
    from sklearn.feature_selection import VarianceThreshold
    print("Removing zero-variance features…")
    vt = VarianceThreshold(threshold=0.0)
    X_train = vt.fit_transform(X_train)
    X_test  = vt.transform(X_test)
    feature_cols = [f for f, keep in zip(feature_cols, vt.get_support()) if keep]

    # 5b. Now select top‐k via ANOVA F‐test
    print("Selecting best features…")
    best_feats = select_k_best_features(
        X_train, y_train,
        feature_names=feature_cols,
        k=30,
        score_func='f_classif'
    )
    X_train = X_train[:, best_feats]
    X_test  = X_test[:, best_feats]
    feature_cols = [feature_cols[i] for i in best_feats]
    
    # 6. Check class balance
    print("\nChecking class balance...")
    class_counts = check_class_balance(y_train)
    
    # 7. Handle imbalance if needed
    print("\nHandling class imbalance...")
    if handle_imbalance_strategy == 'smote':
        X_train, y_train = handle_imbalance(X_train, y_train, 'smote')
        print("Applied SMOTE resampling")
        # Check new balance
        check_class_balance(y_train)
    elif handle_imbalance_strategy == 'class_weights':
        X_train, y_train, class_weights = handle_imbalance(X_train, y_train, 'class_weights')
        print("Using class weights:")
        for stage, weight in class_weights.items():
            print(f"  Stage {stage}: {weight:.4f}")
    
    # 8. Split training data to create a validation set
    print("\nCreating validation set for hyperparameter tuning...")
    # Use smaller training portion for fast model selection
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )
    
    # 9. Hyper-parameter search with sampling and optional incremental checkpointing
    # Optionally load best pre-defined hyperparameters to skip expensive search
    if hasattr(BEST_HYPERPARAMS, model_type) or model_type in BEST_HYPERPARAMS:
        if args.use_best_params:
            best_params = BEST_HYPERPARAMS[model_type][dataset_num]
            print(f"Using predefined hyperparameters: {best_params}")
            # Skip hyperparameter search and use predefined params directly
            model = CustomRandomForest(**best_params,
                                   n_jobs=n_jobs,
                                   class_weights=class_weights if handle_imbalance_strategy=='class_weights' else None)
        else:
            # Perform hyperparameter search
            print(f"\nPerforming hyperparameter search ({'incremental' if hp_incremental else 'randomized'})...")
            param_dist = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [5, 10],
                'max_features': ['sqrt', 'log2']
            }
            # Continue with hyperparameter search as before...
    else:
        # if no predefined, always search
        print(f"\nPerforming hyperparameter search ({'incremental' if hp_incremental else 'randomized'})...")
        param_dist = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
        }
        # sample random configurations
        param_list = list(ParameterSampler(param_dist, n_iter=hp_n_iter, random_state=42))
        # smaller validation subset
        val_size = min(hp_val_size, len(X_val))
        idx = np.random.choice(len(X_val), val_size, replace=False)
        X_val_small, y_val_small = X_val[idx], y_val[idx]

        # helper to evaluate one hyperparameter setting
        def evaluate_params(params):
            # train a single RF on the split training data
            rf = CustomRandomForest(
                **params,
                n_jobs=1,
                class_weights=class_weights if handle_imbalance_strategy=='class_weights' else None
            )
            rf.fit(X_tr, y_tr)
            preds = rf.predict(X_val_small)
            score = f1_score(y_val_small, preds, average='weighted')
            return params, score

        # checkpoint file
        hp_file = os.path.join(results_dir, 'hp_results.json')
        def run_search_incremental():
            try:
                with open(hp_file) as f: hp_results = json.load(f)
            except:
                hp_results = {}
            for i, params in enumerate(param_list):
                key = json.dumps(params, sort_keys=True)
                if key in hp_results: continue
                _, f1 = evaluate_params(params)
                hp_results[key] = f1
                if (i+1) % hp_checkpoint_every == 0:
                    with open(hp_file, 'w') as f: json.dump(hp_results, f, indent=2)
            with open(hp_file, 'w') as f: json.dump(hp_results, f, indent=2)
            best_key, best_f1 = max(hp_results.items(), key=lambda x: x[1])
            return json.loads(best_key), best_f1
        def run_search_parallel():
            results = Parallel(n_jobs=max(1, n_jobs//2))(delayed(evaluate_params)(p) for p in param_list)
            # save all
            out = {json.dumps(p, sort_keys=True): v for p, v in results}
            with open(hp_file, 'w') as f: json.dump(out, f, indent=2)
            return max(results, key=lambda x: x[1])
        if hp_incremental:
            best_params, best_f1 = run_search_incremental()
        else:
            best_params, best_f1 = run_search_parallel()
        print("Best params:", best_params, "val_f1=", best_f1)
        model = CustomRandomForest(**best_params,
                                   n_jobs=n_jobs,
                                   class_weights=class_weights if handle_imbalance_strategy=='class_weights' else None)

    # 10. Train final model
    print("\nTraining final model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Model training completed in {train_time:.2f} seconds")
    
    # 11. Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Save confusion matrix plot
    cm_fig = plt.gcf()
    cm_fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.close(cm_fig)
    
    # 12. Analyze feature importance
    print("\nAnalyzing feature importance...")
    importances = analyze_feature_importance(model, feature_cols)
    if importances is not None:
        fi_fig = plt.gcf()
        fi_fig.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=300)
        plt.close(fi_fig)
    
    # 13. Dump metrics.json
    out = {k: float(v) for k, v in metrics.items() if k != "confusion_matrix"}
    with open(os.path.join(results_dir, "metrics.json"), "w") as fp:
        json.dump(out, fp, indent=2)

    # 14. Save predictions for train & test
    train_df["predicted_stage"] = model.predict(X_train_orig)
    test_df["predicted_stage"]  = model.predict(X_test)
    train_df.to_csv(os.path.join(results_dir, "train_with_preds.csv"), index=False)
    test_df.to_csv(os.path.join(results_dir, "test_with_preds.csv"), index=False)
    
    # 15. Save model & scaler
    model_dir = "./Code/models/classification"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"model_FD00{dataset_num}_{model_type}.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, f"scaler_FD00{dataset_num}.joblib"))
    
    print(f"Model saved to {os.path.join(model_dir, f'model_FD00{dataset_num}_{model_type}.joblib')}")
    print(f"Scaler saved to {os.path.join(model_dir, f'scaler_FD00{dataset_num}.joblib')}")
    
    return model, metrics, importances

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run classification pipeline')
    parser.add_argument('--dataset', type=int, default=1, choices=[1, 2, 3, 4],
                      help='Dataset number (1-4)')
    parser.add_argument('--model', type=str, default='random_forest', 
                      choices=['random_forest', 'logistic_regression'],
                      help='Model type')
    parser.add_argument('--balance', type=str, default='smote', 
                      choices=['smote', 'class_weights', 'none'],
                      help='Method to handle class imbalance')
    parser.add_argument('--n_jobs', type=int, default=-1,
                      help='Number of parallel jobs. Default: -1 (use all CPUs)')
    parser.add_argument('--sample_units', type=int, default=None,
                      help='Number of engine units to sample (default: use all units)')
    parser.add_argument('--hp_n_iter', type=int, default=10,
                      help='Number of random hyperparameter configurations')
    parser.add_argument('--hp_val_size', type=int, default=1000,
                      help='Max number of validation samples for tuning')
    parser.add_argument('--hp_checkpoint_every', type=int, default=5,
                      help='Save hyperparameter results after this many iterations')
    parser.add_argument('--hp_incremental', action='store_true',
                      help='Enable incremental hyperparameter search and checkpointing')
    parser.add_argument('--use_best_params', action='store_true',
                      help='Use predefined best hyperparameters if available')
    args = parser.parse_args()
    
    # Run the pipeline
    run_classification(dataset_num=args.dataset,
                       model_type=args.model,
                       handle_imbalance_strategy=args.balance,
                       n_jobs=args.n_jobs,
                       sample_units=args.sample_units,
                       hp_n_iter=args.hp_n_iter,
                       hp_val_size=args.hp_val_size,
                       hp_checkpoint_every=args.hp_checkpoint_every,
                       hp_incremental=args.hp_incremental)