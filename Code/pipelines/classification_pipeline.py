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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Import custom modules
sys.path.append('/Users/garga4/Desktop/MUGithub/Year2/2-2/CS3102/p3-project/G_AI096_AI115_ECM025_CSE223/Code')
from classification.base_model import CustomLogisticRegression, CustomRandomForest
from classification.evaluator import evaluate_model, analyze_feature_importance
from classification.smote import handle_imbalance

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

def prepare_data(data, test_size=0.2, random_state=42):
    """Prepare data for modeling by splitting by unit_number"""
    # Get unique unit numbers
    units = data['unit_number'].unique()
    
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

def run_classification(dataset_num=1, model_type='random_forest', handle_imbalance_strategy='smote', n_jobs=-1):
    """Run the full classification pipeline with parallel processing and caching"""
    # 1. Results folder
    results_dir = f"./Code/results/classification/FD00{dataset_num}/{model_type}"
    os.makedirs(results_dir, exist_ok=True)

    # 2. Load & preprocess
    print(f"Loading and preprocessing dataset FD00{dataset_num}...")
    data = load_and_preprocess(dataset_num)
    
    # 3. Prepare (now returns train/test dfs)
    print("Preparing data for modeling...")
    X_train, y_train, X_test, y_test, feature_cols, scaler, train_df, test_df = prepare_data(data)
    # keep a copy of the original train split (before SMOTE) for later annotation
    X_train_orig = X_train.copy()
    
    # 4. Check class balance
    print("\nChecking class balance...")
    class_counts = check_class_balance(y_train)
    
    # 5. Handle imbalance if needed
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
    
    # 6. Train model
    print("\nTraining model...")
    if model_type == 'random_forest':
        if handle_imbalance_strategy == 'class_weights':
            model = CustomRandomForest(n_estimators=50, max_depth=10, 
                                       class_weights=class_weights, n_jobs=n_jobs)
        else:
            model = CustomRandomForest(n_estimators=50, max_depth=10, n_jobs=n_jobs)
    elif model_type == 'logistic_regression':
        if handle_imbalance_strategy == 'class_weights':
            model = CustomLogisticRegression(learning_rate=0.01, iterations=1000, class_weights=class_weights)
        else:
            model = CustomLogisticRegression(learning_rate=0.01, iterations=1000)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Model training completed in {train_time:.2f} seconds")
    
    # 7. Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # Save confusion matrix plot
    cm_fig = plt.gcf()
    cm_fig.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.close(cm_fig)
    
    # 8. Analyze feature importance
    print("\nAnalyzing feature importance...")
    importances = analyze_feature_importance(model, feature_cols)
    if importances is not None:
        fi_fig = plt.gcf()
        fi_fig.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=300)
        plt.close(fi_fig)
    
    # 9. Dump metrics.json
    out = {k: float(v) for k, v in metrics.items() if k != "confusion_matrix"}
    with open(os.path.join(results_dir, "metrics.json"), "w") as fp:
        json.dump(out, fp, indent=2)

    # 10. Save predictions for train & test
    train_df["predicted_stage"] = model.predict(X_train_orig)
    test_df["predicted_stage"]  = model.predict(X_test)
    train_df.to_csv(os.path.join(results_dir, "train_with_preds.csv"), index=False)
    test_df.to_csv(os.path.join(results_dir, "test_with_preds.csv"), index=False)
    
    # 11. Save model & scaler
    model_dir = "./Code/models/classification"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, f"model_FD00{dataset_num}_{model_type}.joblib"))
    joblib.dump(scaler, os.path.join(model_dir, f"scaler_FD00{dataset_num}.joblib"))
    
    print(f"Model saved to {os.path.join(model_dir, f'model_FD00{dataset_num}_{model_type}.joblib')}")
    print(f"Scaler saved to {os.path.join(model_dir, f'scaler_FD00{dataset_num}.joblib')}")
    
    return model, metrics, importances

if __name__ == "__main__":
    # Parse command line arguments if any
    import argparse
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
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_classification(dataset_num=args.dataset, model_type=args.model, 
                    handle_imbalance_strategy=args.balance,
                    n_jobs=args.n_jobs)