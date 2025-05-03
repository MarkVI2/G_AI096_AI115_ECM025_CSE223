import os, sys
# allow imports from project root (Code/) folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import CMAPSSDataLoader
from data.preprocessor import CMAPSSPreprocessor, create_sequence_data
from classification.base_model import BasicGBDTClassifier, MultiClassGBDTClassifier
from classification.evaluator import ClassificationEvaluator
from classification.smote import smote
from classification.classifiers import RandomForestScratch, SVMScratch
from classification.base_model import LogisticRegressionScratch
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import parallel_backend
import gc
from functools import partial
import logging

logger = logging.getLogger(__name__)

# Define an efficient SMOTE batch processing function
def _process_smote_batch(X_batch, y_batch, random_state):
    """Process SMOTE in batches to save memory"""
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=random_state).fit_resample(X_batch, y_batch)
    
# Create a memory-efficient pipeline with custom SMOTE handling
class MemoryEfficientPipeline(ImbPipeline):
    """Pipeline extension that handles SMOTE in memory-efficient batches"""
    def __init__(self, steps, memory=None, batch_size=None, random_state=42):
        super().__init__(steps, memory=memory)
        self.batch_size = batch_size
        self.random_state = random_state
        
    def fit_resample(self, X, y, **fit_params):
        """Apply SMOTE in batches to avoid memory issues"""
        if self.batch_size and hasattr(self, 'steps') and self.steps[0][0] == 'smote':
            # Process data in batches for large datasets
            if len(X) > self.batch_size:
                print(f"Using batched SMOTE processing with {self.batch_size} samples per batch")
                batches = []
                for i in range(0, len(X), self.batch_size):
                    end = min(i + self.batch_size, len(X))
                    X_batch, y_batch = _process_smote_batch(
                        X[i:end], y[i:end], self.random_state + i
                    )
                    batches.append((X_batch, y_batch))
                    # Force garbage collection to free memory
                    gc.collect()
                    
                X_resampled = np.vstack([b[0] for b in batches])
                y_resampled = np.hstack([b[1] for b in batches])
                
                # Skip SMOTE step since we've done it manually
                # Pass directly to the classifier
                return self.steps[-1][1].fit(X_resampled, y_resampled)
        
        # Default implementation for small datasets or non-SMOTE pipelines
        return super().fit_resample(X, y, **fit_params)

class ClassificationPipeline:
    """
    Pipeline for classification of degradation stages (Phase 2).
    Uses cluster labels from Phase 1 or RUL-based discretization as targets.
    """
    def __init__(self,
                 data_dir: str,
                 dataset_id: str = "FD001",
                 n_classes: int = 5,
                 sequence_length: int = 30,
                 feature_selection_k: int = None,
                 oversample: bool = False,
                 n_pca_components: int = None,
                 n_jobs: int = -1,
                 random_state: int = 42):
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.feature_selection_k = feature_selection_k
        self.oversample = oversample
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        np.random.seed(self.random_state)
        
        # PCA for dimensionality reduction
        self.n_pca_components = n_pca_components
        self.pca = PCA(n_components=self.n_pca_components) if self.n_pca_components else None

        # Components
        self.loader = CMAPSSDataLoader(self.data_dir)
        self.preprocessor = CMAPSSPreprocessor(normalization_method='minmax', scale_per_unit=False)
        self.evaluator = ClassificationEvaluator()
        
        # Data containers
        self.train_seq = None
        self.train_labels = None
        self.train_groups = None
        self.test_seq = None
        self.test_labels = None
        self.test_groups = None
        self.model = None
        
    def load_and_preprocess(self):
        """
        Load train/test data, compute RUL, preprocess features.
        """
        logger.info("Loading and preprocessing data for classification")
        # Load dataset and label data
        self.loader.load_dataset(self.dataset_id)
        train_df = self.loader.calculate_rul_for_training(self.dataset_id)
        test_df = self.loader.prepare_test_with_rul(self.dataset_id)
        
        # Preprocess
        train_proc = self.preprocessor.fit_transform(train_df, add_remaining_features=True, add_sensor_diff=True)
        test_proc = self.preprocessor.transform(test_df, add_remaining_features=True, add_sensor_diff=True)
        
        # Discretize RUL into classes for classification target
        train_proc['degradation_stage'] = pd.cut(
            train_proc['RUL'], bins=self.n_classes, labels=False, include_lowest=True
        )
        test_proc['degradation_stage'] = pd.cut(
            test_proc['RUL'], bins=self.n_classes, labels=False, include_lowest=True
        )
        
        # Feature selection to reduce dimensionality if requested
        feature_cols = [c for c in train_proc.columns if c not in ['unit_number','time_cycles','RUL','degradation_stage']]
        if self.feature_selection_k:
            # use ANOVA F-score to pick top K features
            feature_cols = self.preprocessor.select_top_k_features(
                train_proc, feature_cols, target_column='degradation_stage', k=self.feature_selection_k
            )
        # Sequence creation using helper
        logger.info("Creating sequence data for train and test")
        self.train_seq, self.train_labels, self.train_groups = create_sequence_data(
            train_proc, sequence_length=self.sequence_length, feature_columns=feature_cols, target_column='degradation_stage'
        )
        self.test_seq, self.test_labels, self.test_groups = create_sequence_data(
            test_proc, sequence_length=self.sequence_length, feature_columns=feature_cols, target_column='degradation_stage'
        )
        return self

    def train(self):
        """
        Train the meta-classifier on sequence data.
        """
        logger.info("Training classification model with %d samples", self.train_seq.shape[0])
        # Flatten features
        n_samples = self.train_seq.shape[0]
        X = self.train_seq.reshape(n_samples, -1)
        # apply PCA if configured
        if self.pca and not hasattr(self.pca, 'components_'):
            # fit and transform training data
            X = self.pca.fit_transform(X)
            # save PCA explained variance curve
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'classification')
            os.makedirs(results_dir, exist_ok=True)
            plt.figure()
            plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('PCA Explained Variance')
            plt.savefig(os.path.join(results_dir, 'pca_variance.png'))
            plt.close()
        elif self.pca:
            X = self.pca.transform(X)
        y = self.train_labels
        # Oversample class imbalance using imblearn SMOTE
        if self.oversample:
            logger.info("Applying SMOTE to training data")
            X, y = SMOTE(random_state=self.random_state).fit_resample(X, y)
        # Select and fit scratch classifier
        model = MultiClassGBDTClassifier(
            n_classes=self.n_classes,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8
        )
        logger.info("Fitting MultiClassGBDTClassifier: %s trees", model.n_estimators)
        model.fit(X, y)
        self.model = model
        logger.info("Training complete")
        return self

    def evaluate(self):
        """
        Evaluate trained model on test data and print metrics.
        """
        logger.info("Evaluating model on test set with %d samples", self.test_seq.shape[0])
        # Flatten and transform test data same as train
        n_samples = self.test_seq.shape[0]
        X_test = self.test_seq.reshape(n_samples, -1)
        if self.pca:
            X_test = self.pca.transform(X_test)
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)
        # Compute and print custom metrics
        metrics = self.evaluator.evaluate(self.test_labels, preds, probs)
        logger.info("Classification evaluation metrics: %s", metrics)
        # Compute confusion matrix from scratch
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true, pred in zip(self.test_labels, preds):
            cm[true, pred] += 1
        # Per-class precision, recall, f1
        for cls in range(self.n_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            logger.info("Stage %d - precision: %.3f, recall: %.3f, f1: %.3f, support: %d", 
                        cls, prec, rec, f1, self.test_labels.tolist().count(cls))
        # Plot confusion matrix heatmap
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # save confusion matrix
        import os
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'classification')
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
        logger.info("Confusion matrix saved to %s", results_dir)
        plt.close()
        return metrics

    def save_model(self, output_path: str):
        """
        Save the trained model to disk.
        """
        joblib.dump(self.model, output_path)
        return self

    def run_cross_validation(self, n_splits: int = 5) -> List[Dict]:
        """
        Perform grouped k-fold cross-validation using GroupKFold and SMOTE.
        Returns list of per-fold metrics.
        """
        # Flatten and apply PCA
        X = self.train_seq.reshape(self.train_seq.shape[0], -1)
        if self.pca and not hasattr(self.pca, 'components_'):
            X = self.pca.fit_transform(X)
        elif self.pca:
            X = self.pca.transform(X)
        y = self.train_labels
        groups = self.train_groups
        gkf = GroupKFold(n_splits=n_splits)
        cv_results = []
        
        # Define a function to train and evaluate one fold for parallel processing
        def _process_one_fold(fold, train_idx, val_idx):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            if self.oversample:
                X_tr, y_tr = SMOTE(random_state=self.random_state).fit_resample(X_tr, y_tr)
            model = MultiClassGBDTClassifier(
                n_classes=self.n_classes,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                n_jobs=1  # Use single thread within each fold to avoid nested parallelism
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            probs = model.predict_proba(X_val)
            metrics = self.evaluator.evaluate(y_val, preds, probs)
            print(f"Fold {fold} metrics: {metrics}")
            return {'fold': fold, 'metrics': metrics}
        
        # Process folds in parallel using joblib
        from joblib import Parallel, delayed
        
        logger.info("Starting cross-validation with %d folds and %d jobs", n_splits, self.n_jobs)
        cv_splits = list(enumerate(gkf.split(X, y, groups)))
        cv_results = Parallel(n_jobs=self.n_jobs)(
            delayed(_process_one_fold)(fold, train_idx, val_idx)
            for fold, (train_idx, val_idx) in cv_splits
        )
        logger.info("Cross-validation complete")
        return cv_results

def run_classification(data_dir: str,
                      cluster_labels=None,
                      dataset_id: str = "FD001",
                      n_classes: int = 5,
                      sequence_length: int = 30,
                      random_state: int = 42,
                      oversample: bool = True,
                      n_splits: int = 5,
                      n_pca_components: int = None,
                      n_jobs: int = -1,
                      memory_limit_gb: float = None,
                      tuning_iterations: int = 10,
                      batch_size: int = None):
    """
    Convenience wrapper to run the classification pipeline end-to-end.
    Includes stratified k-fold cross-validation and final test evaluation.
    """
    
    # Setup memory management
    if memory_limit_gb:
        memory_mb = int(memory_limit_gb * 1024)
        memory_limit = f"{memory_mb}MB"
    else:
        memory_limit = None
    
    # Determine batch size for memory-efficient processing
    # If not set, we'll estimate based on data size later
    
    pipeline = ClassificationPipeline(
        data_dir=data_dir,
        dataset_id=dataset_id,
        n_classes=n_classes,
        sequence_length=sequence_length,
        oversample=oversample,
        random_state=random_state,
        n_pca_components=n_pca_components,
        n_jobs=n_jobs
    )
    
    # Load data
    pipeline.load_and_preprocess()
    # Check for cached trained model
    model_cache_dir = os.path.join(data_dir, 'cache', 'classification')
    os.makedirs(model_cache_dir, exist_ok=True)
    cache_file = os.path.join(model_cache_dir, f"{dataset_id}_model.joblib")
    if os.path.exists(cache_file):
        pipeline.model = joblib.load(cache_file)
        logger.info("Loaded cached model from %s, skipping training and tuning", cache_file)
        # Evaluate directly
        test_metrics = pipeline.evaluate()
        return {'cv_avg_metrics': {}, 'test_metrics': test_metrics}
    
    # Estimate reasonable batch size if none provided
    if batch_size is None:
        # Rough heuristic: aim for batches that are ~100MB in memory
        # Assuming each float is 8 bytes
        n_samples = pipeline.train_seq.shape[0]
        n_features = pipeline.train_seq.size // n_samples
        bytes_per_sample = n_features * 8  # 8 bytes per float64
        target_batch_bytes = 100 * 1024 * 1024  # 100MB
        batch_size = max(1, min(n_samples, target_batch_bytes // bytes_per_sample))
        print(f"Auto-determined batch size: {batch_size} samples")
    
    # Parallel hyperparameter tuning with efficient memory management
    logger.info("Running parallel hyperparameter optimization with %d jobs", n_jobs)
    
    # Flatten and transform data
    n_samples = pipeline.train_seq.shape[0]
    X = pipeline.train_seq.reshape(n_samples, -1)
    if pipeline.pca:
        X = pipeline.pca.fit_transform(X)
    y = pipeline.train_labels
    groups = pipeline.train_groups
    
    # Import hyperparameter optimization module
    from classification.hyperparameter_tuning import optimize_hyperparameters
    from sklearn.model_selection import GroupKFold
    
    # Create GroupKFold to prevent data leakage
    gkf = GroupKFold(n_splits=n_splits)
    
    # Setup cross-validation folds before SMOTE to ensure proper stratification
    # This ensures groups are properly maintained
    cv_splits = list(gkf.split(X, y, groups)) if groups is not None else n_splits
    
    # Limit parameter grid size based on requested tuning_iterations
    logger.info("Optimizing hyperparameters (max_iter=%d)", tuning_iterations)
    
    # Run parallel hyperparameter optimization with joblib backend
    best_params, best_model = optimize_hyperparameters(
        X=X, 
        y=y, 
        n_classes=n_classes,
        cv=cv_splits,
        n_jobs=n_jobs,
        max_iter=tuning_iterations,
        apply_smote=oversample,
        random_state=random_state,
        scoring='f1_macro'  # Better for imbalanced data
    )
    
    print("Best parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Assign best model
    pipeline.model = best_model
    # Cache trained model for future runs
    joblib.dump(best_model, cache_file)
    logger.info("Cached trained model to %s", cache_file)

    # Force memory cleanup
    gc.collect()
    
    # Evaluate final model on test set
    print("\nEvaluating final model on test set...")
    test_metrics = pipeline.evaluate()
    print(f"\nFinal Test Metrics: {test_metrics}")
    
    return {'cv_avg_metrics': {'best_params': best_params}, 'test_metrics': test_metrics}
