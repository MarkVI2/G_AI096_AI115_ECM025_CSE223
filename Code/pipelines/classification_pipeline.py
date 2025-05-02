import os, sys
# allow imports from project root (Code/) folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import CMAPSSDataLoader
from data.preprocessor import CMAPSSPreprocessor, create_sequence_data
from classification.base_model import BasicGBDTClassifier, MultiClassGBDTClassifier
from classification.meta_classifier import MetaClassifier
from classification.evaluator import ClassificationEvaluator
from classification.smote import smote
from classification.classifiers import RandomForestScratch, SVMScratch
from classification.base_model import LogisticRegressionScratch
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

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
                 classifier_type: str = 'random_forest',  # options: random_forest, svm, logistic, scratch
                 oversample: bool = False,
                 n_pca_components: int = None,
                 random_state: int = 42):
        self.data_dir = data_dir
        self.dataset_id = dataset_id
        self.n_classes = n_classes
        self.sequence_length = sequence_length
        self.classifier_type = classifier_type
        self.oversample = oversample
        self.n_pca_components = n_pca_components
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        # Components
        self.loader = CMAPSSDataLoader(self.data_dir)
        self.preprocessor = CMAPSSPreprocessor(normalization_method='minmax', scale_per_unit=False)
        # scratch GBDT/meta for classifier_type='scratch'
        self.base_model = BasicGBDTClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
        self.meta_classifier = MetaClassifier(random_state=self.random_state)
        self.evaluator = ClassificationEvaluator()
        
        # Data containers
        self.train_seq = None
        self.train_labels = None
        self.test_seq = None
        self.test_labels = None
        self.model = None
        
    def load_and_preprocess(self):
        """
        Load train/test data, compute RUL, preprocess features.
        """
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
        
        # Sequence creation using helper
        feature_cols = [c for c in train_proc.columns if c not in ['unit_number','time_cycles','RUL','degradation_stage']]
        self.train_seq, self.train_labels = create_sequence_data(
            train_proc, sequence_length=self.sequence_length, feature_columns=feature_cols, target_column='degradation_stage'
        )
        self.test_seq, self.test_labels = create_sequence_data(
            test_proc, sequence_length=self.sequence_length, feature_columns=feature_cols, target_column='degradation_stage'
        )
        return self

    def train(self):
        """
        Train the meta-classifier on sequence data.
        """
        # Flatten features
        n_samples = self.train_seq.shape[0]
        X = self.train_seq.reshape(n_samples, -1)
        y = self.train_labels
        # Oversample class imbalance
        if self.oversample:
            # Use custom SMOTE implementation
            X, y = smote(X, y, k_neighbors=5, random_state=self.random_state)
        # Select and fit scratch classifier
        if self.classifier_type == 'random_forest':
            model = RandomForestScratch(
                n_estimators=100,
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=int(X.shape[1]**0.5),
                random_state=self.random_state
            )
        elif self.classifier_type == 'svm':
            model = SVMScratch(
                C=1.0,
                learning_rate=0.001,
                n_iter=500,
                class_weight='balanced'
            )
        elif self.classifier_type == 'logistic':
            model = LogisticRegressionScratch(
                learning_rate=0.01,
                n_iterations=500,
                reg_lambda=1e-3
            )
        elif self.classifier_type == 'scratch':
            # scratch gradient-boost ensemble
            self.base_model.fit(self.train_seq, y)
            self.meta_classifier.fit(self.train_seq, y)
            self.model = self.meta_classifier
            return self
        else:
            raise ValueError(f"Unknown classifier_type '{self.classifier_type}'")
        model.fit(X, y)
        self.model = model
        return self

    def evaluate(self):
        """
        Evaluate trained model on test data and print metrics.
        """
        # Flatten and transform test data same as train
        n_samples = self.test_seq.shape[0]
        X_test = self.test_seq.reshape(n_samples, -1)
        if self.n_pca_components:
            X_test = pca.transform(X_test)
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)
        # Compute and print custom metrics
        metrics = self.evaluator.evaluate(self.test_labels, preds, probs)
        print("Classification Evaluation:")
        print(metrics)
        # Compute confusion matrix from scratch
        cm = np.zeros((self.n_classes, self.n_classes), dtype=int)
        for true, pred in zip(self.test_labels, preds):
            cm[true, pred] += 1
        # Per-class precision, recall, f1
        print("\nPer-class metrics:")
        for cls in range(self.n_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            print(f"Stage {cls} - precision: {prec:.3f}, recall: {rec:.3f}, f1: {f1:.3f}, support: {self.test_labels.tolist().count(cls)}")
        # Plot confusion matrix heatmap
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        return metrics

    def save_model(self, output_path: str):
        """
        Save the trained model to disk.
        """
        joblib.dump(self.model, output_path)
        return self

def run_classification(data_dir: str, cluster_labels=None, dataset_id: str = "FD001", n_classes: int = 5, sequence_length: int = 30, random_state: int = 42):
    """
    Convenience wrapper to run the classification pipeline end-to-end.
    """
    pipeline = ClassificationPipeline(
        data_dir=data_dir,
        dataset_id=dataset_id,
        n_classes=n_classes,
        sequence_length=sequence_length,
        random_state=random_state
    )
    # Optionally merge provided cluster_labels (unused for now)
    pipeline.load_and_preprocess()
    pipeline.train()
    metrics = pipeline.evaluate()
    return metrics
