import sys, os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler()]
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from pipelines.clustering_pipeline import run_clustering
from pipelines.classification_pipeline import run_classification
from pipelines.regression_pipeline import run_regression
from pipelines.risk_pipeline import run_risk_assessment
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description='Run Predictive Maintenance Pipeline')
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3, 4], default=0,
                        help='Pipeline phase to run (0=all, 1=clustering, 2=classification, 3=regression, 4=risk)')
    parser.add_argument('--data_path', type=str, default='./Code/data/',
                        help='Path to CMAPSS dataset')
    parser.add_argument('--oversample', dest='oversample', action='store_true',
                        help='Enable SMOTE oversampling in classification')
    parser.add_argument('--no-oversample', dest='oversample', action='store_false',
                        help='Disable SMOTE oversampling in classification')
    parser.set_defaults(oversample=True)
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of splits for cross-validation')
    parser.add_argument('--n_pca_components', type=int, default=None,
                        help='Number of PCA components for dimension reduction')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs. Default: max(1, cpu_count-1)')
    parser.add_argument('--memory_limit', type=float, default=None,
                        help='Memory limit in GB for parallel operations')
    parser.add_argument('--tuning_iter', type=int, default=10,
                        help='Number of hyperparameter search iterations (smaller = faster)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for memory-efficient processing')
    args = parser.parse_args()
    
    # Configure parallel processing settings
    if args.n_jobs is None:
        # Use n_cores - 1 by default to avoid system freeze
        n_jobs = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_jobs = args.n_jobs
        
    print(f"Using {n_jobs} parallel jobs for processing")
    
    if args.phase == 0 or args.phase == 1:
        print("Running Phase 1: Clustering for degradation stages...")
        cluster_results = run_clustering(args.data_path, n_jobs=n_jobs)
        
    if args.phase == 0 or args.phase == 2:
        print("Running Phase 2: Classification model for degradation stage prediction...")
        classification_results = run_classification(
            data_dir=args.data_path,
            cluster_labels=cluster_results if args.phase == 0 else None,
            oversample=args.oversample,
            n_splits=args.n_splits,
            n_pca_components=args.n_pca_components,
            n_jobs=n_jobs,
            memory_limit_gb=args.memory_limit,
            tuning_iterations=args.tuning_iter,
            batch_size=args.batch_size
        )
        
    if args.phase == 0 or args.phase == 3:
        print("Running Phase 3: Regression model for time-to-failure prediction...")
        regression_results = run_regression(
            args.data_path, 
            classification_results if args.phase == 0 else None,
            n_jobs=n_jobs
        )
        
    if args.phase == 0 or args.phase == 4:
        print("Running Phase 4: Risk score computation and decision logic...")
        run_risk_assessment(classification_results if args.phase == 0 else None, 
                           regression_results if args.phase == 0 else None)
        
    print("Pipeline execution complete!")

if __name__ == "__main__":
    main()