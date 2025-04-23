import argparse
from pipelines.clustering_pipeline import run_clustering
from pipelines.classification_pipeline import run_classification
from pipelines.regression_pipeline import run_regression
from pipelines.risk_pipeline import run_risk_assessment

def main():
    parser = argparse.ArgumentParser(description='Run Predictive Maintenance Pipeline')
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3, 4], default=0,
                        help='Pipeline phase to run (0=all, 1=clustering, 2=classification, 3=regression, 4=risk)')
    parser.add_argument('--data_path', type=str, default='../CMAPSSData/',
                        help='Path to CMAPSS dataset')
    args = parser.parse_args()
    
    if args.phase == 0 or args.phase == 1:
        print("Running Phase 1: Clustering for degradation stages...")
        cluster_results = run_clustering(args.data_path)
        
    if args.phase == 0 or args.phase == 2:
        print("Running Phase 2: Classification model for degradation stage prediction...")
        classification_results = run_classification(args.data_path, cluster_results if args.phase == 0 else None)
        
    if args.phase == 0 or args.phase == 3:
        print("Running Phase 3: Regression model for time-to-failure prediction...")
        regression_results = run_regression(args.data_path, classification_results if args.phase == 0 else None)
        
    if args.phase == 0 or args.phase == 4:
        print("Running Phase 4: Risk score computation and decision logic...")
        run_risk_assessment(classification_results if args.phase == 0 else None, 
                           regression_results if args.phase == 0 else None)
        
    print("Pipeline execution complete!")

if __name__ == "__main__":
    main()