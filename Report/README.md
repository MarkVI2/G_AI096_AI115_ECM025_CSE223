# G_AI096_AI115_ECM025_CSE223

This project contains the source code for the Machine Learning (CS2202) project
titled, "Hybrid Predictive Maintenance using Enhanced CMAPSS NASA Dataset".

## Project Overview

This project implements a hybrid approach for predictive maintenance using the
NASA CMAPSS Turbofan Engine dataset. The approach combines clustering,
classification, and regression techniques to provide a comprehensive health
monitoring and failure prediction system.

## Project Contributions

- **Atharv Ashish Garg (SE23UCSE223)** - 40%

- **Shrivatsh Kuppu Subramaniam (SE23UARI115)** - 20%

- **Pranav Dubey (SE23UARI096)** - 20%

- **Janaki Beeram Reddy (SE23UECM025)** - 20%

## Implementation Plan

The implementation is divided into four distinct phases:

### Phase 1: Clustering for Multi-Stage Failure Labeling

- Uses raw sensor data from the CMAPSS dataset (not using standard RUL labels)
- Implements a Hierarchical Agglomerative Clustering
- Visualizes clusters using PCA or t-SNE for validation
- Derives 5 degradation stages:
  - Stage 0: Normal
  - Stage 1: Slightly degraded
  - Stage 2: Moderately degraded
  - Stage 3: Critical
  - Stage 4: Failure

### Phase 2: Classification Model

- Uses the cluster‑labeled data (from Phase 1) as ground truth labels
- Trains a meta‑classifier framework on engine data features
- Employs Random Forest as the primary classifier
- Predicts the current degradation stage (0 to 4) for each time cycle
- Uses both training and test datasets (cluster labels on test set are generated
  via the clustering pipeline) for training and evaluation

### Phase 3: Regression Model

- Predicts the time (cycles) remaining until the next degradation stage
- Implements an ensemble of regression models:
  - Random Forest Regressor
  - Ridge Regression
  - Quantile Regression
  - Histogram Gradient Boosting Regressor
- Provides quantile-based uncertainty estimates

### Phase 4: Risk Score Computation and Decision Logic

- Computes risk score based on current degradation stage and estimated time to
  next stage
- Uses regression output for estimated time left until next stage transition
- Creates a normalized risk score for maintenance decision-making

## Innovative Approaches

### Phase 1: Clustering Innovations

- **Hierarchical Agglomerative Clustering**: Uses Ward linkage method to create
  well-defined degradation stages
- **Advanced Feature Weighting**: Applies variance-based weights to features to
  highlight important degradation patterns
- **Cross-Validation Framework**: Implements k-fold cross-validation with
  comprehensive quality metrics

### Phase 2: Classification Innovations

- **Base Classification Framework**: Focuses on robust implementation with
  proper handling of imbalanced data
- **SMOTE Implementation**: Uses Synthetic Minority Over-sampling Technique for
  handling class imbalance
- **Comprehensive Evaluation**: Detailed evaluation metrics for classification
  performance

### Phase 3: Regression Innovations

- **Ensemble Regression Approach**: Combines multiple regression models for
  improved stability and accuracy
- **Gradient Boosting Regression**: Specialized implementation for handling
  time-series data
- **Quantile Regression**: Provides uncertainty bounds for remaining useful life
  predictions

## Running the Project

The project can be run using the main.py script with the following options:

```sh
python3 Code/main.py --phase [0-4] --data_path [path_to_data] --datasets [dataset_ids]
```

Where:

- `--phase`: Pipeline phase to run (0=all, 1=clustering, 2=classification,
  3=regression, 4=risk)
- `--data_path`: Path to CMAPSS dataset (default: ./Code/data/)
- `--datasets`: Comma-separated list of datasets to process (e.g., FD001,FD003)

## Dependencies

Required packages are listed in requirements.txt and can be installed with:

```sh
pip install -r requirements.txt
```

It is recommended that you use a venv when installing the requirements like so:

```sh
python3 -m venv venv
source venv/bin/activate

# or equivalent code block
```

## Development Notes

The project structure details are available in the project_structure.txt file,
which provides a comprehensive overview of all project components and their
relationships.
