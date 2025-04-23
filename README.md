# G_AI096_AI115_ECM025_CSE223

This project contains the source code for the Machine Learning (CS2202) project
titled, "Hybrid Predictive Maintenance using Enhanced CMAPSS NASA Dataset".

## Project Overview

This project implements a hybrid approach for predictive maintenance using the
NASA CMAPSS Turbofan Engine dataset. The approach combines clustering,
classification, and regression techniques to provide a comprehensive health
monitoring and failure prediction system.

## Implementation Plan

The implementation is divided into four distinct phases:

### Phase 1: Clustering for Multi-Stage Failure Labeling

- Uses raw sensor data from the CMAPSS dataset (not using standard RUL labels)
- Implements a hybrid clustering approach combining DBSCAN and Spectral
  Clustering
- Visualizes clusters using PCA or t-SNE for validation
- Derives 5 degradation stages:
  - Stage 0: Normal
  - Stage 1: Slightly degraded
  - Stage 2: Moderately degraded
  - Stage 3: Critical
  - Stage 4: Failure

### Phase 2: Classification Model

- Uses the cluster-labeled data to train a meta-classifier framework
- Employs XGBoost as the primary classifier, enhanced with ensemble techniques
- Predicts the current degradation stage (0 to 4)
- Provides probability estimates for each degradation stage

### Phase 3: Regression Model

- Predicts the time (cycles) remaining until the next degradation stage
- Implements an ensemble of regression models:
  - Random Forest Regressor
  - Ridge Regression
  - Support Vector Regression (SVR)
- Provides quantile-based uncertainty estimates

### Phase 4: Risk Score Computation and Decision Logic

- Computes risk score using the formula: Risk Score = Failure Probability ×
  (1/Time Left to Failure)
- Extracts failure probability from the classifier (probability of Stage 4)
- Uses regression output for estimated time left until failure
- Creates a normalized risk score for maintenance decision-making

## Project Structure

```
G_AI096_AI115_ECM025_CSE223/
│
├── Code/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py               # Global configuration parameters
│   │   └── hyperparameters.py        # Model hyperparameters
│   │
│   ├── data/
│   │   ├── loader.py                 # Data loading utilities for CMAPSS dataset
│   │   ├── preprocessor.py           # Data cleaning and preprocessing
│   │   ├── splitter.py               # Train/test/validation splitting
│   │   ├── RUL/                      # RUL label files
│   │   ├── test/                     # Test dataset files
│   │   └── train/                    # Training dataset files
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── time_series.py            # Time series feature extraction
│   │   ├── statistical.py            # Statistical feature generation
│   │   ├── engineering.py            # Domain-specific feature engineering
│   │   └── selector.py               # Feature selection techniques
│   │
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── dbscan.py                 # DBSCAN implementation
│   │   ├── spectral.py               # Spectral clustering implementation
│   │   ├── ensemble.py               # Ensemble clustering methods
│   │   └── stage_mapper.py           # Maps clusters to degradation stages
│   │
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── base_model.py             # Individual classifier implementations
│   │   ├── meta_classifier.py        # Stacking/ensemble classifier
│   │   ├── online_learning.py        # Incremental learning components
│   │   └── evaluator.py              # Classification metrics and evaluation
│   │
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── base_models.py            # Individual regressor implementations
│   │   ├── ensemble_regressor.py     # Combined regression approach
│   │   ├── quantile.py               # Quantile regression for uncertainty
│   │   └── evaluator.py              # Regression metrics and evaluation
│   │
│   ├── risk/
│   │   ├── calibration.py            # Calibration of probability estimates
│   │   ├── scoring.py                # Risk score computation
│   │   └── thresholds.py             # Decision thresholds for maintenance
│   │
│   ├── visualisation/
│   │   ├── __init__.py
│   │   ├── clusters.py               # Cluster visualization with PCA/t-SNE
│   │   ├── degradation.py            # Degradation stage visualization
│   │   ├── predictions.py            # Model prediction visualizations
│   │   └── risk_dashboard.py         # Risk score dashboard/plots
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py                 # Logging configuration
│   │   ├── metrics.py                # Custom evaluation metrics
│   │   └── validators.py             # Input validation utilities
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── clustering_pipeline.py    # Phase 1 pipeline
│   │   ├── classification_pipeline.py # Phase 2 pipeline
│   │   ├── regression_pipeline.py    # Phase 3 pipeline
│   │   └── risk_pipeline.py          # Phase 4 pipeline
│   │
│   ├── main.py                       # Main entry point
│   ├── requirements.txt              # Package dependencies
│   └── G_AI096_AI115_ECM025_CSE223_code.ipynb  # Complete workflow notebook
│
└── Report/                          # Project report and documentation
```

## Innovative Approaches

### Phase 1: Clustering Innovations

- **Hybrid Clustering Approach**: Combines DBSCAN for noise detection with
  Spectral Clustering for refined stage identification
- **Time-Series Feature Extraction**: Applies change-point detection and
  extracts statistical features that capture degradation patterns
- **Sensor Importance Weighting**: Uses SHAP or permutation importance to weight
  sensors based on their relevance to failure patterns

### Phase 2: Classification Innovations

- **Meta-Classification Framework**: Trains multiple diverse classifiers and
  uses stacking to combine their predictions
- **Online Learning Component**: Implements incremental learning for XGBoost to
  adapt to new patterns
- **Custom Feature Transformations**: Creates interaction features between
  sensors and applies non-linear transformations

### Phase 3: Regression Innovations

- **Multi-Target Regression**: Predicts time to each future stage, not just the
  next stage
- **Two-Stage Regression**: First model predicts if failure will happen within a
  threshold time, second predicts exact time
- **Domain-Adaptive Regression**: Customizes regression models for different
  operational conditions

## Running the Project

The project can be run using the main.py script with the following options:

```sh
python3 main.py --phase [0-4] --data_path [path_to_data]
```

Where:

- `--phase`: Pipeline phase to run (0=all, 1=clustering, 2=classification,
  3=regression, 4=risk)
- `--data_path`: Path to CMAPSS dataset

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

## Contributors

Group members' contributions are detailed in the Report/README.md file.
