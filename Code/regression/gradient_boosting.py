import numpy as np
from regression.base_models import BaseRegressor

class DecisionTreeHistogram:
    """Decision tree using histogram binning for splits"""
    
    def __init__(self, max_depth=5, max_bins=256, min_samples_split=2):
        """
        Initialize histogram-based decision tree.
        
        Args:
            max_depth: Maximum depth of the tree
            max_bins: Maximum number of bins for histograms
            min_samples_split: Minimum samples required to split a node
        """
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_split = min_samples_split
        self.tree = None
        self.bin_edges = None
        self.bin_mapped_data = None
        
    def _create_histogram_bins(self, X):
        """Create histogram bins for each feature"""
        n_samples, n_features = X.shape
        self.bin_edges = []
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            # Create bins based on quantiles to handle outliers better
            bins = np.percentile(feature_values, np.linspace(0, 100, self.max_bins + 1))
            # Remove duplicate bin edges
            bins = np.unique(bins)
            self.bin_edges.append(bins)
            
        return self.bin_edges
    
    def _map_to_bins(self, X):
        """Map data to histogram bins"""
        n_samples, n_features = X.shape
        binned_data = np.zeros_like(X, dtype=np.int32)
        
        for feature_idx in range(n_features):
            # Use digitize to assign bin indices
            binned_data[:, feature_idx] = np.digitize(
                X[:, feature_idx], 
                self.bin_edges[feature_idx], 
                right=False
            ) - 1  # -1 because digitize returns 1-based indices
            
            # Handle values that are less than the smallest bin edge
            binned_data[:, feature_idx] = np.maximum(0, binned_data[:, feature_idx])
            
            # Handle values that are greater than the largest bin edge
            binned_data[:, feature_idx] = np.minimum(
                len(self.bin_edges[feature_idx]) - 2, 
                binned_data[:, feature_idx]
            )
            
        self.bin_mapped_data = binned_data
        return binned_data
    
    def fit(self, X, y, sample_weight=None):
        """Fit histogram-based decision tree"""
        n_samples, n_features = X.shape
        
        # Create histogram bins for each feature
        self._create_histogram_bins(X)
        
        # Map data to bins
        binned_X = self._map_to_bins(X)
        
        # Build the tree recursively
        self.tree = self._build_tree(binned_X, y, sample_weight, depth=0)
        
        return self
    
    def _build_tree(self, X, y, sample_weight=None, depth=0):
        """Recursively build the tree"""
        n_samples, n_features = X.shape
        
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
            
        # Calculate weighted sum and sum of weights
        sum_y = np.sum(y * sample_weight)
        sum_weight = np.sum(sample_weight)
        
        # Calculate the weighted mean prediction for this node
        if sum_weight > 0:
            node_value = sum_y / sum_weight
        else:
            node_value = 0.0
            
        # Create leaf node if stopping criteria met
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            np.all(y == y[0])):
            return {'value': node_value, 'is_leaf': True}
            
        # Find best split
        best_feature, best_bin, best_gain = self._find_best_split(X, y, sample_weight)
        
        # If no good split found, create a leaf node
        if best_gain <= 0:
            return {'value': node_value, 'is_leaf': True}
            
        # Split the data
        left_mask = X[:, best_feature] <= best_bin
        right_mask = ~left_mask
        
        # Check if split produces empty node
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {'value': node_value, 'is_leaf': True}
            
        # Recursively build left and right subtrees
        left_tree = self._build_tree(
            X[left_mask], 
            y[left_mask], 
            sample_weight[left_mask], 
            depth + 1
        )
        
        right_tree = self._build_tree(
            X[right_mask], 
            y[right_mask], 
            sample_weight[right_mask], 
            depth + 1
        )
        
        # Simplified tree structure
        return {
            'value': node_value,
            'is_leaf': False,
            'feature': best_feature,
            'bin': best_bin,
            'left': left_tree,
            'right': right_tree
        }
    
    def _find_best_split(self, X, y, sample_weight):
        """Find the best feature and bin threshold for splitting"""
        n_samples, n_features = X.shape
        
        # Initialize variables to track best split
        best_feature = -1
        best_bin = -1
        best_gain = -float('inf')
        
        # Calculate total variance before split
        weighted_mean = np.sum(y * sample_weight) / np.sum(sample_weight)
        total_variance = np.sum(sample_weight * (y - weighted_mean) ** 2)
        
        # If total variance is 0, no gain possible
        if total_variance == 0:
            return best_feature, best_bin, 0
            
        # For each feature
        for feature_idx in range(n_features):
            # For each possible bin threshold
            unique_bins = np.unique(X[:, feature_idx])
            
            for bin_threshold in unique_bins:
                # Split data
                left_mask = X[:, feature_idx] <= bin_threshold
                right_mask = ~left_mask
                
                # Skip if split is degenerate
                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue
                    
                # Calculate weighted mean for left and right
                left_weight_sum = np.sum(sample_weight[left_mask])
                right_weight_sum = np.sum(sample_weight[right_mask])
                
                left_mean = np.sum(y[left_mask] * sample_weight[left_mask]) / left_weight_sum
                right_mean = np.sum(y[right_mask] * sample_weight[right_mask]) / right_weight_sum
                
                # Calculate variance reduction
                left_variance = np.sum(sample_weight[left_mask] * (y[left_mask] - left_mean) ** 2)
                right_variance = np.sum(sample_weight[right_mask] * (y[right_mask] - right_mean) ** 2)
                
                variance_after_split = left_variance + right_variance
                gain = total_variance - variance_after_split
                
                # Update best split if better
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_bin = bin_threshold
                    
        return best_feature, best_bin, best_gain
    
    def predict(self, X):
        """Predict using the decision tree"""
        if self.tree is None:
            raise ValueError("Tree has not been fitted. Call fit() first.")
            
        # Map data to bins
        binned_X = self._map_to_bins(X)
        
        # Predict each sample
        return np.array([self._predict_sample(x, self.tree) for x in binned_X])
    
    def _predict_sample(self, x, node):
        """Predict single sample by traversing the tree"""
        # If leaf node, return value
        if node['is_leaf']:
            return node['value']
            
        # Determine which branch to follow
        if x[node['feature']] <= node['bin']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])


class HistGradientBoostingRegressor(BaseRegressor):
    """
    Histogram-based Gradient Boosting regressor.
    
    Uses histogram binning for efficient training and
    gradient boosting for ensemble learning.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, 
                 max_bins=256, min_samples_split=2, subsample=1.0):
        """
        Initialize Histogram Gradient Boosting Regressor.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Contribution of each tree
            max_depth: Maximum tree depth
            max_bins: Maximum number of bins for histograms
            min_samples_split: Minimum samples to split a node
            subsample: Fraction of samples to use for fitting individual trees
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X, y):
        """
        Fit Histogram Gradient Boosting Regressor.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            y: Target values array of shape (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Initialize prediction with mean value
        self.initial_prediction = np.mean(y)
        
        # Initialize current predictions
        y_pred = np.full_like(y, self.initial_prediction, dtype=np.float64)
        
        # For each boosting iteration
        for i in range(self.n_estimators):
            print(f"Training tree {i+1}/{self.n_estimators}", end='\r')
            
            # Calculate negative gradients (residuals for MSE loss)
            residuals = y - y_pred
            
            # Create sample weights if using subsampling
            if self.subsample < 1.0:
                sample_indices = np.random.choice(
                    n_samples, 
                    size=int(n_samples * self.subsample),
                    replace=False
                )
                subsample_weights = np.zeros(n_samples)
                subsample_weights[sample_indices] = 1.0
            else:
                subsample_weights = np.ones(n_samples)
            
            # Fit a new tree to the negative gradients
            tree = DecisionTreeHistogram(
                max_depth=self.max_depth,
                max_bins=self.max_bins,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals, sample_weight=subsample_weights)
            
            # Add tree to ensemble
            self.trees.append(tree)
            
            # Update predictions
            tree_pred = tree.predict(X)
            y_pred += self.learning_rate * tree_pred
            
        print("")  # New line after progress updates
        return self
    
    def predict(self, X):
        """
        Make predictions using the gradient boosting ensemble.
        
        Args:
            X: Features array of shape (n_samples, n_features)
            
        Returns:
            Predictions array of shape (n_samples,)
        """
        if len(self.trees) == 0:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Start with initial prediction
        y_pred = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)
        
        # Add contributions from each tree
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
            
        return y_pred