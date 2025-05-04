import numpy as np

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, class_weights=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.class_weights = class_weights
        self.weights = None
        self.bias = None
        
    def softmax(self, z):
        # Avoid numerical instability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
    def compute_cost(self, X, y, weights, bias):
        m = X.shape[0]
        # One-hot encode y
        y_onehot = np.zeros((m, self.num_classes))
        y_onehot[np.arange(m), y] = 1
        
        # Forward propagation
        z = np.dot(X, weights) + bias
        A = self.softmax(z)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Apply weights to each sample based on its class
            sample_weights = np.array([self.class_weights[cls] for cls in y])
            sample_weights = sample_weights.reshape(-1, 1)
            cost = -np.sum(sample_weights * y_onehot * np.log(A + 1e-8)) / m
        else:
            cost = -np.sum(y_onehot * np.log(A + 1e-8)) / m
            
        return cost, A
    
    def fit(self, X, y):
        m, n = X.shape
        self.num_classes = len(np.unique(y))
        
        # Initialize weights and bias
        self.weights = np.zeros((n, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))
        
        # Calculate balanced weights if class_weights='balanced'
        if self.class_weights == 'balanced':
            class_counts = np.bincount(y)
            total_samples = len(y)
            self.class_weights = {i: total_samples / (len(class_counts) * count) 
                                 for i, count in enumerate(class_counts)}
        
        # Training loop
        for i in range(self.iterations):
            # Forward propagation
            cost, A = self.compute_cost(X, y, self.weights, self.bias)
            
            # One-hot encode y
            y_onehot = np.zeros((m, self.num_classes))
            y_onehot[np.arange(m), y] = 1
            
            # Apply class weights if provided
            if self.class_weights is not None:
                sample_weights = np.array([self.class_weights[cls] for cls in y])
                sample_weights = sample_weights.reshape(-1, 1)
                dZ = (A - y_onehot) * sample_weights
            else:
                dZ = A - y_onehot
            
            # Backpropagation
            dW = np.dot(X.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Update weights
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db
            
            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.root = None
    
    def fit(self, X, y, sample_weights=None):
        self.n_features = X.shape[1]
        if self.max_features is None:
            self.max_features = self.n_features
        else:
            self.max_features = min(self.max_features, self.n_features)
        
        self.root = self._grow_tree(X, y, sample_weights, depth=0)
    
    def _grow_tree(self, X, y, sample_weights, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # New safety check - if no samples, return a default leaf
        if n_samples == 0:
            return Node(value=0)  # Default to class 0
        
        # Check stopping criteria
        if (depth >= self.max_depth if self.max_depth else False or 
            n_samples < self.min_samples_split or n_classes == 1):
            return Node(value=self._calculate_leaf_value(y, sample_weights))
        
        # Randomly select features to consider
        feat_idxs = np.random.choice(n_features, min(self.max_features, n_features), replace=False)
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y, feat_idxs, sample_weights)
        
        # If no good split is found
        if best_feature is None:
            return Node(value=self._calculate_leaf_value(y, sample_weights))
        
        # Create child nodes
        left_idxs = X[:, best_feature] <= best_threshold
        right_idxs = ~left_idxs
        
        # Make sure we don't create empty nodes
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return Node(value=self._calculate_leaf_value(y, sample_weights))
        
        left = self._grow_tree(X[left_idxs], y[left_idxs], 
                              None if sample_weights is None else sample_weights[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], 
                               None if sample_weights is None else sample_weights[right_idxs], depth+1)
        
        return Node(best_feature, best_threshold, left, right)
    
    def _best_split(self, X, y, feat_idxs, sample_weights):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for threshold in thresholds:
                # Calculate information gain
                gain = self._information_gain(y, X_column, threshold, sample_weights)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _information_gain(self, y, X_column, threshold, sample_weights):
        # Calculate parent entropy
        parent_entropy = self._entropy(y, sample_weights)
        
        # Create children
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        
        if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
            return 0
        
        # Calculate children entropy and weights
        n = len(y)
        n_l = np.sum(left_idxs)
        n_r = n - n_l
        
        if sample_weights is not None:
            w_l = np.sum(sample_weights[left_idxs])
            w_r = np.sum(sample_weights[right_idxs])
            w = np.sum(sample_weights)
            
            e_l = self._entropy(y[left_idxs], sample_weights[left_idxs])
            e_r = self._entropy(y[right_idxs], sample_weights[right_idxs])
            
            # Weighted entropy calculation
            child_entropy = (w_l / w) * e_l + (w_r / w) * e_r
        else:
            e_l = self._entropy(y[left_idxs], None)
            e_r = self._entropy(y[right_idxs], None)
            
            # Weighted entropy calculation
            child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Information gain
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    def _entropy(self, y, sample_weights):
        class_counts = np.bincount(y, weights=sample_weights, minlength=5)  # 5 classes (0-4)
        class_probs = class_counts / np.sum(class_counts)
        
        # Filter out zero probabilities
        class_probs = class_probs[class_probs > 0]
        
        # Calculate entropy
        entropy = -np.sum(class_probs * np.log2(class_probs))
        return entropy
    
    def _calculate_leaf_value(self, y, sample_weights):
        # Check if y is empty
        if len(y) == 0:
            return 0  # Default to class 0 if no samples
            
        if sample_weights is not None:
            # Return weighted majority class
            unique_classes, counts = np.unique(y, return_counts=True)
            
            # Check if we have any classes
            if len(unique_classes) == 0:
                return 0  # Default to class 0 if no classes
                
            weighted_counts = np.zeros_like(unique_classes, dtype=float)
            
            for i, cls in enumerate(unique_classes):
                mask = y == cls
                if np.any(mask):  # Make sure mask has at least one True value
                    weighted_counts[i] = np.sum(sample_weights[mask])
            
            # Check if all weights are zero
            if np.sum(weighted_counts) == 0:
                # Fall back to unweighted majority
                return unique_classes[np.argmax(counts)]
                
            return unique_classes[np.argmax(weighted_counts)]
        else:
            # Return majority class
            values, counts = np.unique(y, return_counts=True)
            if len(values) == 0:
                return 0  # Default to class 0 if no classes
            return values[np.argmax(counts)]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class CustomRandomForest:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 max_features='sqrt', class_weights=None, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.class_weights = class_weights
        self.n_jobs = n_jobs
        self.trees = []
    
    def fit(self, X, y):
        from joblib import Parallel, delayed
        
        # Calculate class weights if specified
        if self.class_weights == 'balanced':
            class_counts = np.bincount(y)
            n_samples = len(y)
            self.class_weights = {i: n_samples / (len(class_counts) * count) 
                                 for i, count in enumerate(class_counts)}
        
        # Calculate sample weights based on class weights
        if self.class_weights is not None:
            sample_weights = np.array([self.class_weights[cls] for cls in y])
        else:
            sample_weights = np.ones(len(y))
        
        # Determine max_features
        n_features = X.shape[1]
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            # Ensure max_features is not larger than the number of features
            max_features = min(self.max_features, n_features) if isinstance(self.max_features, int) else n_features
        
        # Function to train a single tree
        def train_tree(_):
            # Bootstrap sampling with weights
            n_samples = X.shape[0]
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True, 
                                            p=sample_weights/np.sum(sample_weights))
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            bootstrap_weights = sample_weights[bootstrap_indices]
            
            # Create and train a new tree
            tree = DecisionTree(max_depth=self.max_depth, 
                            min_samples_split=self.min_samples_split,
                            max_features=max_features)
            tree.fit(X_bootstrap, y_bootstrap, bootstrap_weights)
            return tree
        
        # Train trees in parallel
        self.trees = Parallel(n_jobs=self.n_jobs)(
            delayed(train_tree)(i) for i in range(self.n_estimators)
        )
        
        return self
    
    def predict(self, X):
        # Make predictions with each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Take majority vote for each sample
        final_pred = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            unique_vals, counts = np.unique(tree_preds[:, i], return_counts=True)
            final_pred[i] = unique_vals[np.argmax(counts)]
        
        return final_pred
    
    def predict_proba(self, X):
        # Get predictions from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate class probabilities
        n_samples = X.shape[0]
        n_classes = 5  # We have 5 classes (0-4)
        class_probs = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j in range(n_classes):
                class_probs[i, j] = np.sum(tree_preds[:, i] == j) / self.n_estimators
                
        return class_probs