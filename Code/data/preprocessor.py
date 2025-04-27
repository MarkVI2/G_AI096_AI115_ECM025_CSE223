import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class CMAPSSPreprocessor:
    """
    Preprocessor for CMAPSS dataset that performs data cleaning, feature engineering, 
    and normalization to prepare data for modeling.
    """
    
    def __init__(self, normalization_method: str = 'minmax', scale_per_unit: bool = False):
        """
        Initialize the preprocessor.
        
        Args:
            normalization_method: Method to use for scaling features.
                Options: 'minmax', 'standard', 'robust', or 'none'
            scale_per_unit: Whether to scale each engine unit independently
        """
        self.normalization_method = normalization_method
        self.scale_per_unit = scale_per_unit
        self.scalers = {}
        self.sensor_columns = None
        self.operating_setting_columns = None
        self.sensor_stats = {}
        
    def fit(self, train_df: pd.DataFrame, sensor_columns: Optional[List[str]] = None,
            operating_setting_columns: Optional[List[str]] = None) -> 'CMAPSSPreprocessor':
        """
        Fit the preprocessor on training data.
        
        Args:
            train_df: Training dataframe
            sensor_columns: List of sensor columns to preprocess
            operating_setting_columns: List of operating setting columns
            
        Returns:
            Self
        """
        if sensor_columns is None:
            self.sensor_columns = [col for col in train_df.columns if col.startswith('sensor_')]
        else:
            self.sensor_columns = sensor_columns
            
        if operating_setting_columns is None:
            self.operating_setting_columns = [col for col in train_df.columns if col.startswith('setting_')]
        else:
            self.operating_setting_columns = operating_setting_columns
            
        # Calculate statistics for sensors
        self._calculate_sensor_statistics(train_df)
        
        # Fit scalers
        self._fit_scalers(train_df)
        
        return self
    
    def transform(self, df: pd.DataFrame, add_remaining_features: bool = True,
                  add_sensor_diff: bool = True, window_size: int = 5) -> pd.DataFrame:
        """
        Transform the data by cleaning, scaling, and adding engineered features.
        
        Args:
            df: DataFrame to transform
            add_remaining_features: Whether to add remaining useful life features
            add_sensor_diff: Whether to add differences between consecutive sensor readings
            window_size: Window size for rolling statistics
            
        Returns:
            Transformed DataFrame
        """
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Clean data
        result_df = self._clean_data(result_df)
        
        # Add engineered features
        if add_remaining_features:
            result_df = self._add_remaining_features(result_df)
            
        if add_sensor_diff:
            result_df = self._add_sensor_differences(result_df, window_size)
        
        # Scale features
        if self.normalization_method != 'none':
            result_df = self._scale_features(result_df)
        
        return result_df
    
    def fit_transform(self, train_df: pd.DataFrame, sensor_columns: Optional[List[str]] = None,
                      operating_setting_columns: Optional[List[str]] = None,
                      add_remaining_features: bool = True, add_sensor_diff: bool = True,
                      window_size: int = 5) -> pd.DataFrame:
        """
        Fit the preprocessor and transform the data in one step.
        
        Args:
            train_df: Training dataframe
            sensor_columns: List of sensor columns to preprocess
            operating_setting_columns: List of operating setting columns
            add_remaining_features: Whether to add remaining useful life features
            add_sensor_diff: Whether to add differences between consecutive sensor readings
            window_size: Window size for rolling statistics
            
        Returns:
            Transformed DataFrame
        """
        self.fit(train_df, sensor_columns, operating_setting_columns)
        return self.transform(train_df, add_remaining_features, add_sensor_diff, window_size)
    
    def _calculate_sensor_statistics(self, df: pd.DataFrame) -> None:
        """Calculate statistical properties of sensor data."""
        for sensor in self.sensor_columns:
            self.sensor_stats[sensor] = {
                'mean': df[sensor].mean(),
                'std': df[sensor].std(),
                'min': df[sensor].min(),
                'max': df[sensor].max(),
                'median': df[sensor].median(),
                'skew': df[sensor].skew(),
                'kurtosis': df[sensor].kurtosis()
            }
    
    def _fit_scalers(self, df: pd.DataFrame) -> None:
        """Fit scalers on the training data."""
        # Create appropriate scaler
        if self.normalization_method == 'minmax':
            scaler_class = MinMaxScaler
        elif self.normalization_method == 'standard':
            scaler_class = StandardScaler
        elif self.normalization_method == 'robust':
            scaler_class = RobustScaler
        else:
            return  # No scaling

        if self.scale_per_unit:
            # Fit a separate scaler for each engine unit
            for unit in df['unit_number'].unique():
                unit_data = df[df['unit_number'] == unit]
                
                # Sensor scaler
                sensor_scaler = scaler_class()
                sensor_scaler.fit(unit_data[self.sensor_columns])
                self.scalers[f'sensor_{unit}'] = sensor_scaler
                
                # Operating settings scaler
                if self.operating_setting_columns:
                    settings_scaler = scaler_class()
                    settings_scaler.fit(unit_data[self.operating_setting_columns])
                    self.scalers[f'settings_{unit}'] = settings_scaler
        else:
            # Fit one scaler for all engine units
            # Sensor scaler
            sensor_scaler = scaler_class()
            sensor_scaler.fit(df[self.sensor_columns])
            self.scalers['sensor'] = sensor_scaler
            
            # Operating settings scaler
            if self.operating_setting_columns:
                settings_scaler = scaler_class()
                settings_scaler.fit(df[self.operating_setting_columns])
                self.scalers['settings'] = settings_scaler
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data by handling missing values, outliers etc.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        result_df = df.copy()
        
        # Handle missing values with a simple imputer
        if result_df[self.sensor_columns].isnull().any().any():
            imputer = SimpleImputer(strategy='median')
            result_df[self.sensor_columns] = imputer.fit_transform(result_df[self.sensor_columns])
            
        # Handle extreme outliers using capping
        for sensor in self.sensor_columns:
            if sensor in self.sensor_stats:
                mean, std = self.sensor_stats[sensor]['mean'], self.sensor_stats[sensor]['std']
                # Cap at 5 standard deviations
                lower_bound = mean - 5 * std
                upper_bound = mean + 5 * std
                result_df[sensor] = result_df[sensor].clip(lower_bound, upper_bound)
                
        return result_df
    
    def _add_remaining_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features related to remaining useful life and cycles.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added features
        """
        result_df = df.copy()
        
        # Calculate max cycles for each unit
        max_cycles = result_df.groupby('unit_number')['time_cycles'].transform('max')
        
        # Add cycle-based features
        result_df['cycle_ratio'] = result_df['time_cycles'] / max_cycles
        result_df['remaining_cycles'] = max_cycles - result_df['time_cycles']
        
        # Add RUL-based features if RUL is present
        if 'RUL' in result_df.columns:
            # Normalized RUL (0-1 range)
            max_rul = result_df['RUL'].max()
            result_df['normalized_RUL'] = result_df['RUL'] / max_rul
            
            # RUL quartile (discretized RUL)
            result_df['RUL_quartile'] = pd.qcut(result_df['RUL'], q=4, labels=False)
            
        return result_df
    
    def _add_sensor_differences(self, df: pd.DataFrame, window_size: int = 5) -> pd.DataFrame:
        """
        Add features capturing changes in sensor readings over time.
        
        Args:
            df: Input DataFrame
            window_size: Size of the window for rolling calculations
            
        Returns:
            DataFrame with added features
        """
        result_df = df.copy()
        
        # For each unit, calculate sensor differences and moving averages
        for unit in result_df['unit_number'].unique():
            unit_mask = result_df['unit_number'] == unit
            unit_data = result_df.loc[unit_mask].sort_values('time_cycles')
            
            # For each sensor, calculate differences and moving averages
            for sensor in self.sensor_columns:
                # Calculate difference from previous reading
                diff_col = f'{sensor}_diff'
                unit_data[diff_col] = unit_data[sensor].diff().fillna(0)
                
                # Calculate rolling mean
                roll_mean_col = f'{sensor}_roll_mean_{window_size}'
                unit_data[roll_mean_col] = unit_data[sensor].rolling(window=window_size, min_periods=1).mean()
                
                # Calculate rolling standard deviation
                roll_std_col = f'{sensor}_roll_std_{window_size}'
                unit_data[roll_std_col] = unit_data[sensor].rolling(window=window_size, min_periods=1).std().fillna(0)
                
            # Update the original DataFrame
            result_df.loc[unit_mask] = unit_data
        
        return result_df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features using the fitted scalers.
        
        Args:
            df: DataFrame to scale
            
        Returns:
            Scaled DataFrame
        """
        result_df = df.copy()
        
        if self.scale_per_unit:
            # Scale each unit separately
            for unit in result_df['unit_number'].unique():
                unit_mask = result_df['unit_number'] == unit
                
                # Scale sensor columns
                sensor_key = f'sensor_{unit}'
                if sensor_key in self.scalers:
                    result_df.loc[unit_mask, self.sensor_columns] = self.scalers[sensor_key].transform(
                        result_df.loc[unit_mask, self.sensor_columns]
                    )
                elif 'sensor' in self.scalers:  # Fallback to global scaler
                    result_df.loc[unit_mask, self.sensor_columns] = self.scalers['sensor'].transform(
                        result_df.loc[unit_mask, self.sensor_columns]
                    )
                
                # Scale operating settings
                if self.operating_setting_columns:
                    settings_key = f'settings_{unit}'
                    if settings_key in self.scalers:
                        result_df.loc[unit_mask, self.operating_setting_columns] = self.scalers[settings_key].transform(
                            result_df.loc[unit_mask, self.operating_setting_columns]
                        )
                    elif 'settings' in self.scalers:  # Fallback to global scaler
                        result_df.loc[unit_mask, self.operating_setting_columns] = self.scalers['settings'].transform(
                            result_df.loc[unit_mask, self.operating_setting_columns]
                        )
        else:
            # Scale all units with the same scaler
            if 'sensor' in self.scalers:
                result_df[self.sensor_columns] = self.scalers['sensor'].transform(result_df[self.sensor_columns])
                
            if self.operating_setting_columns and 'settings' in self.scalers:
                result_df[self.operating_setting_columns] = self.scalers['settings'].transform(
                    result_df[self.operating_setting_columns]
                )
                
        return result_df
    
    def get_sensor_correlations_with_rul(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate the correlation of each sensor with RUL.
        
        Args:
            df: DataFrame containing sensor data and RUL
            
        Returns:
            Series with sensors and their correlation with RUL
        """
        if 'RUL' not in df.columns:
            raise ValueError("RUL column not found in DataFrame")
            
        correlations = pd.Series(dtype='float64')
        for sensor in self.sensor_columns:
            correlations[sensor] = df[sensor].corr(df['RUL'])
            
        return correlations.sort_values(ascending=False)
    
    def plot_sensor_trends(self, df: pd.DataFrame, sensors: Optional[List[str]] = None, 
                          sample_units: int = 5, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot sensor trends over time for a sample of engine units.
        
        Args:
            df: DataFrame with sensor data
            sensors: List of sensors to plot (if None, uses top 6 most correlated with RUL)
            sample_units: Number of engine units to sample
            figsize: Figure size (width, height) in inches
        """
        if sensors is None:
            if 'RUL' in df.columns:
                correlations = self.get_sensor_correlations_with_rul(df)
                sensors = correlations.head(6).index.tolist()
            else:
                sensors = self.sensor_columns[:6]  # Use first 6 sensors
        
        # Sample engine units
        sample_units = min(sample_units, df['unit_number'].nunique())
        sampled_units = np.random.choice(df['unit_number'].unique(), sample_units, replace=False)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        for i, sensor in enumerate(sensors):
            plt.subplot(len(sensors), 1, i+1)
            
            for unit in sampled_units:
                unit_data = df[df['unit_number'] == unit].sort_values('time_cycles')
                plt.plot(unit_data['time_cycles'], unit_data[sensor], label=f'Unit {unit}')
                
            plt.title(f'{sensor} readings over time')
            plt.ylabel(sensor)
            if i == 0:
                plt.legend(loc='upper right')
            if i == len(sensors) - 1:
                plt.xlabel('Cycle')
                
        plt.tight_layout()
        plt.show()
        
    def plot_degradation_stages(self, df: pd.DataFrame, rul_column: str = 'RUL',
                               stage_column: str = 'degradation_stage', 
                               figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Visualize the degradation stages for different engine units.
        
        Args:
            df: DataFrame containing degradation stage information
            rul_column: Name of the column containing RUL values
            stage_column: Name of the column containing degradation stage labels
            figsize: Figure size (width, height) in inches
        """
        if stage_column not in df.columns:
            raise ValueError(f"Stage column '{stage_column}' not found in DataFrame")
            
        plt.figure(figsize=figsize)
        
        # Get unique units and stages
        units = df['unit_number'].unique()
        n_units = min(10, len(units))  # Limit to 10 units for clarity
        stages = sorted(df[stage_column].unique())
        n_stages = len(stages)
        
        # Create a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, n_stages))
        
        for i, unit in enumerate(units[:n_units]):
            unit_data = df[df['unit_number'] == unit]
            
            # Sort by cycle
            unit_data = unit_data.sort_values('time_cycles')
            
            # Plot RUL and stages
            plt.subplot(n_units, 1, i+1)
            
            for stage_idx, stage in enumerate(stages):
                stage_data = unit_data[unit_data[stage_column] == stage]
                if len(stage_data) > 0:
                    plt.scatter(
                        stage_data['time_cycles'],
                        stage_data[rul_column],
                        color=colors[stage_idx],
                        label=f"Stage {stage}",
                        alpha=0.7
                    )
            
            if i == 0:
                plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
                
            plt.ylabel(f"Unit {unit}\nRUL")
            
            if i == n_units - 1:
                plt.xlabel("Cycle")
        
        plt.tight_layout()
        plt.show()

    def get_important_features(self, df: pd.DataFrame, target_col: str = 'RUL', 
                              method: str = 'correlation', top_n: int = 10) -> pd.Series:
        """
        Identify the most important features for predicting the target.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            method: Method to measure importance ('correlation' or 'mutual_info')
            top_n: Number of top features to return
            
        Returns:
            Series with feature importance scores
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
            
        feature_cols = self.sensor_columns.copy()
        
        # Add derived features if present
        for col in df.columns:
            if (col.endswith('_diff') or col.endswith('_roll_mean_5') or 
                col.endswith('_roll_std_5')) and col in df.columns:
                feature_cols.append(col)
        
        if method == 'correlation':
            importance = pd.Series(dtype='float64')
            for col in feature_cols:
                importance[col] = abs(df[col].corr(df[target_col]))
                
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            
            # Select only numeric columns
            X = df[feature_cols]
            y = df[target_col]
            
            mi_scores = mutual_info_regression(X, y)
            importance = pd.Series(mi_scores, index=feature_cols)
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
        return importance.sort_values(ascending=False).head(top_n)


# Helper functions
def prepare_data_for_classification(df: pd.DataFrame, 
                                   n_classes: int = 5, 
                                   rul_column: str = 'RUL',
                                   label_column: str = 'degradation_stage',
                                   method: str = 'equal_width') -> pd.DataFrame:
    """
    Prepare data for classification by discretizing RUL into classes.
    
    Args:
        df: DataFrame with RUL values
        n_classes: Number of classes (stages) to create
        rul_column: Name of the column containing RUL values
        label_column: Name of the output column for class labels
        method: Method for discretization ('equal_width', 'equal_freq', or 'custom')
        
    Returns:
        DataFrame with added class labels
    """
    result_df = df.copy()
    
    if rul_column not in result_df.columns:
        raise ValueError(f"RUL column '{rul_column}' not found in DataFrame")
    
    if method == 'equal_width':
        # Equal width binning
        result_df[label_column] = pd.cut(
            result_df[rul_column], 
            bins=n_classes, 
            labels=range(n_classes-1, -1, -1)  # Reverse order: higher RUL = lower stage
        )
    elif method == 'equal_freq':
        # Equal frequency binning
        result_df[label_column] = pd.qcut(
            result_df[rul_column], 
            q=n_classes, 
            labels=range(n_classes-1, -1, -1),  # Reverse order
            duplicates='drop'
        )
    elif method == 'custom':
        # Custom thresholds (example: for RUL-based stages)
        max_rul = result_df[rul_column].max()
        thresholds = [0, max_rul * 0.2, max_rul * 0.4, max_rul * 0.6, max_rul * 0.8, max_rul]
        labels = list(range(n_classes-1, -1, -1))  # [4, 3, 2, 1, 0]
        
        result_df[label_column] = pd.cut(
            result_df[rul_column],
            bins=thresholds,
            labels=labels,
            include_lowest=True
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to integer
    result_df[label_column] = result_df[label_column].astype(int)
    
    return result_df


def create_sequence_data(df: pd.DataFrame, sequence_length: int, 
                        feature_columns: List[str], target_column: str,
                        step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of data for sequence modeling (RNN, LSTM, etc.)
    
    Args:
        df: DataFrame with time series data
        sequence_length: Length of sequences to create
        feature_columns: Columns to use as features
        target_column: Column to use as target
        step: Step size between sequences
        
    Returns:
        Tuple of (X_sequences, y_sequences)
    """
    sequences = []
    targets = []
    
    # Process each engine unit separately
    for unit in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == unit].sort_values('time_cycles')
        
        # Skip units with fewer cycles than sequence_length
        if len(unit_data) < sequence_length:
            continue
            
        # Extract features and targets
        unit_features = unit_data[feature_columns].values
        unit_targets = unit_data[target_column].values
        
        # Create sequences
        for i in range(0, len(unit_data) - sequence_length + 1, step):
            sequences.append(unit_features[i:i+sequence_length])
            targets.append(unit_targets[i+sequence_length-1])  # Target is the last value in sequence
    
    return np.array(sequences), np.array(targets)


if __name__ == "__main__":
    # Example usage
    from loader import CMAPSSDataLoader
    import os
    
    # Load data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    loader = CMAPSSDataLoader(data_dir)
    
    # Load dataset FD001
    dataset = loader.load_dataset("FD001")
    train_df = dataset['train']
    test_df = dataset['test']
    
    # Calculate RUL for training data
    train_with_rul = loader.calculate_rul_for_training("FD001")
    
    # Prepare test data with RUL
    test_with_rul = loader.prepare_test_with_rul("FD001")
    
    # Initialize preprocessor
    preprocessor = CMAPSSPreprocessor(normalization_method='minmax', scale_per_unit=False)
    
    # Preprocess training data
    train_processed = preprocessor.fit_transform(train_with_rul)
    
    # Preprocess test data
    test_processed = preprocessor.transform(test_with_rul)
    
    # Prepare data for classification (discretize RUL into stages)
    train_classified = prepare_data_for_classification(
        train_processed, 
        n_classes=5, 
        method='equal_width'
    )
    
    # Get important features for prediction
    important_features = preprocessor.get_important_features(train_processed, top_n=10)
    print("Top 10 important features:")
    print(important_features)
    
    # Plot sensor trends
    preprocessor.plot_sensor_trends(train_with_rul, sample_units=3)
    
    # Plot degradation stages
    preprocessor.plot_degradation_stages(train_classified)
    
    # Create sequence data for time series modeling
    feature_cols = important_features.index.tolist()
    X_seq, y_seq = create_sequence_data(
        train_classified, 
        sequence_length=30, 
        feature_columns=feature_cols,
        target_column='RUL',
        step=1
    )
    
    print(f"Sequence data shape: X={X_seq.shape}, y={y_seq.shape}")