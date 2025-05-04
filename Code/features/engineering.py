import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def add_polynomial_features(df, sensor_cols, degree=2):
    """
    Add polynomial and interaction features for key sensors
    to help the model capture complex relationships
    """
    # Select only numeric sensor columns for polynomial transformation
    sensor_data = df[sensor_cols].fillna(0)
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(sensor_data)
    
    # Convert to DataFrame with appropriate column names
    feature_names = poly.get_feature_names_out(sensor_cols)
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Drop original features to avoid duplication
    poly_df = poly_df.loc[:, ~poly_df.columns.isin(sensor_cols)]
    
    # Concatenate with original dataframe
    result = pd.concat([df, poly_df], axis=1)
    return result

def add_ratio_features(df, sensor_cols):
    """
    Add ratio features between sensor readings to capture 
    relationships that may be important for distinguishing stages
    """
    result = df.copy()
    # Add ratios between important sensors
    for i, col1 in enumerate(sensor_cols):
        for col2 in sensor_cols[i+1:]:
            # Avoid division by zero
            denominator = df[col2].replace(0, np.nan)
            ratio_name = f"ratio_{col1}_{col2}"
            result[ratio_name] = (df[col1] / denominator).fillna(0)
    
    return result

def add_acceleration_features(df, sensor_cols, group_col='unit_number'):
    """
    Add acceleration features (second derivative) to capture
    how quickly sensor readings are changing over time
    """
    result = df.copy()
    
    for col in sensor_cols:
        # First get the diff (first derivative)
        diff_col = f"{col}_diff"
        if diff_col not in df.columns:
            result[diff_col] = result.groupby(group_col)[col].diff().fillna(0)
        
        # Then get the diff of the diff (second derivative/acceleration)
        accel_col = f"{col}_accel"
        result[accel_col] = result.groupby(group_col)[diff_col].diff().fillna(0)
    
    return result

def add_moving_average_crossings(df, sensor_cols, windows=[5, 10, 20], group_col='unit_number'):
    """
    Add features that track when short-term averages cross long-term averages,
    which can be good indicators of trend changes
    """
    result = df.copy()
    
    for col in sensor_cols:
        for short_window in windows:
            for long_window in [w for w in windows if w > short_window]:
                # Skip if we don't have enough data points
                if len(df) < long_window:
                    continue
                
                # Calculate short and long moving averages
                short_ma = f"{col}_ma{short_window}"
                long_ma = f"{col}_ma{long_window}"
                
                # Calculate moving averages for each unit separately
                for unit in df[group_col].unique():
                    unit_mask = df[group_col] == unit
                    unit_data = df.loc[unit_mask, col]
                    
                    # Only calculate if we have enough data points
                    if len(unit_data) >= long_window:
                        result.loc[unit_mask, short_ma] = unit_data.rolling(window=short_window, min_periods=1).mean()
                        result.loc[unit_mask, long_ma] = unit_data.rolling(window=long_window, min_periods=1).mean()
                
                # Fill NaN values
                result[short_ma] = result[short_ma].fillna(0)
                result[long_ma] = result[long_ma].fillna(0)
                
                # Create crossing indicator
                cross_col = f"{col}_cross_{short_window}_{long_window}"
                result[cross_col] = ((result[short_ma] > result[long_ma]) & 
                                    (result[short_ma].shift(1) <= result[long_ma].shift(1))).astype(int)
                
                # Drop the moving average columns to save space
                result = result.drop([short_ma, long_ma], axis=1)
    
    return result