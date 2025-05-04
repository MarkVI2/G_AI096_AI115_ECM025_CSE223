import pandas as pd

def add_time_series_features(data: pd.DataFrame,
                             group_col: str,
                             sensor_cols: list,
                             windows: list = (5, 10, 20)
                            ) -> pd.DataFrame:
    """
    For each sensor in sensor_cols and each window in windows,
    compute rolling mean & std grouped by group_col.
    """
    df = data.copy()
    df = df.sort_values([group_col, 'time_cycles'])
    
    for w in windows:
        for col in sensor_cols:
            mean_col = f"{col}_roll{w}_mean"
            std_col  = f"{col}_roll{w}_std"
            df[mean_col] = (
                df.groupby(group_col)[col]
                  .rolling(window=w, min_periods=1)
                  .mean()
                  .reset_index(level=0, drop=True)
            )
            df[std_col] = (
                df.groupby(group_col)[col]
                  .rolling(window=w, min_periods=1)
                  .std()
                  .fillna(0)
                  .reset_index(level=0, drop=True)
            )
    return df