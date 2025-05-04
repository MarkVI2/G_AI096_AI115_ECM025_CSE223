import pandas as pd
import numpy as np

def add_statistical_features(data: pd.DataFrame,
                             group_col: str,
                             sensor_cols: list
                            ) -> pd.DataFrame:
    """
    Add diff, expanding mean & expanding std for each sensor.
    """
    # make a sorted copy
    df = data.sort_values([group_col, 'time_cycles']).copy()
    # collect new columns in a dict
    new_cols = {}
    for col in sensor_cols:
        diff_col = f"{col}_diff"
        cummean = f"{col}_cummean"
        cumstd = f"{col}_cumstd"
        grp = df.groupby(group_col)[col]
        new_cols[diff_col] = grp.diff().fillna(0)
        new_cols[cummean] = grp.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
        new_cols[cumstd] = grp.expanding(min_periods=1).std().fillna(0).reset_index(level=0, drop=True)
    # concatenate all at once to avoid fragmentation
    new_df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return new_df