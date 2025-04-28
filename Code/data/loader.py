import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union


class CMAPSSDataLoader:
    """
    Data loader for the NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.
    This class handles loading and preprocessing of the Turbofan Engine Degradation Simulation datasets.
    """
    
    # Column names based on the dataset description
    _column_names = [
        'unit_number', 'time_cycles',
        'setting_1', 'setting_2', 'setting_3'
    ] + [f'sensor_{i}' for i in range(1, 27)]
    
    # Dataset information
    _dataset_info = {
        'FD001': {'train_units': 100, 'test_units': 100, 'conditions': 'ONE (Sea Level)', 'fault_modes': 'ONE (HPC Degradation)'},
        'FD002': {'train_units': 260, 'test_units': 259, 'conditions': 'SIX', 'fault_modes': 'ONE (HPC Degradation)'},
        'FD003': {'train_units': 100, 'test_units': 100, 'conditions': 'ONE (Sea Level)', 'fault_modes': 'TWO (HPC Degradation, Fan Degradation)'},
        'FD004': {'train_units': 248, 'test_units': 249, 'conditions': 'SIX', 'fault_modes': 'TWO (HPC Degradation, Fan Degradation)'}
    }
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.train_data = {}
        self.test_data = {}
        self.rul_data = {}
    
    def load_dataset(self, dataset_id: str) -> Dict[str, pd.DataFrame]:
        """
        Load a specific CMAPSS dataset by ID (FD001, FD002, FD003, or FD004).
        
        Args:
            dataset_id: Dataset identifier (FD001, FD002, FD003, or FD004)
            
        Returns:
            Dictionary with train, test, and RUL dataframes
        """
        if dataset_id not in self._dataset_info:
            raise ValueError(f"Invalid dataset ID: {dataset_id}. Must be one of: FD001, FD002, FD003, FD004")
        
        # Load train data
        train_file = os.path.join(self.data_dir, 'train', f'train_{dataset_id}.txt')
        train_df = self._load_data_file(train_file)
        
        # Load test data
        test_file = os.path.join(self.data_dir, 'test', f'test_{dataset_id}.txt')
        test_df = self._load_data_file(test_file)
        
        # Load RUL data
        rul_file = os.path.join(self.data_dir, 'RUL', f'RUL_{dataset_id}.txt')
        rul_df = self._load_rul_file(rul_file)
        
        # Store data
        self.train_data[dataset_id] = train_df
        self.test_data[dataset_id] = test_df
        self.rul_data[dataset_id] = rul_df
        
        return {
            'train': train_df,
            'test': test_df,
            'rul': rul_df
        }
    
    def load_all_datasets(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load all CMAPSS datasets (FD001-FD004).
        
        Returns:
            Dictionary with dataset IDs as keys and dictionaries of train, test, and RUL dataframes as values
        """
        all_data = {}
        for dataset_id in self._dataset_info.keys():
            all_data[dataset_id] = self.load_dataset(dataset_id)
        return all_data
    
    def _load_data_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a CMAPSS data file (train or test) into a pandas DataFrame.
        """
        # Load the data with whitespace delimiter
        df = pd.read_csv(file_path, delimiter=r'\s+', header=None)
        # Dynamically assign column names to include all sensor measurements
        total_cols = df.shape[1]
        if total_cols < 6:
            raise ValueError(f"Unexpected number of columns: {total_cols}")
        num_sensors = total_cols - 5
        column_names = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1, num_sensors + 1)]
        df.columns = column_names
        return df
    
    def _load_rul_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a RUL (Remaining Useful Life) data file into a pandas DataFrame.
        
        Args:
            file_path: Path to the RUL file
            
        Returns:
            DataFrame with RUL values
        """
        # RUL files contain just one column of numbers
        rul_values = pd.read_csv(file_path, delimiter=r'\s+', header=None)
        rul_df = pd.DataFrame({'RUL': rul_values.iloc[:, 0]})
        # Add unit index (1-based, matching the convention in the dataset)
        rul_df['unit_number'] = range(1, len(rul_df) + 1)
        return rul_df
    
    def calculate_rul_for_training(self, dataset_id: str) -> pd.DataFrame:
        """
        Calculate RUL (Remaining Useful Life) for training data.
        For training data, we know when each engine fails (the last cycle in the time series).
        
        Args:
            dataset_id: Dataset identifier (FD001, FD002, FD003, or FD004)
            
        Returns:
            Training DataFrame with added RUL column
        """
        if dataset_id not in self.train_data:
            raise ValueError(f"Dataset {dataset_id} not loaded. Call load_dataset first.")
        
        train_df = self.train_data[dataset_id].copy()
        
        # Get maximum cycle for each unit (the failure point)
        max_cycles = train_df.groupby('unit_number')['time_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycles']
        
        # Merge with the training data
        train_df = train_df.merge(max_cycles, on='unit_number', how='left')
        
        # Calculate RUL: max_cycles - current_cycle
        train_df['RUL'] = train_df['max_cycles'] - train_df['time_cycles']
        
        # Drop the helper column
        train_df.drop('max_cycles', axis=1, inplace=True)
        
        return train_df
    
    def prepare_test_with_rul(self, dataset_id: str) -> pd.DataFrame:
        """
        Prepare test data by combining with RUL information.
        For test data, we add the actual RUL at the last cycle of each engine.
        
        Args:
            dataset_id: Dataset identifier (FD001, FD002, FD003, or FD004)
            
        Returns:
            Test DataFrame with added RUL information
        """
        if dataset_id not in self.test_data or dataset_id not in self.rul_data:
            raise ValueError(f"Dataset {dataset_id} not loaded. Call load_dataset first.")
        
        test_df = self.test_data[dataset_id].copy()
        rul_df = self.rul_data[dataset_id]
        
        # Get the last cycle for each unit in the test data
        last_cycles = test_df.groupby('unit_number')['time_cycles'].max().reset_index()
        last_cycles.columns = ['unit_number', 'last_cycle']
        
        # Add RUL information to the last cycles
        last_cycles_with_rul = last_cycles.merge(rul_df, on='unit_number', how='left')
        
        # Add a helper column to identify last cycles
        test_df = test_df.merge(
            last_cycles[['unit_number', 'last_cycle']], 
            on='unit_number', 
            how='left'
        )
        test_df['is_last_cycle'] = test_df['time_cycles'] == test_df['last_cycle']
        
        # Add RUL to the last cycles
        test_with_rul = test_df.merge(
            last_cycles_with_rul[['unit_number', 'last_cycle', 'RUL']], 
            on=['unit_number', 'last_cycle'], 
            how='left'
        )
        
        # For each engine, calculate RUL for all cycles based on the last cycle's RUL
        # RUL at cycle t = RUL at last cycle + (last_cycle - t)
        for unit in test_with_rul['unit_number'].unique():
            unit_data = test_with_rul[test_with_rul['unit_number'] == unit]
            last_rul = unit_data[unit_data['is_last_cycle']]['RUL'].values[0]
            last_cycle = unit_data['last_cycle'].values[0]
            
            # Calculate RUL for each cycle
            mask = test_with_rul['unit_number'] == unit
            test_with_rul.loc[mask, 'RUL'] = last_rul + (last_cycle - test_with_rul.loc[mask, 'time_cycles'])
        
        # Drop helper columns
        test_with_rul.drop(['last_cycle', 'is_last_cycle'], axis=1, inplace=True)
        
        return test_with_rul
    
    @classmethod
    def get_dataset_info(cls) -> Dict[str, Dict[str, Union[int, str]]]:
        """
        Get information about all datasets.
        
        Returns:
            Dictionary with information about each dataset
        """
        return cls._dataset_info
    
    def get_important_sensors(self) -> List[str]:
        """
        Return a list of all sensors with indication of which ones are commonly considered important for RUL prediction.
        The important sensors are based on domain knowledge and research papers on the CMAPSS dataset.
        
        Returns:
            List of all sensor names with 'important_' prefix for those considered significant by research
        """
        # All 21 sensors
        all_sensors = [f'sensor_{i}' for i in range(1, 22)]
        
        # Sensors often found to be most relevant in literature
        important_ones = [
            'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
            'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 
            'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
        ]
        
        # Add 'important_' prefix to sensors considered important
        return all_sensors
    
    def normalize_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      columns_to_normalize: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize the data using min-max normalization based on the training set.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            columns_to_normalize: List of columns to normalize. If None, all sensor columns are normalized.
            
        Returns:
            Tuple of normalized (train_df, test_df)
        """
        if columns_to_normalize is None:
            # Normalize all sensor columns by default
            columns_to_normalize = [col for col in train_df.columns if col.startswith('sensor_')]
        
        # Create copies to avoid modifying the original data
        train_normalized = train_df.copy()
        test_normalized = test_df.copy()
        
        # Perform min-max normalization
        for col in columns_to_normalize:
            if col in train_df.columns and col in test_df.columns:
                min_val = train_df[col].min()
                max_val = train_df[col].max()
                
                # Avoid division by zero
                if max_val > min_val:
                    train_normalized[col] = (train_df[col] - min_val) / (max_val - min_val)
                    test_normalized[col] = (test_df[col] - min_val) / (max_val - min_val)
        
        return train_normalized, test_normalized


def load_dataset(dataset_id: str, data_dir: str = None) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load a specific CMAPSS dataset by ID.
    
    Args:
        dataset_id: Dataset identifier (FD001, FD002, FD003, or FD004)
        data_dir: Path to the directory containing the CMAPSS dataset files.
                 If None, uses the current directory's parent.
    
    Returns:
        Dictionary with train, test, and RUL dataframes
    """
    if data_dir is None:
        # Default to the directory where this script is located
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    loader = CMAPSSDataLoader(data_dir)
    return loader.load_dataset(dataset_id)


def load_all_datasets(data_dir: str = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Convenience function to load all CMAPSS datasets.
    
    Args:
        data_dir: Path to the directory containing the CMAPSS dataset files.
                 If None, uses the current directory's parent.
    
    Returns:
        Dictionary with dataset IDs as keys and dictionaries of train, test, and RUL dataframes as values
    """
    if data_dir is None:
        # Default to the directory where this script is located
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    loader = CMAPSSDataLoader(data_dir)
    return loader.load_all_datasets()


if __name__ == "__main__":
    # Example usage
    data_dir = os.path.dirname(os.path.abspath(__file__))
    loader = CMAPSSDataLoader(data_dir)
    
    # Load a single dataset
    print("Loading dataset FD001...")
    data = loader.load_dataset("FD001")
    
    # Display information about the dataset
    train_df = data['train']
    test_df = data['test']
    rul_df = data['rul']
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"RUL shape: {rul_df.shape}")
    
    # Calculate RUL for training data
    train_with_rul = loader.calculate_rul_for_training("FD001")
    print(f"Train with RUL shape: {train_with_rul.shape}")
    
    # Prepare test data with RUL
    test_with_rul = loader.prepare_test_with_rul("FD001")
    print(f"Test with RUL shape: {test_with_rul.shape}")
    
    # Print sample of the data
    print("\nSample of training data:")
    print(train_df.head())
    
    print("\nSample of training data with RUL:")
    print(train_with_rul.head())
    
    print("\nSample of test data with RUL:")
    print(test_with_rul.head())