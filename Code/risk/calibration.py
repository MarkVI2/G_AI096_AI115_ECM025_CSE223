import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

class RiskCalibrator:
    """
    Calibrates risk scores using both classifier probabilities 
    and regression predictions.
    """
    
    def __init__(self, method='minmax', failure_stage=4):
        """
        Initialize the Risk Calibrator.
        
        Args:
            method: Method to normalize risk scores ('minmax' or 'urgency')
            failure_stage: The stage considered as failure (default: 4)
        """
        self.method = method
        self.failure_stage = failure_stage
        self.scaler = MinMaxScaler()
        self.max_time_cycles = None  # Will be determined from data
        
    def calculate_stage_probability(self, df, stage_col='predicted_stage'):
        """
        Extract the probability of being in the failure stage.
        
        In this implementation, we use a simple heuristic based on current stage.
        For a more sophisticated approach, actual probabilities from the classifier would be used.
        
        Args:
            df: DataFrame containing stage predictions
            stage_col: Column name for stage predictions
            
        Returns:
            Series of probabilities for the failure stage
        """
        # Simple heuristic: higher stages have higher failure probability
        current_stages = df[stage_col].values
        
        # Calculate a probability based on proximity to failure stage
        # Stage 0: 0.1, Stage 1: 0.3, Stage 2: 0.5, Stage 3: 0.8, Stage 4: 1.0 (example values)
        stage_probs = {
            0: 0.1,
            1: 0.3, 
            2: 0.5,
            3: 0.8,
            4: 1.0
        }
        
        return pd.Series([stage_probs.get(int(stage), 0.0) for stage in current_stages], 
                         index=df.index)
    
    def calculate_time_to_failure(self, df, time_col='predicted_time_to_next_stage', stage_col='predicted_stage'):
        """
        Calculate estimated time to failure based on regression predictions.
        
        Args:
            df: DataFrame with time predictions
            time_col: Column name for time predictions
            stage_col: Column name for stage predictions
            
        Returns:
            Series of estimated time to failure
        """
        times_to_next_stage = df[time_col].values
        current_stages = df[stage_col].values
        
        # Calculate time to failure based on current stage and time to next stage
        time_to_failure = []
        for i, stage in enumerate(current_stages):
            stage = int(stage)
            time_to_next = max(0, times_to_next_stage[i])
            
            # If already in failure stage
            if stage >= self.failure_stage:
                time_to_failure.append(0)
            else:
                # Simple estimation: time to next stage + (failure_stage - (stage + 1)) * average_time_per_stage
                # For sophistication, this could use the actual predicted times for each future stage transition
                average_time_per_stage = 30  # Example assumption, should be derived from data
                stages_remaining = self.failure_stage - (stage + 1)
                time_to_failure.append(time_to_next + stages_remaining * average_time_per_stage)
                
        return pd.Series(time_to_failure, index=df.index)
    
    def calculate_raw_risk_score(self, failure_probability, time_to_failure):
        """
        Calculate raw risk score as product of failure probability and time to failure.
        
        Args:
            failure_probability: Probability of failure
            time_to_failure: Time to failure
            
        Returns:
            Raw risk score
        """
        if self.method == 'minmax':
            # Higher probability and lower time to failure means higher risk
            return failure_probability * (self.max_time_cycles - time_to_failure)
        elif self.method == 'urgency':
            # Urgency-based inversion: probability / time
            return failure_probability / (time_to_failure + 1e-6)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def normalize_risk_score(self, raw_scores):
        """
        Normalize risk scores to [0, 1] range.
        
        Args:
            raw_scores: Raw risk scores
            
        Returns:
            Normalized risk scores
        """
        if self.method == 'minmax':
            # Already using scaler initialized in fit
            return self.scaler.transform(raw_scores.reshape(-1, 1)).flatten()
        elif self.method == 'urgency':
            # Urgency scores are already relative, but clip to [0,1] for safety
            return np.clip(raw_scores, 0, 1)
        
    def fit(self, df):
        """
        Fit the risk calibrator on data.
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            Self
        """
        # Determine maximum time cycles from data for normalization purposes
        time_cycles = df['time_cycles'].max()
        self.max_time_cycles = max(100, time_cycles * 2)  # Set a reasonable maximum
        
        # If using minmax method, fit the scaler on example data
        if self.method == 'minmax':
            # Generate example raw scores across the expected range
            failure_probs = np.linspace(0.1, 1.0, 10)
            time_estimates = np.linspace(0, self.max_time_cycles, 10)
            
            example_scores = []
            for prob in failure_probs:
                for time_est in time_estimates:
                    example_scores.append(self.calculate_raw_risk_score(prob, time_est))
                    
            self.scaler.fit(np.array(example_scores).reshape(-1, 1))
            
        return self
    
    def transform(self, df):
        """
        Calculate risk scores for the given data.
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            DataFrame with added risk scores
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate failure probability
        result_df['failure_probability'] = self.calculate_stage_probability(df)
        
        # Calculate time to failure
        result_df['time_to_failure'] = self.calculate_time_to_failure(df)
        
        # Calculate raw risk score
        raw_scores = []
        for i, row in result_df.iterrows():
            raw_score = self.calculate_raw_risk_score(
                row['failure_probability'], 
                row['time_to_failure']
            )
            raw_scores.append(raw_score)
            
        result_df['raw_risk_score'] = raw_scores
        
        # Normalize risk scores
        result_df['normalized_risk_score'] = self.normalize_risk_score(np.array(raw_scores))
        
        return result_df
    
    def fit_transform(self, df):
        """
        Fit and transform in one step.
        
        Args:
            df: DataFrame with prediction data
            
        Returns:
            DataFrame with added risk scores
        """
        return self.fit(df).transform(df)