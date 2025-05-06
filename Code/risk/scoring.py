import numpy as np
import pandas as pd

class RiskScorer:
    """
    A class to calculate risk scores based on failure probability and time to failure.
    """
    
    def __init__(self, failure_stage=4, epsilon=1e-6):
        """
        Initialize the RiskScorer.
        
        Args:
            failure_stage: The stage that represents failure (default: 4)
            epsilon: Small value to avoid division by zero
        """
        self.failure_stage = failure_stage
        self.epsilon = epsilon
        # Default time horizon for risk calculation (cycles)
        self.time_horizon = 100
    
    def calculate_raw_risk_score(self, failure_probability, time_left):
        """
        Calculate raw risk score based on failure probability and time left.
        The closer to failure (lower time_left) and higher probability yield higher risk.
        
        Args:
            failure_probability: Probability of reaching failure stage
            time_left: Estimated time left until failure in cycles
            
        Returns:
            Raw risk score
        """
        # For time-based risk, lower time means higher risk, so we invert the relationship
        # We cap the time value to avoid extremely large risk values for very small times
        capped_time = max(time_left, 1.0)
        
        # Higher probability and lower time to failure = higher risk
        return failure_probability * (self.time_horizon / capped_time)
    
    def calculate_urgency_score(self, failure_probability, time_left):
        """
        Calculate urgency-based risk score as failure probability divided by time left.
        
        Args:
            failure_probability: Probability of reaching failure stage
            time_left: Estimated time left until failure in cycles
            
        Returns:
            Urgency-based risk score
        """
        # Cap minimum time to avoid extreme values and apply non-linear scaling
        capped_time = max(time_left, 1.0)
        
        # Higher probability and lower time means higher urgency
        return failure_probability / (capped_time + self.epsilon)
    
    def extract_failure_probability(self, current_stage, stages_total=4):
        """
        Estimate failure probability based on current stage.
        Improved model with non-linear scaling.
        
        Args:
            current_stage: The current degradation stage
            stages_total: Total number of stages
            
        Returns:
            Estimated probability of failure
        """
        # Convert to float to ensure proper division
        stage = float(current_stage)
        
        # Exponential scaling to make higher stages much more likely to fail
        # This provides a better distinction between stages
        if stage >= stages_total:
            return 1.0
        else:
            # Exponential increase in probability as stage increases
            return min(1.0, (stage / stages_total)**2 + 0.1 * stage)
    
    def calculate_risk_scores(self, df):
        """
        Calculate risk scores for a DataFrame of units.
        
        Args:
            df: DataFrame with predicted_stage and predicted_time_to_next_stage columns
            
        Returns:
            DataFrame with raw_risk_score and urgency_score columns added
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Extract failure probability from current stage
        result_df['failure_probability'] = result_df['predicted_stage'].apply(
            lambda x: self.extract_failure_probability(x)
        )
        
        # Calculate time to failure based on current stage and time to next stage
        result_df['time_to_failure'] = result_df.apply(
            lambda row: self._calculate_time_to_failure(row['predicted_stage'], row['predicted_time_to_next_stage']),
            axis=1
        )
        
        # Calculate raw risk score
        result_df['raw_risk_score'] = result_df.apply(
            lambda row: self.calculate_raw_risk_score(
                row['failure_probability'], 
                row['time_to_failure']
            ), axis=1
        )
        
        # Calculate urgency-based risk score
        result_df['urgency_score'] = result_df.apply(
            lambda row: self.calculate_urgency_score(
                row['failure_probability'], 
                row['time_to_failure']
            ), axis=1
        )
        
        # Ensure all scores are positive
        result_df['raw_risk_score'] = result_df['raw_risk_score'].apply(lambda x: max(0, x))
        result_df['urgency_score'] = result_df['urgency_score'].apply(lambda x: max(0, x))
        
        # Add normalized urgency score for better scale across different datasets
        min_urgency = result_df['urgency_score'].min()
        max_urgency = result_df['urgency_score'].max()
        if max_urgency > min_urgency:
            result_df['normalized_urgency'] = (result_df['urgency_score'] - min_urgency) / (max_urgency - min_urgency)
        else:
            result_df['normalized_urgency'] = 0.5  # Default value if all scores are the same
        
        return result_df
    
    def _calculate_time_to_failure(self, current_stage, time_to_next_stage):
        """
        Estimate total time to failure based on current stage and time to next stage.
        
        Args:
            current_stage: Current degradation stage
            time_to_next_stage: Predicted time to next stage
            
        Returns:
            Estimated time to failure in cycles
        """
        # Ensure we have valid inputs
        current_stage = int(float(current_stage))
        time_to_next = max(0, float(time_to_next_stage))
        
        # If already in failure stage
        if current_stage >= self.failure_stage:
            return 0
            
        # Calculate remaining stages until failure
        remaining_stages = self.failure_stage - current_stage
        
        # If just one stage remains, time to failure is just time to next stage
        if remaining_stages == 1:
            return time_to_next
            
        # For multiple remaining stages, we estimate time for each transition
        # Simple model: assume each future stage transition takes the same time as current one
        # Can be improved with historical data on average time per stage
        # Add 20% buffer to the estimated time for each additional stage
        return time_to_next * (1 + 0.8 * (remaining_stages - 1))