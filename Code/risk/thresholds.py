import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

class RiskThresholdManager:
    """
    Manages risk thresholds and risk score calculations for the predictive maintenance system.
    """
    def __init__(self, 
                 high_risk_threshold=0.6, 
                 medium_risk_threshold=0.3,
                 low_risk_threshold=0.1):
        """
        Initialize the Risk Threshold Manager with default thresholds.
        
        Args:
            high_risk_threshold: Threshold for high risk alerts
            medium_risk_threshold: Threshold for medium risk alerts
            low_risk_threshold: Threshold for low risk alerts
        """
        self.high_risk_threshold = high_risk_threshold
        self.medium_risk_threshold = medium_risk_threshold
        self.low_risk_threshold = low_risk_threshold
        
    def calculate_raw_risk_score(self, failure_probability, time_left):
        """
        Calculate raw risk score as the product of failure probability and time left.
        
        Args:
            failure_probability: Probability of failure (0-1)
            time_left: Estimated time left until failure (in cycles)
            
        Returns:
            Raw risk score
        """
        return failure_probability * time_left
    
    def calculate_urgency_based_risk(self, failure_probability, time_left, epsilon=1e-6):
        """
        Calculate urgency-based risk score as the ratio of failure probability to time left.
        
        Args:
            failure_probability: Probability of failure (0-1)
            time_left: Estimated time left until failure (in cycles)
            epsilon: Small constant to avoid division by zero
            
        Returns:
            Urgency-based risk score
        """
        # Cap minimum time value to avoid division by extremely small numbers
        capped_time = max(time_left, 1.0)
        return failure_probability / (capped_time + epsilon)
    
    def min_max_normalize(self, scores, min_score=None, max_score=None):
        """
        Normalize scores using min-max scaling.
        
        Args:
            scores: Array of risk scores
            min_score: Minimum score for scaling (if None, use minimum of scores)
            max_score: Maximum score for scaling (if None, use maximum of scores)
            
        Returns:
            Normalized scores
        """
        if min_score is None:
            min_score = np.min(scores)
        if max_score is None:
            max_score = np.max(scores)
            
        # Avoid division by zero
        if max_score == min_score:
            return np.ones_like(scores) * 0.5  # Return middle value instead of zeros
            
        return (scores - min_score) / (max_score - min_score)
    
    def get_risk_level(self, normalized_score):
        """
        Determine risk level based on normalized score.
        
        Args:
            normalized_score: Normalized risk score (0-1)
            
        Returns:
            Risk level as string ('HIGH', 'MEDIUM', 'LOW', 'NEGLIGIBLE')
        """
        # Ensure score is positive
        score = max(0, float(normalized_score))
        
        if score >= self.high_risk_threshold:
            return 'HIGH'
        elif score >= self.medium_risk_threshold:
            return 'MEDIUM'
        elif score >= self.low_risk_threshold:
            return 'LOW'
        else:
            return 'NEGLIGIBLE'
            
    def optimize_threshold(self, true_failures, risk_scores, beta=1.0):
        """
        Optimize threshold using precision-recall curve.
        
        Args:
            true_failures: Binary array indicating actual failures
            risk_scores: Array of risk scores
            beta: Weight of recall vs precision (beta>1 favors recall)
            
        Returns:
            Optimal threshold
        """
        precision, recall, thresholds = precision_recall_curve(true_failures, risk_scores)
        
        # Calculate F-beta score
        f_scores = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall + 1e-10)
        
        # Find the threshold that maximizes F-beta score
        optimal_idx = np.argmax(f_scores)
        if optimal_idx < len(thresholds):
            return thresholds[optimal_idx]
        else:
            # Edge case: the best threshold might correspond to the last point in the PR curve
            return np.max(risk_scores) + 0.01  # Slightly higher than all scores
    
    def update_thresholds(self, true_failures, risk_scores, beta=1.0):
        """
        Update risk thresholds based on observed data.
        
        Args:
            true_failures: Binary array indicating actual failures
            risk_scores: Array of risk scores
            beta: Weight of recall vs precision
        """
        optimal = self.optimize_threshold(true_failures, risk_scores, beta)
        
        # Set high risk threshold to the optimal threshold
        self.high_risk_threshold = optimal
        
        # Set medium and low risk thresholds proportionally
        self.medium_risk_threshold = max(0.1, optimal * 0.6)
        self.low_risk_threshold = max(0.05, optimal * 0.3)
        
        return {
            'high_risk': self.high_risk_threshold,
            'medium_risk': self.medium_risk_threshold,
            'low_risk': self.low_risk_threshold
        }
        
    def get_dynamic_thresholds(self, scores):
        """
        Compute dynamic thresholds based on the distribution of scores.
        This helps adapt thresholds to different datasets.
        
        Args:
            scores: Array of risk scores
            
        Returns:
            Dictionary with high, medium and low thresholds
        """
        if len(scores) == 0:
            return {
                'high_risk': self.high_risk_threshold,
                'medium_risk': self.medium_risk_threshold,
                'low_risk': self.low_risk_threshold
            }
            
        # Calculate percentile-based thresholds
        # High risk: top 15% of scores
        # Medium risk: top 40% of scores
        # Low risk: top 70% of scores
        high = np.percentile(scores, 85)
        medium = np.percentile(scores, 60)
        low = np.percentile(scores, 30)
        
        # Only update if thresholds make sense (in increasing order)
        if high > medium > low:
            self.high_risk_threshold = high
            self.medium_risk_threshold = medium
            self.low_risk_threshold = low
            
        return {
            'high_risk': self.high_risk_threshold,
            'medium_risk': self.medium_risk_threshold,
            'low_risk': self.low_risk_threshold
        }