"""
Core calculations for heat adaptation analysis.

This module contains the fundamental calculations used throughout the heat
adaptation analysis, including Heat Strain Score (HSS) calculation, heat
index computation, and various normalization functions.
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from datetime import datetime, timedelta


class HeatCalculations:
    """Core calculations for heat adaptation analysis."""
    
    @staticmethod
    def normalize(value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to 0-1 range.
        
        Args:
            value: Value to normalize
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
            
        Returns:
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5  # Avoid division by zero
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def scale_distance(distance: float, min_dist: float = 1, max_dist: float = 15) -> float:
        """
        Scale distance for HSS calculation.
        
        Args:
            distance: Distance in miles
            min_dist: Minimum distance for scaling
            max_dist: Maximum distance for scaling
            
        Returns:
            Scaled distance factor
        """
        norm_dist = HeatCalculations.normalize(distance, min_dist, max_dist)
        return 0.5 + (norm_dist * 0.5)
    
    @staticmethod
    def pace_to_seconds(pace_str: str) -> int:
        """
        Convert pace string (MM:SS) to seconds per mile.
        
        Args:
            pace_str: Pace in MM:SS format (e.g., "7:30")
            
        Returns:
            Pace in seconds per mile
            
        Raises:
            ValueError: If pace string format is invalid
        """
        try:
            if ':' not in pace_str:
                raise ValueError("Pace must be in MM:SS format")
            
            minutes, seconds = map(int, pace_str.split(":"))
            if minutes < 0 or seconds < 0 or seconds >= 60:
                raise ValueError("Invalid time values")
            
            return minutes * 60 + seconds
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid pace format '{pace_str}': {e}")
    
    @staticmethod
    def seconds_to_pace(seconds: int) -> str:
        """
        Convert seconds per mile to pace string (MM:SS).
        
        Args:
            seconds: Seconds per mile
            
        Returns:
            Pace string in MM:SS format
        """
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"
    
    @staticmethod
    def calculate_heat_index(temp_f: Union[float, pd.Series], 
                           humidity_pct: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """
        Calculate heat index (feels like temperature).
        
        Uses the National Weather Service heat index formula.
        
        Args:
            temp_f: Temperature in Fahrenheit
            humidity_pct: Humidity percentage
            
        Returns:
            Heat index in Fahrenheit
        """
        T = temp_f
        H = humidity_pct
        
        # Handle both single values and pandas Series
        if hasattr(T, '__iter__') and not isinstance(T, str):
            # For pandas Series/arrays - use vectorized operations
            heat_index = np.where(
                T < 80,
                T,  # If temp < 80, just return temperature
                (  # Otherwise calculate heat index
                    -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
                    - 6.83783e-3*T**2 - 5.481717e-2*H**2 + 1.22874e-3*T**2*H
                    + 8.5282e-4*T*H**2 - 1.99e-6*T**2*H**2
                )
            )
        else:
            # For single values - use simple if statement
            if T < 80:
                heat_index = T
            else:
                heat_index = (
                    -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
                    - 6.83783e-3*T**2 - 5.481717e-2*H**2 + 1.22874e-3*T**2*H
                    + 8.5282e-4*T*H**2 - 1.99e-6*T**2*H**2
                )
        
        return heat_index
    
    @staticmethod
    def heat_strain_score(temp: float, humidity: float, pace_sec_per_mile: float, 
                         avg_hr: float, max_hr: float, distance: float, 
                         multiplier: float = 1.0) -> float:
        """
        Calculate Heat Strain Score (HSS).
        
        HSS is a comprehensive metric that combines environmental factors
        (temperature, humidity) with performance factors (pace, heart rate, distance).
        
        Args:
            temp: Temperature in Fahrenheit
            humidity: Humidity percentage
            pace_sec_per_mile: Pace in seconds per mile
            avg_hr: Average heart rate during run
            max_hr: Maximum heart rate
            distance: Distance in miles
            multiplier: Adjustment multiplier for score
            
        Returns:
            Heat Strain Score
        """
        # Normalize environmental factors
        T_norm = HeatCalculations.normalize(temp, 20, 110)
        H_norm = HeatCalculations.normalize(humidity, 20, 99)
        
        # Scale distance
        dist_scaled = HeatCalculations.scale_distance(distance)
        
        # Calculate composite score
        score = multiplier * (
            ((2 * T_norm) + (1.2 * (H_norm ** 1.3)) + (4.2 * (T_norm * H_norm))) *
            ((avg_hr / max_hr) ** 1.1) *
            ((600 / pace_sec_per_mile) ** 1.3) *
            (dist_scaled ** 0.4)
        )
        
        return score
    
    @staticmethod
    def estimate_threshold(scores: List[float]) -> float:
        """
        Estimate threshold score for heat strain classification.
        
        Args:
            scores: List of heat strain scores
            
        Returns:
            Estimated threshold (median of scores)
        """
        if not scores:
            return 10.0  # Default threshold
        return float(np.median(scores))
    
    @staticmethod
    def adjust_multiplier(raw_score: float, threshold: float, 
                         pace_sec: float, mile_pr_sec: float) -> float:
        """
        Calculate adjustment multiplier for HSS based on performance factors.
        
        Args:
            raw_score: Raw heat strain score
            threshold: Threshold for heat strain classification
            pace_sec: Actual pace in seconds per mile
            mile_pr_sec: Mile PR pace in seconds per mile
            
        Returns:
            Adjustment multiplier
        """
        multiplier = 1.0
        
        # Adjust down if score is above threshold (already stressed)
        if raw_score > threshold:
            multiplier *= 0.9
        
        # Adjust up if running close to PR pace (extra effort)
        if pace_sec <= mile_pr_sec * 1.1:
            multiplier *= 1.1
        
        return multiplier
    
    @staticmethod
    def detect_outliers(scores: List[float], method: str = 'iqr', 
                       threshold: float = 1.5) -> List[bool]:
        """
        Detect outliers in heat strain scores.
        
        Args:
            scores: List of scores to analyze
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Boolean list indicating outliers
        """
        if not scores or len(scores) < 3:
            return [False] * len(scores)
        
        scores_array = np.array(scores)
        
        if method == 'iqr':
            Q1 = np.percentile(scores_array, 25)
            Q3 = np.percentile(scores_array, 75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # All values are the same
                return [False] * len(scores)
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (scores_array < lower_bound) | (scores_array > upper_bound)
        
        elif method == 'zscore':
            mean_score = np.mean(scores_array)
            std_score = np.std(scores_array)
            
            if std_score == 0:  # All values are the same
                return [False] * len(scores)
            
            z_scores = np.abs((scores_array - mean_score) / std_score)
            outliers = z_scores > threshold
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return outliers.tolist()
    
    @staticmethod
    def apply_physiological_limits(improvement_pct: float, 
                                 max_improvement: float = 25.0) -> Tuple[float, str]:
        """
        Apply physiological limits to heat adaptation improvement predictions.
        
        Args:
            improvement_pct: Raw predicted improvement percentage
            max_improvement: Maximum physiologically realistic improvement
            
        Returns:
            Tuple of (limited_improvement, interpretation_message)
        """
        if improvement_pct is None:
            return None, "No prediction available"
        
        # Cap improvement at physiologically realistic levels
        limited_improvement = min(abs(improvement_pct), max_improvement)
        
        # Generate interpretation message
        if limited_improvement > 20:
            message = f"Raw model predicted {abs(improvement_pct):.1f}% improvement, capped at {limited_improvement:.1f}% (physiological maximum)"
        elif limited_improvement > 15:
            message = f"{limited_improvement:.1f}% improvement suggests you're relatively heat-naive"
        elif limited_improvement > 10:
            message = f"{limited_improvement:.1f}% improvement suggests moderate heat adaptation potential"
        else:
            message = f"{limited_improvement:.1f}% improvement suggests you're already well heat-adapted"
        
        return limited_improvement, message