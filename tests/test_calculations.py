"""
Test suite for heat_adaptation.core.calculations module.

Tests cover:
- Heat score calculations
- Temperature and pace conversions
- Normalization functions
- Heat index calculations
- Physiological limits
- Outlier detection
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the functions we're testing
# These would be imported from the modularized heat_adaptation.core.calculations
# For now, we'll assume the structure based on the original code

def normalize(value, min_val, max_val):
    """Normalize value between 0 and 1"""
    return (value - min_val) / (max_val - min_val)

def scale_distance(distance, min_dist=1, max_dist=15):
    """Scale distance with min/max bounds"""
    norm_dist = normalize(distance, min_dist, max_dist)
    return 0.5 + (norm_dist * 0.5)

def pace_to_seconds(pace_str):
    """Convert MM:SS pace string to seconds per mile"""
    minutes, seconds = map(int, pace_str.split(":"))
    return minutes * 60 + seconds

def heat_score(temp, humidity, pace_sec_per_mile, avg_hr, max_hr, distance, multiplier=1.0):
    """Calculate heat strain score"""
    T_norm = normalize(temp, 20, 110)
    H_norm = normalize(humidity, 20, 99)
    dist_scaled = scale_distance(distance)

    score = multiplier * (
        ((2 * T_norm) + (1.2 * (H_norm ** 1.3)) + (4.2 * (T_norm * H_norm))) *
        ((avg_hr / max_hr) ** 1.1) *
        ((600 / pace_sec_per_mile) ** 1.3) *
        (dist_scaled ** 0.4)
    )
    return score

def calculate_heat_index(temp_f, humidity_pct):
    """Calculate heat index (feels like temperature)"""
    T = temp_f
    H = humidity_pct
    
    if hasattr(T, '__iter__') and not isinstance(T, str):
        heat_index = np.where(
            T < 80,
            T,
            (
                -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
                - 6.83783e-3*T**2 - 5.481717e-2*H**2 + 1.22874e-3*T**2*H
                + 8.5282e-4*T*H**2 - 1.99e-6*T**2*H**2
            )
        )
    else:
        if T < 80:
            heat_index = T
        else:
            heat_index = (
                -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
                - 6.83783e-3*T**2 - 5.481717e-2*H**2 + 1.22874e-3*T**2*H
                + 8.5282e-4*T*H**2 - 1.99e-6*T**2*H**2
            )
    return heat_index

def apply_physiological_limits(improvement_pct, max_improvement=25.0):
    """Apply realistic physiological limits to heat adaptation improvement"""
    if improvement_pct is None:
        return None
    
    limited_improvement = min(abs(improvement_pct), max_improvement)
    return limited_improvement

def detect_outliers(scores, method='iqr', threshold=1.5):
    """Detect outliers using IQR or Z-score method"""
    scores_array = np.array(scores)
    
    if method == 'iqr':
        Q1 = np.percentile(scores_array, 25)
        Q3 = np.percentile(scores_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (scores_array < lower_bound) | (scores_array > upper_bound)
    
    elif method == 'zscore':
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        z_scores = np.abs((scores_array - mean_score) / std_score)
        outliers = z_scores > threshold
    
    return outliers


class TestNormalization:
    """Test normalization and scaling functions"""
    
    def test_normalize_basic(self):
        """Test basic normalization"""
        assert normalize(50, 0, 100) == 0.5
        assert normalize(0, 0, 100) == 0.0
        assert normalize(100, 0, 100) == 1.0
        
    def test_normalize_edge_cases(self):
        """Test normalization edge cases"""
        # Same min and max should handle gracefully
        with pytest.raises(ZeroDivisionError):
            normalize(5, 10, 10)
            
    def test_scale_distance(self):
        """Test distance scaling function"""
        # Test normal range
        result = scale_distance(8, min_dist=1, max_dist=15)
        assert 0.5 <= result <= 1.0
        
        # Test boundaries
        min_result = scale_distance(1, min_dist=1, max_dist=15)
        max_result = scale_distance(15, min_dist=1, max_dist=15)
        assert min_result == 0.5
        assert max_result == 1.0


class TestPaceConversion:
    """Test pace conversion functions"""
    
    def test_pace_to_seconds_valid(self):
        """Test valid pace string conversion"""
        assert pace_to_seconds("7:30") == 450  # 7*60 + 30
        assert pace_to_seconds("10:00") == 600
        assert pace_to_seconds("6:45") == 405
        
    def test_pace_to_seconds_edge_cases(self):
        """Test edge cases for pace conversion"""
        assert pace_to_seconds("0:30") == 30
        assert pace_to_seconds("12:59") == 779
        
    def test_pace_to_seconds_invalid(self):
        """Test invalid pace strings"""
        with pytest.raises(ValueError):
            pace_to_seconds("invalid")
        with pytest.raises(ValueError):
            pace_to_seconds("7.30")  # Wrong separator


class TestHeatScore:
    """Test heat score calculation"""
    
    @pytest.fixture
    def sample_run_data(self):
        """Sample run data for testing"""
        return {
            'temp': 85,
            'humidity': 75,
            'pace_sec': 450,  # 7:30 pace
            'avg_hr': 165,
            'max_hr': 190,
            'distance': 5.0
        }
    
    def test_heat_score_calculation(self, sample_run_data):
        """Test basic heat score calculation"""
        score = heat_score(
            sample_run_data['temp'],
            sample_run_data['humidity'],
            sample_run_data['pace_sec'],
            sample_run_data['avg_hr'],
            sample_run_data['max_hr'],
            sample_run_data['distance']
        )
        
        assert isinstance(score, (int, float))
        assert score > 0  # Score should be positive
        
    def test_heat_score_multiplier(self, sample_run_data):
        """Test heat score with different multipliers"""
        base_score = heat_score(
            sample_run_data['temp'],
            sample_run_data['humidity'],
            sample_run_data['pace_sec'],
            sample_run_data['avg_hr'],
            sample_run_data['max_hr'],
            sample_run_data['distance'],
            multiplier=1.0
        )
        
        doubled_score = heat_score(
            sample_run_data['temp'],
            sample_run_data['humidity'],
            sample_run_data['pace_sec'],
            sample_run_data['avg_hr'],
            sample_run_data['max_hr'],
            sample_run_data['distance'],
            multiplier=2.0
        )
        
        assert abs(doubled_score - 2 * base_score) < 0.001  # Account for floating point
        
    def test_heat_score_temperature_sensitivity(self, sample_run_data):
        """Test that heat score increases with temperature"""
        low_temp_score = heat_score(
            70, sample_run_data['humidity'], sample_run_data['pace_sec'],
            sample_run_data['avg_hr'], sample_run_data['max_hr'], sample_run_data['distance']
        )
        
        high_temp_score = heat_score(
            95, sample_run_data['humidity'], sample_run_data['pace_sec'],
            sample_run_data['avg_hr'], sample_run_data['max_hr'], sample_run_data['distance']
        )
        
        assert high_temp_score > low_temp_score
        
    def test_heat_score_humidity_sensitivity(self, sample_run_data):
        """Test that heat score increases with humidity"""
        low_humidity_score = heat_score(
            sample_run_data['temp'], 40, sample_run_data['pace_sec'],
            sample_run_data['avg_hr'], sample_run_data['max_hr'], sample_run_data['distance']
        )
        
        high_humidity_score = heat_score(
            sample_run_data['temp'], 90, sample_run_data['pace_sec'],
            sample_run_data['avg_hr'], sample_run_data['max_hr'], sample_run_data['distance']
        )
        
        assert high_humidity_score > low_humidity_score


class TestHeatIndex:
    """Test heat index calculations"""
    
    def test_heat_index_low_temp(self):
        """Test heat index for temperatures below 80°F"""
        temp = 75
        humidity = 50
        heat_index = calculate_heat_index(temp, humidity)
        assert heat_index == temp  # Should return temperature if < 80
        
    def test_heat_index_high_temp(self):
        """Test heat index for temperatures above 80°F"""
        temp = 90
        humidity = 70
        heat_index = calculate_heat_index(temp, humidity)
        assert heat_index > temp  # Should be higher than actual temp
        
    def test_heat_index_vectorized(self):
        """Test heat index with pandas Series (vectorized)"""
        temps = pd.Series([75, 85, 95])
        humidity = pd.Series([50, 60, 70])
        
        heat_indices = calculate_heat_index(temps, humidity)
        
        assert len(heat_indices) == 3
        assert heat_indices.iloc[0] == 75  # Below 80, should equal temp
        assert heat_indices.iloc[1] > 85   # Above 80, should be higher
        assert heat_indices.iloc[2] > 95   # Above 80, should be higher
        
    def test_heat_index_extreme_conditions(self):
        """Test heat index under extreme conditions"""
        # Very hot and humid
        extreme_heat_index = calculate_heat_index(100, 90)
        moderate_heat_index = calculate_heat_index(85, 60)
        
        assert extreme_heat_index > moderate_heat_index


class TestPhysiologicalLimits:
    """Test physiological limits application"""
    
    def test_apply_physiological_limits_normal(self):
        """Test normal improvement values"""
        assert apply_physiological_limits(15.0) == 15.0
        assert apply_physiological_limits(10.0) == 10.0
        
    def test_apply_physiological_limits_capped(self):
        """Test that extreme values are capped"""
        assert apply_physiological_limits(30.0) == 25.0  # Default max
        assert apply_physiological_limits(40.0, max_improvement=20.0) == 20.0
        
    def test_apply_physiological_limits_none(self):
        """Test None input"""
        assert apply_physiological_limits(None) is None
        
    def test_apply_physiological_limits_negative(self):
        """Test negative improvement values"""
        assert apply_physiological_limits(-15.0) == 15.0  # Should take absolute value


class TestOutlierDetection:
    """Test outlier detection functions"""
    
    @pytest.fixture
    def sample_scores_with_outliers(self):
        """Sample scores with known outliers"""
        return [10, 12, 11, 13, 12, 10, 11, 25, 9, 12, 11]  # 25 is clear outlier
    
    @pytest.fixture
    def sample_scores_no_outliers(self):
        """Sample scores without outliers"""
        return [10, 12, 11, 13, 12, 10, 11, 14, 9, 12, 11]
    
    def test_detect_outliers_iqr(self, sample_scores_with_outliers):
        """Test IQR-based outlier detection"""
        outliers = detect_outliers(sample_scores_with_outliers, method='iqr', threshold=1.5)
        
        assert len(outliers) == len(sample_scores_with_outliers)
        assert outliers[7] == True  # The value 25 should be detected as outlier
        assert sum(outliers) >= 1  # At least one outlier should be detected
        
    def test_detect_outliers_zscore(self, sample_scores_with_outliers):
        """Test Z-score based outlier detection"""
        outliers = detect_outliers(sample_scores_with_outliers, method='zscore', threshold=2.0)
        
        assert len(outliers) == len(sample_scores_with_outliers)
        assert outliers[7] == True  # The value 25 should be detected as outlier
        
    def test_detect_outliers_no_outliers(self, sample_scores_no_outliers):
        """Test outlier detection when there are no outliers"""
        outliers = detect_outliers(sample_scores_no_outliers, method='iqr', threshold=1.5)
        
        # Should detect few or no outliers in normal distribution
        assert sum(outliers) <= 2  # Allow for some false positives in edge cases
        
    def test_detect_outliers_single_value(self):
        """Test outlier detection with single value"""
        outliers = detect_outliers([10], method='iqr')
        assert len(outliers) == 1
        assert outliers[0] == False  # Single value can't be an outlier
        
    def test_detect_outliers_empty_array(self):
        """Test outlier detection with empty array"""
        outliers = detect_outliers([], method='iqr')
        assert len(outliers) == 0


class TestIntegration:
    """Integration tests for multiple functions working together"""
    
    def test_full_calculation_pipeline(self):
        """Test complete calculation pipeline"""
        # Sample run data
        run_data = [
            {'temp': 85, 'humidity': 75, 'pace_str': '7:30', 'avg_hr': 165, 'max_hr': 190, 'distance': 5.0},
            {'temp': 88, 'humidity': 80, 'pace_str': '7:45', 'avg_hr': 170, 'max_hr': 190, 'distance': 3.1},
            {'temp': 82, 'humidity': 70, 'pace_str': '7:20', 'avg_hr': 160, 'max_hr': 190, 'distance': 6.0},
        ]
        
        # Convert paces to seconds and calculate heat scores
        heat_scores = []
        for run in run_data:
            pace_sec = pace_to_seconds(run['pace_str'])
            score = heat_score(
                run['temp'], run['humidity'], pace_sec, 
                run['avg_hr'], run['max_hr'], run['distance']
            )
            heat_scores.append(score)
            
        # Test that we get reasonable scores
        assert all(score > 0 for score in heat_scores)
        assert len(heat_scores) == 3
        
        # Test outlier detection on scores
        outliers = detect_outliers(heat_scores)
        assert len(outliers) == 3
        
    def test_heat_index_integration(self):
        """Test heat index calculation with various conditions"""
        conditions = [
            (75, 50),   # Cool and dry
            (85, 75),   # Warm and humid  
            (95, 85),   # Hot and very humid
        ]
        
        heat_indices = [calculate_heat_index(temp, hum) for temp, hum in conditions]
        
        # Heat indices should generally increase with conditions
        assert heat_indices[0] <= heat_indices[1] <= heat_indices[2]
        
    def test_physiological_limits_integration(self):
        """Test physiological limits with various improvement scenarios"""
        improvements = [5.0, 15.0, 25.0, 35.0, -10.0]
        limited_improvements = [apply_physiological_limits(imp) for imp in improvements]
        
        # All should be positive and within limits
        assert all(0 <= imp <= 25.0 for imp in limited_improvements if imp is not None)
        assert limited_improvements[3] == 25.0  # 35% should be capped at 25%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
