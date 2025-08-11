"""
Test suite for heat_adaptation_analysis package.

This package contains comprehensive tests for all modules in the heat adaptation analysis system:

Test Modules:
- test_calculations.py: Tests for core calculation functions (heat scores, normalization, etc.)
- test_ml_models.py: Tests for machine learning models and feature engineering
- test_data_manager.py: Tests for data loading, CSV processing, and validation

Test Categories:
- Unit tests: Test individual functions and methods
- Integration tests: Test component interactions
- Edge case tests: Test error handling and boundary conditions
- Performance tests: Test with realistic data scenarios

Usage:
    # Run all tests
    python -m pytest tests/

    # Run specific test file
    python -m pytest tests/test_calculations.py

    # Run with verbose output
    python -m pytest tests/ -v

    # Run with coverage
    python -m pytest tests/ --cov=heat_adaptation

Test Data:
    Tests use both synthetic data and realistic scenarios to ensure
    robust functionality across various use cases.

Requirements:
    - pytest
    - pytest-cov (for coverage reports)
    - All dependencies from requirements.txt
"""

import sys
import os
import warnings

# Add the parent directory to the path so we can import the heat_adaptation package
# This is useful when running tests from the tests directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress common warnings during testing
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Test configuration
import pytest

def pytest_configure(config):
    """Configure pytest settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "ml: marks tests that require machine learning libraries"
    )

# Common test fixtures and utilities
@pytest.fixture(scope="session")
def sample_run_data():
    """Sample run data used across multiple test files"""
    from datetime import datetime
    
    return [
        {
            'date': datetime(2024, 7, 1),
            'temp': 85,
            'humidity': 75,
            'pace_sec': 450,  # 7:30
            'avg_hr': 165,
            'max_hr': 190,
            'distance': 3.1,
            'raw_score': 12.5,
            'mile_pr_sec': 400  # 6:40 PR
        },
        {
            'date': datetime(2024, 7, 3),
            'temp': 88,
            'humidity': 80,
            'pace_sec': 465,  # 7:45
            'avg_hr': 170,
            'max_hr': 190,
            'distance': 5.0,
            'raw_score': 15.2,
            'mile_pr_sec': 400
        },
        {
            'date': datetime(2024, 7, 5),
            'temp': 92,
            'humidity': 85,
            'pace_sec': 480,  # 8:00
            'avg_hr': 175,
            'max_hr': 190,
            'distance': 3.1,
            'raw_score': 18.1,
            'mile_pr_sec': 400
        }
    ]

@pytest.fixture(scope="session")
def sample_environmental_conditions():
    """Sample environmental conditions for testing"""
    return [
        {'temp': 75, 'humidity': 60, 'description': 'cool_dry'},
        {'temp': 85, 'humidity': 75, 'description': 'warm_humid'},
        {'temp': 95, 'humidity': 85, 'description': 'hot_very_humid'},
        {'temp': 100, 'humidity': 90, 'description': 'extreme'},
        {'temp': 68, 'humidity': 40, 'description': 'cool_dry'}
    ]

# Test utilities
class TestUtils:
    """Utility class for common test operations"""
    
    @staticmethod
    def create_temp_csv(content, suffix='.csv'):
        """Create a temporary CSV file with given content"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    @staticmethod
    def cleanup_temp_file(filepath):
        """Clean up temporary file"""
        import os
        try:
            os.unlink(filepath)
        except (OSError, FileNotFoundError):
            pass  # File might already be deleted
    
    @staticmethod
    def assert_heat_score_reasonable(score, min_score=0, max_score=50):
        """Assert that a heat score is within reasonable bounds"""
        assert isinstance(score, (int, float)), f"Score should be numeric, got {type(score)}"
        assert min_score <= score <= max_score, f"Score {score} should be between {min_score} and {max_score}"
    
    @staticmethod
    def assert_dataframe_structure(df, expected_columns, min_rows=1):
        """Assert basic dataframe structure"""
        import pandas as pd
        
        assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
        assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"
        
        for col in expected_columns:
            assert col in df.columns, f"Expected column '{col}' not found in DataFrame"
    
    @staticmethod
    def generate_realistic_run_data(n_runs=10, base_temp=85, temp_variation=10, 
                                  base_humidity=75, humidity_variation=15,
                                  start_date='2024-06-01'):
        """Generate realistic run data for testing"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        np.random.seed(42)  # For reproducible tests
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        dates = [start + timedelta(days=i*2) for i in range(n_runs)]  # Every other day
        
        temps = np.random.normal(base_temp, temp_variation/3, n_runs)
        temps = np.clip(temps, base_temp - temp_variation, base_temp + temp_variation)
        
        humidities = np.random.normal(base_humidity, humidity_variation/3, n_runs)
        humidities = np.clip(humidities, base_humidity - humidity_variation, base_humidity + humidity_variation)
        
        # Pace gets slightly faster over time (adaptation effect)
        base_pace_sec = 450  # 7:30
        pace_improvement = np.linspace(0, -30, n_runs)  # Up to 30 seconds improvement
        paces = base_pace_sec + pace_improvement + np.random.normal(0, 15, n_runs)
        paces = np.clip(paces, 360, 600)  # 6:00 to 10:00 pace range
        
        # Heart rate decreases slightly with adaptation
        base_hr = 170
        hr_improvement = np.linspace(0, -10, n_runs)
        hrs = base_hr + hr_improvement + np.random.normal(0, 5, n_runs)
        hrs = np.clip(hrs, 140, 190)
        
        distances = np.random.choice([3.1, 5.0, 6.0, 4.0, 3.0], n_runs)
        
        return pd.DataFrame({
            'date': dates,
            'temp': temps,
            'humidity': humidities,
            'pace_sec': paces.astype(int),
            'avg_hr': hrs.astype(int),
            'max_hr': [190] * n_runs,
            'distance': distances,
            'mile_pr_sec': [400] * n_runs  # 6:40 PR
        })

# Performance testing utilities
class PerformanceUtils:
    """Utilities for performance testing"""
    
    @staticmethod
    def time_function(func, *args, **kwargs):
        """Time function execution"""
        import time
        
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        
        return result, end - start
    
    @staticmethod
    def assert_performance(execution_time, max_time, function_name="function"):
        """Assert that function executed within time limit"""
        assert execution_time <= max_time, \
            f"{function_name} took {execution_time:.3f}s, expected <= {max_time}s"

# Mock data generators for testing
class MockDataGenerators:
    """Generators for mock data in tests"""
    
    @staticmethod
    def generate_outlier_scores(base_scores, outlier_indices, outlier_multiplier=3):
        """Generate scores with specific outliers"""
        scores = base_scores.copy()
        for idx in outlier_indices:
            if idx < len(scores):
                scores[idx] *= outlier_multiplier
        return scores
    
    @staticmethod
    def generate_csv_content(data_dict, include_headers=True):
        """Generate CSV content string from data dictionary"""
        lines = []
        
        if include_headers:
            lines.append(','.join(data_dict.keys()))
        
        # Assuming all values are lists of same length
        n_rows = len(list(data_dict.values())[0])
        
        for i in range(n_rows):
            row = []
            for key in data_dict.keys():
                row.append(str(data_dict[key][i]))
            lines.append(','.join(row))
        
        return '\n'.join(lines)
    
    @staticmethod
    def generate_problematic_csv_data():
        """Generate CSV data with known problems for testing error handling"""
        return {
            'date': ['2024-07-01', 'invalid_date', '2024-07-05', '07/07/2024', ''],
            'temp': [85, 88, 'not_a_number', 92, 89],
            'humidity': [75, 80, 85, 'invalid', 78],
            'pace': ['7:30', '7:45', 'invalid_pace', '8:00', '7:35'],
            'avg_hr': [165, 'invalid_hr', 175, 180, 168],
            'distance': [3.1, 5.0, 3.1, 6.0, 4.0]
        }


# Test configuration constants
class TestConfig:
    """Configuration constants for tests"""
    
    # Performance thresholds (in seconds)
    MAX_CALCULATION_TIME = 1.0
    MAX_ML_TRAINING_TIME = 30.0
    MAX_DATA_PROCESSING_TIME = 5.0
    
    # Data validation thresholds
    MIN_HEAT_SCORE = 0
    MAX_HEAT_SCORE = 50
    MIN_TEMPERATURE = -20  # Very cold
    MAX_TEMPERATURE = 130  # Very hot
    MIN_HUMIDITY = 0
    MAX_HUMIDITY = 100
    MIN_PACE_SEC = 300     # 5:00 mile (very fast)
    MAX_PACE_SEC = 900     # 15:00 mile (very slow)
    MIN_HEART_RATE = 100
    MAX_HEART_RATE = 220
    MIN_DISTANCE = 0.1     # Very short
    MAX_DISTANCE = 50.0    # Very long
    
    # ML model validation thresholds
    MIN_R2_SCORE = -10.0   # Very poor model
    MAX_R2_SCORE = 1.0     # Perfect model
    MIN_MAE = 0.0
    MAX_MAE = 100.0        # Very high error
    
    # Sample data ranges
    TYPICAL_TEMP_RANGE = (70, 100)
    TYPICAL_HUMIDITY_RANGE = (40, 95)
    TYPICAL_PACE_RANGE = (360, 600)  # 6:00 to 10:00
    TYPICAL_HR_RANGE = (140, 190)
    TYPICAL_DISTANCE_RANGE = (1.0, 26.2)  # 1 mile to marathon


# Pytest fixtures that can be used across all test files
@pytest.fixture
def test_utils():
    """Provide TestUtils class to tests"""
    return TestUtils

@pytest.fixture
def performance_utils():
    """Provide PerformanceUtils class to tests"""
    return PerformanceUtils

@pytest.fixture
def mock_generators():
    """Provide MockDataGenerators class to tests"""
    return MockDataGenerators

@pytest.fixture
def test_config():
    """Provide TestConfig class to tests"""
    return TestConfig

# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup():
    """Automatic cleanup after test session"""
    yield
    
    # Clean up any temporary files that might have been left behind
    import os
    import glob
    
    temp_patterns = [
        'test_*.csv',
        'sample_*.csv', 
        'temp_*.csv',
        '*.tmp'
    ]
    
    for pattern in temp_patterns:
        for filepath in glob.glob(pattern):
            try:
                os.unlink(filepath)
            except OSError:
                pass  # File might be in use or already deleted


# Custom pytest assertions
def pytest_assertrepr_compare(config, op, left, right):
    """Custom assertion representations for better error messages"""
    if hasattr(left, 'shape') and hasattr(right, 'shape') and op == "==":
        # Better error messages for array comparisons
        return [
            f"Array comparison failed:",
            f"Left shape: {left.shape}",
            f"Right shape: {right.shape}",
            f"First few values:",
            f"Left:  {left.flatten()[:5] if len(left.flatten()) > 5 else left.flatten()}",
            f"Right: {right.flatten()[:5] if len(right.flatten()) > 5 else right.flatten()}",
        ]
    
    return None


# Test discovery helpers
def collect_ignore_patterns():
    """Return patterns to ignore during test collection"""
    return [
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "temp_*",
        "test_output*"
    ]


# Documentation and help
def print_test_help():
    """Print help information about running tests"""
    help_text = """
Heat Adaptation Analysis Test Suite
=====================================

Available test files:
- test_calculations.py: Core calculation functions
- test_ml_models.py: Machine learning models
- test_data_manager.py: Data loading and processing

Quick Start:
    # Run all tests
    pytest tests/
    
    # Run specific test file
    pytest tests/test_calculations.py
    
    # Run with verbose output and show print statements
    pytest tests/ -v -s
    
    # Run only fast tests (exclude slow ones)
    pytest tests/ -m "not slow"
    
    # Run with coverage report
    pytest tests/ --cov=heat_adaptation --cov-report=html

Test Markers:
    @pytest.mark.slow - Slow tests (ML training, large datasets)
    @pytest.mark.integration - Integration tests
    @pytest.mark.ml - Tests requiring ML libraries

Fixtures Available:
    - sample_run_data: Basic run data for testing
    - sample_environmental_conditions: Various weather conditions
    - test_utils: Utility functions for testing
    - mock_generators: Generate test data
    
Performance Testing:
    Tests include performance assertions to ensure functions
    complete within reasonable time limits.
    
Coverage:
    Aim for >90% code coverage across all modules.
    """
    
    print(help_text)


# Version information
__version__ = "1.0.0"
__test_author__ = "Heat Adaptation Analysis Team"
__test_description__ = "Comprehensive test suite for heat adaptation analysis system"

# Export commonly used items
__all__ = [
    'TestUtils',
    'PerformanceUtils', 
    'MockDataGenerators',
    'TestConfig',
    'print_test_help',
    'sample_run_data',
    'sample_environmental_conditions'
]