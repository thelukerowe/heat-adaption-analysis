"""
Analysis Package for Heat Adaptation

Provides statistical analysis, outlier detection, and data quality assessment.
"""

from .data_analyzer import (
    detect_outliers,
    estimate_threshold,
    loess_smooth,
    check_plateau,
    analyze_data_quality
)

__all__ = [
    'detect_outliers',
    'estimate_threshold', 
    'loess_smooth',
    'check_plateau',
    'analyze_data_quality'
]
