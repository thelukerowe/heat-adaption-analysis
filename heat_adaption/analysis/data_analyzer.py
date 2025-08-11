"""
Data Analysis Functions for Heat Adaptation

Handles outlier detection, threshold calculation, and statistical analysis.
"""

import numpy as np
from typing import List, Tuple
import statsmodels.api as sm


def detect_outliers(scores: List[float], method: str = 'iqr', threshold: float = 1.5) -> List[bool]:
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
    
    return outliers.tolist()


def estimate_threshold(scores: List[float]) -> float:
    """Estimate HSS threshold using median of scores"""
    return np.median(scores)


def loess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> np.ndarray:
    """LOESS smoothing (used internally only)"""
    smoothed = sm.nonparametric.lowess(y, x, frac=frac, it=0, return_sorted=False)
    return smoothed


def check_plateau(smoothed_scores: np.ndarray, threshold_pct: float = 0.01, days: int = 3) -> bool:
    """Plateau detection (safe division)"""
    if len(smoothed_scores) < 2:
        return False
    denom = np.maximum(smoothed_scores[:-1], 1e-8)
    increases = np.diff(smoothed_scores) / denom
    count = 0
    for inc in increases:
        if inc < threshold_pct:
            count += 1
            if count >= days:
                return True
        else:
            count = 0
    return False


def analyze_data_quality(run_data: List[dict]) -> dict:
    """Analyze data quality and provide statistics"""
    if not run_data:
        return {'error': 'No data provided'}
    
    scores = [run['raw_score'] for run in run_data]
    outliers = detect_outliers(scores)
    clean_scores = [score for i, score in enumerate(scores) if not outliers[i]]
    
    stats = {
        'total_runs': len(run_data),
        'outlier_count': sum(outliers),
        'clean_runs': len(clean_scores),
        'outlier_percentage': (sum(outliers) / len(run_data)) * 100,
        'mean_hss': np.mean(scores),
        'median_hss': np.median(scores),
        'std_hss': np.std(scores),
        'min_hss': np.min(scores),
        'max_hss': np.max(scores),
        'clean_mean_hss': np.mean(clean_scores) if clean_scores else 0,
        'clean_median_hss': np.median(clean_scores) if clean_scores else 0
    }
    
    return stats
