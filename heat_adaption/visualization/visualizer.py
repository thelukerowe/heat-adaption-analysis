"""
Visualization Module for Heat Adaptation Analysis

Creates comprehensive visualizations of heat adaptation data and predictions.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from typing import List, Dict
from datetime import datetime


def risk_color_adj(score: float) -> str:
    """Color coding for adjusted HSS scores"""
    if score < 10:
        return 'green'
    elif score < 15:
        return 'orange'
    else:
        return 'red'


def risk_color_rel(score: float) -> str:
    """Color coding for relative HSS scores"""
    if score < 0.8:
        return 'green'
    elif score < 1.2:
        return 'orange'
    else:
        return 'red'


def risk_color_adapted(score: float) -> str:
    """Lighter colors for adapted runs"""
    if score < 10:
        return 'lightgreen'
    elif score < 15:
        return 'moccasin'
    else:
        return 'lightcoral'


def create_original_visualization(run_data: List[Dict], dates: np.ndarray, raw_scores_arr: np.ndarray, 
                                adjusted_scores: np.ndarray, adapted_same_runs: np.ndarray, 
                                adapted_dates: np.ndarray, outliers: List[bool], 
                                improvement_pct: float, threshold: float, model_used: str,
                                ci_50_lower: np.ndarray, ci_50_upper: np.ndarray, 
                                ci_80_lower: np.ndarray, ci_80_upper: np.ndarray, 
                                ci_95_lower: np.ndarray, ci_95_upper: np.ndarray, 
                                plateau_days: int):
    """Create the original single comprehensive visualization"""
    
    adjusted_colors = [risk_color_adj(s) for s in adjusted_scores]
    relative_scores_list = [run['relative_score'] for run in run_data]
    relative_colors = [risk_color_rel(s) for s in relative_scores_list]

    # --- Plot combined graph (EXACTLY from original code) ---
    plt.figure(figsize=(14, 8))

    # Plot current runs with outlier markers
    current_colors = ['red' if outlier else 'blue' for outlier in outliers]
    current_markers = ['x' if outlier else 'o' for outlier in outliers]
    current_sizes = [80 if outlier else 60 for outlier in outliers]

    for i, (date, score, color, marker, size) in enumerate(zip(dates, [run['raw_score'] for run in run_data], current_colors, current_markers, current_sizes)):
        plt.scatter(date, score, color=color, marker=marker, s=size, alpha=0.8, zorder=5)

    plt.plot(dates, [run['raw_score'] for run in run_data], linestyle='-', color='blue', label='Current Raw HSS', linewidth=2, alpha=0.7)

    plt.plot(dates, adjusted_scores, marker='x', linestyle='--', color='purple', label='Adjusted HSS', linewidth=1)
    plt.scatter(dates, adjusted_scores, color=adjusted_colors, s=100, alpha=0.7, label='Adjusted HSS Risk')
    plt.scatter(dates, relative_scores_list, color=relative_colors, s=100, alpha=0.5, edgecolors='k', marker='o', label='Relative HSS Risk')

    # NEW: Plot the adapted same runs in the future with graduated confidence intervals
    if len(adapted_same_runs) > 0:
        plt.plot(adapted_dates, adapted_same_runs, marker='o', linestyle='-', color='darkgreen', 
                 label=f'Same Runs After {plateau_days}d Adaptation ({improvement_pct:.1f}% easier)', linewidth=2, markersize=6, zorder=4)
        
        # Add graduated confidence interval shading (outermost to innermost)
        if len(ci_95_lower) > 0:
            # 95% CI - Red (outermost, least likely)
            plt.fill_between(adapted_dates, ci_95_lower, ci_95_upper, color='red', alpha=0.15, label='95% CI (Possible)', zorder=1)
            
            # 80% CI - Yellow/Orange (middle probability)
            plt.fill_between(adapted_dates, ci_80_lower, ci_80_upper, color='orange', alpha=0.25, label='80% CI (Probable)', zorder=2)
            
            # 50% CI - Green (innermost, most likely)
            plt.fill_between(adapted_dates, ci_50_lower, ci_50_upper, color='darkgreen', alpha=0.35, label='50% CI (Most Likely)', zorder=3)

    # Add legend for outliers
    if sum(outliers) > 0:
        plt.scatter([], [], color='red', marker='x', s=80, label='Outlier Runs (Excluded from Model)', alpha=0.8)

    plt.title(f"Heat Strain Score: Current vs Adapted Performance ({model_used.replace('_', ' ').title()} Model)")
    plt.xlabel("Date")
    plt.ylabel("Heat Strain Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    time_span = (max(dates) - min(dates)).days
    if time_span > 30:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()
