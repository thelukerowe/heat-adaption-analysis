"""
Visualization Package for Heat Adaptation Analysis

Provides visualization functions for heat adaptation data and predictions.
"""

from .visualizer import (
    create_original_visualization,
    risk_color_adj,
    risk_color_rel,
    risk_color_adapted
)

__all__ = [
    'create_original_visualization',
    'risk_color_adj',
    'risk_color_rel', 
    'risk_color_adapted'
]
