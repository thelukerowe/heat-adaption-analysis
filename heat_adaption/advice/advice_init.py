"""
Advice Package for Heat Adaptation Analysis

Provides personalized training recommendations and guidance based on analysis results.
"""

from .advisor import generate_adaptation_advice

__all__ = [
    'generate_adaptation_advice'
]
