"""
Feature engineering for machine learning models.

This module creates enhanced features from raw running data to improve
machine learning model performance in heat adaptation prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .calculations import HeatCalculations


class FeatureEngineer:
    """Feature engineering for heat adaptation ML models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.heat_calc = HeatCalculations()
        self.feature_names_ = None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced features for machine learning models.
        
        Args:
            df: DataFrame with basic run data
            
        Returns:
            DataFrame with engineered features
        """
        features_df = df.copy()
        
        # Basic engineered features
        features_df = self._add_basic_features(features_df)
        
        # Interaction features
        features_df = self._add_interaction_features(features_df)
        
        # Time-based features
        features_df = self._add_time_features(features_df)
        
        # Environmental stress features
        features_df = self._add_stress_features(features_df)
        
        # Performance features
        features_df = self._add_performance_features(features_df)
        
        # Trend features
        features_df = self._add_trend_features(features_df)
        
        return features_df
    
    def _add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features."""
        df = df.copy()
        
        # Heat index (feels-like temperature)
        df['heat_index'] = self.heat_calc.calculate_heat_index(
            df['temp'], df['humidity']
        )
        
        # Performance ratios
        df['pace_vs_pr'] = df['pace_sec'] / df['mile_pr_sec']
        df['hr_efficiency'] = df['avg_hr'] / df['max_hr']
        
        # Environmental severity
        df['temp_severity'] = self.heat_calc.normalize(df['temp'], 60, 100)
        df['humidity_severity'] = self.heat_calc.normalize(df['humidity'], 40, 95)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between variables."""
        df = df.copy()
        
        # Environmental interactions
        df['temp_humidity_interaction'] = df['temp'] * df['humidity'] / 100
        df['heat_index_squared'] = df['heat_index'] ** 2
        df['environmental_load'] = (
            df['temp_severity'] * 0.6 + df['humidity_severity'] * 0.4
        )
        
        # Performance interactions
        df['pace_hr_interaction'] = df['pace_vs_pr'] * df['hr_efficiency']
        df['distance_pace_interaction'] = df['distance'] * df['pace_vs_pr']
        df['effort_index'] = (
            df['hr_efficiency'] * 0.7 + df['pace_vs_pr'] * 0.3
        )
        
        # Combined environmental-performance interactions
        df['env_perf_interaction'] = (
            df['environmental_load'] * df['effort_index']
        )
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()
        
        if 'date' in df.columns and len(df) > 0:
            # Ensure date column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Days since start
            df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
            
            # Day of year (seasonal effects)
            df['day_of_year'] = df['date'].dt.dayofyear
            df['season_factor'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            
            # Week of year
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Days since last run
            df_sorted = df.sort_values('date')
            df['days_since_last_run'] = df_sorted['date'].diff().dt.days.fillna(0)
            
            # Reorder back to original index
            df = df.reindex(df.index)
        else:
            # Default values if no date information
            df['days_since_start'] = 0
            df['day_of_year'] = 180  # Mid-year default
            df['season_factor'] = 0
            df['week_of_year'] = 26
            df['days_since_last_run'] = 0
        
        return df
    
    def _add_stress_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add stress and load features."""
        df = df.copy()
        
        # Environmental stress
        df['environmental_stress'] = (
            self.heat_calc.normalize(df['temp'], 60, 100) * 0.6 + 
            self.heat_calc.normalize(df['humidity'], 40, 95) * 0.4
        )
        
        # Performance stress
        df['performance_stress'] = (
            df['pace_vs_pr'] * 0.4 + 
            df['hr_efficiency'] * 0.6
        )
        
        # Combined stress index
        df['total_stress'] = (
            df['environmental_stress'] * 0.5 +
            df['performance_stress'] * 0.5
        )
        
        # Heat danger categories
        df['heat_danger_level'] = pd.cut(
            df['heat_index'],
            bins=[0, 80, 90, 100, 110, float('inf')],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype(float)
        
        return df
    
    def _add_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add performance-related features."""
        df = df.copy()
        
        # Distance categories
        df['distance_category'] = pd.cut(
            df['distance'],
            bins=[0, 2, 4, 7, 15, float('inf')],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype(float)
        
        # Pace categories relative to PR
        df['pace_category'] = pd.cut(
            df['pace_vs_pr'],
            bins=[0, 1.0, 1.1, 1.2, 1.4, float('inf')],
            labels=[4, 3, 2, 1, 0],  # Reverse order (faster = higher)
            include_lowest=True
        ).astype(float)
        
        # HR categories
        df['hr_category'] = pd.cut(
            df['hr_efficiency'],
            bins=[0, 0.7, 0.8, 0.85, 0.9, 1.0],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        ).astype(float)
        
        # Workout intensity score
        df['workout_intensity'] = (
            df['pace_category'] * 0.4 +
            df['hr_category'] * 0.3 +
            df['distance_category'] * 0.3
        )
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trending and rolling features."""
        df = df.copy()
        
        if len(df) >= 2:
            # Sort by date for proper rolling calculations
            df_sorted = df.sort_values('date') if 'date' in df.columns else df
            
            # Rolling averages (adapt window size to data availability)
            window_size = min(3, len(df))
            
            # Rolling HSS features
            df_sorted['hss_rolling_3'] = (
                df_sorted['raw_score']
                .rolling(window=window_size, min_periods=1)
                .mean()
            )
            
            df_sorted['hss_rolling_std'] = (
                df_sorted['raw_score']
                .rolling(window=window_size, min_periods=1)
                .std()
                .fillna(0)
            )
            
            # Rolling environmental features
            df_sorted['temp_rolling_3'] = (
                df_sorted['temp']
                .rolling(window=window_size, min_periods=1)
                .mean()
            )
            
            df_sorted['humidity_rolling_3'] = (
                df_sorted['humidity']
                .rolling(window=window_size, min_periods=1)
                .mean()
            )
            
            # Trend features (differences)
            df_sorted['hss_trend'] = (
                df_sorted['raw_score'].diff().fillna(0)
            )
            
            df_sorted['temp_trend'] = (
                df_sorted['temp'].diff().fillna(0)
            )
            
            df_sorted['humidity_trend'] = (
                df_sorted['humidity'].diff().fillna(0)
            )
            
            df_sorted['performance_trend'] = (
                df_sorted['pace_vs_pr'].diff().fillna(0)
            )
            
            # Reorder back to original index
            df = df_sorted.reindex(df.index)
        
        else:
            # Single data point - use current values
            df['hss_rolling_3'] = df['raw_score']
            df['hss_rolling_std'] = 0
            df['temp_rolling_3'] = df['temp']
            df['humidity_rolling_3'] = df['humidity']
            df['hss_trend'] = 0
            df['temp_trend'] = 0
            df['humidity_trend'] = 0
            df['performance_trend'] = 0
        
        return df
    
    def get_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Get features suitable for ML models.
        
        Args:
            df: DataFrame with engineered features
            
        Returns:
            Tuple of (feature_dataframe, feature_names_list)
        """
        # Define ML feature columns
        ml_feature_cols = [
            # Basic environmental
            'temp', 'humidity', 'heat_index',
            
            # Performance
            'pace_vs_pr', 'hr_efficiency', 'distance',
            
            # Interactions
            'temp_humidity_interaction', 'pace_hr_interaction',
            'env_perf_interaction',
            
            # Rolling/trend features
            'hss_rolling_3', 'temp_rolling_3', 'humidity_rolling_3',
            'hss_trend', 'temp_trend', 'performance_trend',
            
            # Time features
            'days_since_start', 'season_factor',
            
            # Stress features
            'environmental_stress', 'performance_stress', 'total_stress',
            
            # Categorical features (converted to numeric)
            'heat_danger_level', 'distance_category', 'pace_category',
            'hr_category', 'workout_intensity'
        ]
        
        # Select available features (some might not exist in small datasets)
        available_features = [
            col for col in ml_feature_cols 
            if col in df.columns
        ]
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Store feature names for later reference
        self.feature_names_ = available_features
        
        return X, available_features
    
    def get_feature_importance_interpretation(self, feature_names: List[str], 
                                           importances: List[float], 
                                           top_n: int = 10) -> Dict[str, str]:
        """
        Get human-readable interpretation of feature importance.
        
        Args:
            feature_names: List of feature names
            importances: List of feature importance values
            top_n: Number of top features to interpret
            
        Returns:
            Dictionary mapping feature names to interpretations
        """
        interpretations = {
            # Environmental
            'temp': 'Temperature - Primary heat stress factor',
            'humidity': 'Humidity - Affects sweat evaporation efficiency',
            'heat_index': 'Heat Index - Combined temperature and humidity effect',
            
            # Performance
            'pace_vs_pr': 'Pace vs PR - Running intensity relative to ability',
            'hr_efficiency': 'Heart Rate Efficiency - Cardiovascular stress',
            'distance': 'Distance - Total heat exposure duration',
            
            # Interactions
            'temp_humidity_interaction': 'Temperature × Humidity - Combined environmental stress',
            'pace_hr_interaction': 'Pace × Heart Rate - Combined performance stress',
            'env_perf_interaction': 'Environmental × Performance - Total adaptation challenge',
            
            # Trends
            'hss_rolling_3': 'Recent Heat Strain Average - Current adaptation level',
            'hss_trend': 'Heat Strain Trend - Adaptation progress direction',
            'temp_rolling_3': 'Recent Temperature Average - Environmental exposure',
            'performance_trend': 'Performance Trend - Fitness/adaptation changes',
            
            # Time
            'days_since_start': 'Training Duration - Cumulative adaptation time',
            'season_factor': 'Seasonal Effect - Natural acclimatization timing',
            
            # Stress
            'environmental_stress': 'Environmental Stress Score - Heat challenge level',
            'performance_stress': 'Performance Stress Score - Effort intensity',
            'total_stress': 'Total Stress - Combined adaptation stimulus',
            
            # Categories
            'heat_danger_level': 'Heat Danger Category - Risk classification',
            'workout_intensity': 'Workout Intensity - Training load factor',
        }
        
        # Get top features
        feature_importance_pairs = list(zip(feature_names, importances))
        top_features = sorted(feature_importance_pairs, 
                            key=lambda x: x[1], reverse=True)[:top_n]
        
        result = {}
        for feature_name, importance in top_features:
            interpretation = interpretations.get(
                feature_name, 
                f'{feature_name} - Custom engineered feature'
            )
            result[feature_name] = f'{interpretation} (Importance: {importance:.3f})'
        
        return result
    
    def create_adaptation_features(self, df: pd.DataFrame, 
                                 improvement_pct: float) -> pd.DataFrame:
        """
        Create features for adapted performance prediction.
        
        Args:
            df: Original dataframe with run data
            improvement_pct: Expected improvement percentage
            
        Returns:
            DataFrame with adapted performance features
        """
        adapted_df = df.copy()
        
        # Apply improvement to heat strain score
        adapted_df['adapted_raw_score'] = (
            df['raw_score'] * (1 - improvement_pct / 100)
        )
        adapted_df['adapted_raw_score'] = np.maximum(
            adapted_df['adapted_raw_score'], 0
        )  # Ensure non-negative
        
        # Recalculate derived features based on improved performance
        adapted_df['adapted_relative_score'] = (
            adapted_df['adapted_raw_score'] / df['raw_score']
        ).fillna(1.0)
        
        # Environmental features remain the same (same conditions)
        # Performance stress features would be lower due to adaptation
        adapted_df['adapted_performance_stress'] = (
            df['performance_stress'] * (1 - improvement_pct / 200)
        )  # Less dramatic change in performance stress
        
        adapted_df['adapted_total_stress'] = (
            df['environmental_stress'] * 0.5 +
            adapted_df['adapted_performance_stress'] * 0.5
        )
        
        return adapted_df