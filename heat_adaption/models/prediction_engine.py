"""
Prediction Engine for Heat Adaptation Analysis

This module handles model selection, prediction logic, and adaptation forecasting
for heat adaptation analysis. It integrates machine learning models with
research-based approaches to provide comprehensive heat adaptation predictions.

Classes:
    PredictionEngine: Main class for model selection and predictions
    ModelResult: Data class for storing prediction results
    
Functions:
    apply_physiological_limits: Apply realistic physiological constraints
    calculate_confidence_intervals: Calculate prediction confidence intervals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import stats

from ..models.ml_models import HeatAdaptationMLModel


@dataclass
class ModelResult:
    """Data class for storing model prediction results"""
    model_type: str
    improvement_pct: Optional[float]
    plateau_days: int
    adapted_runs: np.ndarray
    adapted_dates: np.ndarray
    confidence_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    baseline_hss: float
    residuals: np.ndarray
    model_performance: Optional[Dict] = None


class PredictionEngine:
    """
    Main prediction engine for heat adaptation analysis.
    
    This class handles model selection, training, and prediction generation
    based on available data quantity and quality.
    """
    
    # Constants for model selection thresholds
    MIN_DATA_POINTS_ML = 10
    MIN_DATA_POINTS_COMPLEX = 30 
    MIN_DATA_POINTS_SIMPLE = 5
    
    def __init__(self):
        """Initialize the prediction engine"""
        self.ml_model = HeatAdaptationMLModel()
        self.model_result = None
        
    def apply_physiological_limits(self, improvement_pct: Optional[float], 
                                 max_improvement: float = 25.0) -> Optional[float]:
        """
        Apply realistic physiological limits to heat adaptation improvement.
        
        Args:
            improvement_pct: Raw improvement percentage from model
            max_improvement: Maximum physiologically realistic improvement
            
        Returns:
            Limited improvement percentage with context information
        """
        if improvement_pct is None:
            return None
        
        # Cap improvement at physiologically realistic levels
        limited_improvement = min(abs(improvement_pct), max_improvement)
        
        # Add some context based on improvement magnitude
        if limited_improvement > 20:
            print(f"Note: Raw model predicted {abs(improvement_pct):.1f}% improvement, "
                  f"capped at {limited_improvement:.1f}% (physiological maximum)")
        elif limited_improvement > 15:
            print(f"Note: {limited_improvement:.1f}% improvement suggests you're relatively heat-naive")
        elif limited_improvement > 10:
            print(f"Note: {limited_improvement:.1f}% improvement suggests moderate heat adaptation potential")
        else:
            print(f"Note: {limited_improvement:.1f}% improvement suggests you're already well heat-adapted")
        
        return limited_improvement
    
    def calculate_confidence_intervals(self, predictions: np.ndarray, 
                                     residuals: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: Array of predicted values
            residuals: Array of model residuals
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            Dictionary with confidence interval bounds for different levels
        """
        if len(residuals) < 3:
            # Return predictions as bounds if insufficient data
            return {
                '50': (predictions, predictions),
                '80': (predictions, predictions), 
                '95': (predictions, predictions)
            }
        
        # Calculate standard error from residuals
        std_error = np.std(residuals)
        
        # T-distribution critical values
        df = len(residuals) - 2  # degrees of freedom
        alpha = 1 - confidence_level
        
        if df > 0:
            t_50 = stats.t.ppf(0.75, df)    # 50% CI
            t_80 = stats.t.ppf(0.90, df)    # 80% CI  
            t_95 = stats.t.ppf(1 - alpha/2, df)  # 95% CI
        else:
            t_50, t_80, t_95 = 0.67, 1.28, 2.0
        
        # Calculate margins for each confidence level
        margin_50 = t_50 * std_error
        margin_80 = t_80 * std_error
        margin_95 = t_95 * std_error
        
        return {
            '50': (predictions - margin_50, predictions + margin_50),
            '80': (predictions - margin_80, predictions + margin_80),
            '95': (predictions - margin_95, predictions + margin_95)
        }
    
    def _select_model_type(self, clean_data_count: int, ml_trained: bool) -> str:
        """
        Select appropriate model type based on data availability.
        
        Args:
            clean_data_count: Number of clean data points available
            ml_trained: Whether ML model was successfully trained
            
        Returns:
            String identifier for selected model type
        """
        if clean_data_count >= self.MIN_DATA_POINTS_ML and ml_trained:
            return "machine_learning"
        elif clean_data_count >= self.MIN_DATA_POINTS_COMPLEX:
            return "complex_logarithmic"
        elif clean_data_count >= self.MIN_DATA_POINTS_SIMPLE:
            return "research_based"
        else:
            return "insufficient_data"
    
    def _predict_ml_model(self, run_data: List[Dict], df_features: pd.DataFrame, 
                         threshold: float, outliers: List[bool]) -> Tuple[float, int, float]:
        """
        Generate predictions using machine learning model.
        
        Args:
            run_data: List of run dictionaries
            df_features: DataFrame with engineered features
            threshold: HSS threshold value
            outliers: Boolean list indicating outlier runs
            
        Returns:
            Tuple of (improvement_pct, plateau_days, recent_val)
        """
        print(f"   🤖 Using Machine Learning Model")
        
        # Use recent runs for prediction baseline
        recent_runs = run_data[-3:] if len(run_data) >= 3 else run_data
        recent_features = df_features.tail(len(recent_runs))
        
        # Get ML prediction for current performance level
        ml_predictions = self.ml_model.predict(recent_features)
        if ml_predictions is not None:
            recent_val = np.mean(ml_predictions)
        else:
            recent_val = np.mean([run['raw_score'] for run in recent_runs])
        
        # Estimate plateau days using environmental factors
        avg_temp = np.mean([run['temp'] for run in recent_runs])
        avg_humidity = np.mean([run['humidity'] for run in recent_runs])
        
        # ML-enhanced environmental analysis
        if avg_temp >= 88 and avg_humidity >= 85:
            plateau_days = 12
            base_improvement = 18
        elif avg_temp >= 82 and avg_humidity >= 75:
            plateau_days = 10
            base_improvement = 15
        elif avg_temp >= 75 and avg_humidity >= 65:
            plateau_days = 8
            base_improvement = 12
        else:
            plateau_days = 6
            base_improvement = 8
        
        # Adjust based on ML model's understanding of current adaptation
        model_performance = getattr(self.ml_model, '_last_performance', None)
        if model_performance and 'r2' in model_performance:
            # If ML model shows good fit (R² > 0.5), trust its pattern recognition
            if model_performance['r2'] > 0.5:
                adaptation_factor = min(recent_val / threshold, 2.0) if threshold > 0 else 1.0
                base_improvement *= adaptation_factor
        
        raw_improvement_pct = base_improvement
        improvement_pct = self.apply_physiological_limits(raw_improvement_pct, max_improvement=25.0)
        
        print(f"   📊 ML-predicted recent baseline: {recent_val:.2f}")
        print(f"   🎯 ML-enhanced improvement estimate: {improvement_pct:.1f}%")
        
        return improvement_pct, plateau_days, recent_val
    
    def _predict_complex_model(self, clean_days: List[int], 
                              clean_adjusted_scores: List[float],
                              adjusted_scores: np.ndarray, 
                              outliers: List[bool]) -> Tuple[float, int, float, np.ndarray]:
        """
        Generate predictions using complex logarithmic model.
        
        Args:
            clean_days: Days since start for clean data points
            clean_adjusted_scores: Adjusted scores for clean data points
            adjusted_scores: All adjusted scores
            outliers: Boolean list indicating outlier runs
            
        Returns:
            Tuple of (improvement_pct, plateau_days, recent_val, residuals)
        """
        print(f"   📈 Using Complex Logarithmic Model")
        
        # Logarithmic regression
        Xln = np.log(np.array(clean_days) + 1).reshape(-1, 1)
        X_design = np.hstack([Xln, np.ones_like(Xln)])
        y_fit = np.array(clean_adjusted_scores).reshape(-1, 1)

        coef, residuals_sum, _, _ = np.linalg.lstsq(X_design, y_fit, rcond=None)
        a, b = coef.flatten()
        
        fitted_values = (a * np.log(np.array(clean_days) + 1) + b).flatten()
        residuals = np.array(clean_adjusted_scores) - fitted_values

        plateau_days = 14
        recent_val = np.mean([score for i, score in enumerate(adjusted_scores) if not outliers[i]][-3:])
        future_mean = a * np.log(clean_days[-1] + plateau_days + 1) + b
        raw_improvement_pct = ((recent_val - future_mean) / recent_val) * 100 if recent_val != 0 else None
        improvement_pct = self.apply_physiological_limits(raw_improvement_pct, max_improvement=25.0)
        
        return improvement_pct, plateau_days, recent_val, residuals
    
    def _predict_research_model(self, adjusted_scores: np.ndarray, 
                               outliers: List[bool]) -> Tuple[float, int, float]:
        """
        Generate predictions using research-based model.
        
        Args:
            adjusted_scores: Array of adjusted HSS scores
            outliers: Boolean list indicating outlier runs
            
        Returns:
            Tuple of (improvement_pct, plateau_days, baseline_hss)
        """
        print(f"   📚 Using Research-Based Model")
        
        # Use recent clean scores for baseline
        recent_scores = [score for i, score in enumerate(adjusted_scores) if not outliers[i]][-3:]
        baseline_hss = np.mean(recent_scores)
        
        # Research-based improvement estimates
        if baseline_hss > 15:
            base_improvement = 20
            plateau_days = 12
        elif baseline_hss > 10:
            base_improvement = 15
            plateau_days = 10
        elif baseline_hss > 6:
            base_improvement = 10
            plateau_days = 8
        else:
            base_improvement = 5
            plateau_days = 6
        
        improvement_pct = self.apply_physiological_limits(base_improvement, max_improvement=25.0)
        
        return improvement_pct, plateau_days, baseline_hss
    
    def _calculate_adapted_runs(self, run_data: List[Dict], improvement_pct: float,
                               dates: np.ndarray, plateau_days: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate adapted run performance and future dates.
        
        Args:
            run_data: List of run dictionaries
            improvement_pct: Percentage improvement from adaptation
            dates: Array of original run dates
            plateau_days: Days until adaptation plateau
            
        Returns:
            Tuple of (adapted_runs_array, adapted_dates_array)
        """
        print(f"   🔮 Calculating future adapted performance...")
        
        # Calculate future dates
        time_gaps = np.diff(dates) if len(dates) > 1 else [timedelta(days=1)]
        adapted_dates = []
        
        start_date = dates[-1] + timedelta(days=plateau_days)
        adapted_dates.append(start_date)
        
        for i in range(len(time_gaps)):
            next_date = adapted_dates[-1] + time_gaps[i]
            adapted_dates.append(next_date)
        
        adapted_dates = np.array(adapted_dates)
        
        # Calculate adapted runs
        adapted_runs = []
        for run in run_data:
            adapted_score = run['raw_score'] * (1 - improvement_pct / 100)
            adapted_runs.append(max(0, adapted_score))
        
        adapted_runs = np.array(adapted_runs)
        
        print(f"   ✅ Calculated adapted performance for {len(adapted_runs)} runs")
        
        return adapted_runs, adapted_dates
    
    def generate_predictions(self, run_data: List[Dict], df_features: pd.DataFrame,
                           outliers: List[bool], threshold: float,
                           dates: np.ndarray, adjusted_scores: np.ndarray) -> Optional[ModelResult]:
        """
        Generate heat adaptation predictions using appropriate model.
        
        Args:
            run_data: List of run data dictionaries
            df_features: DataFrame with engineered features  
            outliers: Boolean list indicating outlier runs
            threshold: HSS threshold value
            dates: Array of run dates
            adjusted_scores: Array of adjusted HSS scores
            
        Returns:
            ModelResult object with predictions, or None if insufficient data
        """
        print(f"\n🎯 MODEL SELECTION & PREDICTION")
        
        # Prepare clean data
        clean_days = [(dates[i] - dates[0]).days for i in range(len(outliers)) if not outliers[i]]
        clean_adjusted_scores = [adjusted_scores[i] for i in range(len(outliers)) if not outliers[i]]
        clean_df = df_features[~pd.Series(outliers)].copy()
        
        print(f"   Clean data points: {len(clean_days)}")
        
        # Train ML model if enough data
        model_performance = None
        if len(clean_df) >= self.MIN_DATA_POINTS_ML:
            model_performance = self.ml_model.train_models(clean_df, target_col='raw_score')
            
        # Select model type
        model_type = self._select_model_type(len(clean_days), self.ml_model.is_trained)
        
        if model_type == "insufficient_data":
            print(f"   ⚠️  Warning: Only {len(clean_days)} clean data points. "
                  f"Need {self.MIN_DATA_POINTS_SIMPLE}+ for any prediction.")
            return None
        
        # Generate predictions based on model type
        residuals = np.array([0.5, -0.5])  # Default residuals
        
        if model_type == "machine_learning":
            improvement_pct, plateau_days, recent_val = self._predict_ml_model(
                run_data, df_features, threshold, outliers)
            
            # Use ML model residuals if available
            if model_performance:
                ml_pred_all = self.ml_model.predict(clean_df)
                if ml_pred_all is not None:
                    residuals = clean_df['raw_score'].values - ml_pred_all
                    
            baseline_hss = recent_val
            
        elif model_type == "complex_logarithmic":
            improvement_pct, plateau_days, baseline_hss, residuals = self._predict_complex_model(
                clean_days, clean_adjusted_scores, adjusted_scores, outliers)
                
        elif model_type == "research_based":
            improvement_pct, plateau_days, baseline_hss = self._predict_research_model(
                adjusted_scores, outliers)
            
        if improvement_pct is None:
            return None
            
        # Calculate adapted runs and dates
        adapted_runs, adapted_dates = self._calculate_adapted_runs(
            run_data, improvement_pct, dates, plateau_days)
        
        # Calculate confidence intervals
        confidence_intervals = self.calculate_confidence_intervals(adapted_runs, residuals)
        
        # Create result object
        self.model_result = ModelResult(
            model_type=model_type,
            improvement_pct=improvement_pct,
            plateau_days=plateau_days,
            adapted_runs=adapted_runs,
            adapted_dates=adapted_dates,
            confidence_intervals=confidence_intervals,
            baseline_hss=baseline_hss,
            residuals=residuals,
            model_performance=model_performance
        )
        
        return self.model_result
    
    def print_prediction_summary(self, run_data: List[Dict], outliers: List[bool]) -> None:
        """
        Print summary of predictions to console.
        
        Args:
            run_data: List of run data dictionaries
            outliers: Boolean list indicating outlier runs
        """
        if not self.model_result:
            print("No predictions available.")
            return
            
        result = self.model_result
        
        print(f"\n🎯 HEAT ADAPTATION PREDICTIONS ({result.model_type.replace('_', ' ').title()} Model)")
        print(f"{'='*60}")
        
        if len(result.adapted_runs) > 0:
            print(f"Same Runs After {result.plateau_days} Days of Heat Adaptation:")
            
            ci_50_lower, ci_50_upper = result.confidence_intervals['50']
            
            for i, (original, adapted) in enumerate(zip([run['raw_score'] for run in run_data], result.adapted_runs)):
                date_str = run_data[i]['date'].strftime('%Y-%m-%d')
                improvement = ((original - adapted) / original) * 100
                outlier_flag = " [OUTLIER]" if outliers[i] else ""
                ci_50_range = f"{ci_50_lower[i]:.2f}-{ci_50_upper[i]:.2f}"
                
                print(f"Run {date_str}{outlier_flag}: Current HSS = {original:.2f} → Adapted HSS = {adapted:.2f}")
                print(f"  Most likely range (50% CI): {ci_50_range}")
                print(f"  Expected improvement: {improvement:.1f}% easier")


def create_prediction_engine() -> PredictionEngine:
    """Factory function to create a new PredictionEngine instance"""
    return PredictionEngine()
