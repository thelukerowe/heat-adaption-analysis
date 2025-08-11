"""
Machine Learning models for heat adaptation prediction.

This module contains the ML models used to predict heat adaptation improvements,
including Random Forest and Gradient Boosting regressors with cross-validation
and feature importance analysis.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


class ModelPerformance:
    """Container for model performance metrics."""
    
    def __init__(self, model_name: str, mse: float, mae: float, 
                 r2: float, rmse: float):
        self.model_name = model_name
        self.mse = mse
        self.mae = mae
        self.r2 = r2
        self.rmse = rmse
    
    def __repr__(self):
        return (f"ModelPerformance(model='{self.model_name}', "
                f"RMSE={self.rmse:.2f}, MAE={self.mae:.2f}, R²={self.r2:.3f})")


class HeatAdaptationMLModel:
    """
    Machine Learning model for heat adaptation prediction.
    
    This class trains and manages multiple ML models (Random Forest and 
    Gradient Boosting) to predict heat strain scores and adaptation improvements.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ML model.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=2,
                random_state=random_state,
                n_jobs=-1  # Use all available cores
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_state
            )
        }
        
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.is_trained = False
        self.performance_metrics = {}
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    min_samples: int = 5, test_size: float = 0.2) -> Optional[ModelPerformance]:
        """
        Train multiple ML models and select the best one.
        
        Args:
            X: Feature matrix
            y: Target values (heat strain scores)
            min_samples: Minimum samples required for training
            test_size: Fraction of data to use for testing
            
        Returns:
            Performance metrics of the best model, or None if training failed
        """
        print(f"\n🤖 ML MODEL TRAINING")
        print(f"   Training on {len(X)} samples with {len(X.columns)} features...")
        
        if len(X) < min_samples:
            print(f"   ⚠️  Insufficient data ({len(X)} < {min_samples}). "
                  "Using baseline model.")
            self.is_trained = False
            return None
        
        self.feature_names = list(X.columns)
        
        # Split data for training and testing
        if len(X) < 10:
            # Use all data for training if very limited
            X_train, X_test = X, X
            y_train, y_test = y, y
            print(f"   📊 Using all {len(X)} samples for training (limited data)")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
            print(f"   📊 Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate models
        model_performances = {}
        
        for name, model in self.models.items():
            try:
                print(f"   🔄 Training {name.replace('_', ' ').title()}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Store performance
                performance = ModelPerformance(name, mse, mae, r2, rmse)
                model_performances[name] = {
                    'performance': performance,
                    'model': model
                }
                
                print(f"      RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")
                
                # Cross-validation if enough data
                if len(X_train) >= 10:
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, 
                        cv=min(5, len(X_train) // 2), 
                        scoring='neg_mean_squared_error'
                    )
                    cv_rmse = np.sqrt(-cv_scores.mean())
                    print(f"      CV-RMSE: {cv_rmse:.2f} ± {np.sqrt(-cv_scores).std():.2f}")
                
            except Exception as e:
                print(f"   ❌ {name} training failed: {str(e)}")
                continue
        
        if model_performances:
            # Select best model based on lowest RMSE
            best_name = min(
                model_performances.keys(),
                key=lambda k: model_performances[k]['performance'].rmse
            )
            
            self.best_model = model_performances[best_name]['model']
            self.best_model_name = best_name
            self.is_trained = True
            self.performance_metrics = model_performances
            
            print(f"   🏆 Best Model: {best_name.replace('_', ' ').title()}")
            
            # Print feature importance
            self._print_feature_importance()
            
            return model_performances[best_name]['performance']
        
        else:
            print("   ❌ All models failed to train")
            self.is_trained = False
            return None
    
    def predict(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted values, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, 
                               n_estimators: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimation.
        
        For Random Forest, uses individual tree predictions for uncertainty.
        For Gradient Boosting, uses staged predictions.
        
        Args:
            X: Feature matrix for prediction
            n_estimators: Number of estimators to use (None for all)
            
        Returns:
            Tuple of (predictions, prediction_std)
        """
        if not self.is_trained:
            return None, None
        
        # Ensure feature order matches training
        if self.feature_names:
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        
        if self.best_model_name == 'random_forest':
            # Get predictions from individual trees
            tree_predictions = np.array([
                tree.predict(X_scaled) 
                for tree in self.best_model.estimators_[:n_estimators]
            ])
            
            predictions = np.mean(tree_predictions, axis=0)
            prediction_std = np.std(tree_predictions, axis=0)
            
        elif self.best_model_name == 'gradient_boost':
            # Use staged predictions for uncertainty estimation
            staged_predictions = list(
                self.best_model.staged_predict(X_scaled)
            )
            
            if n_estimators:
                staged_predictions = staged_predictions[:n_estimators]
            
            staged_array = np.array(staged_predictions)
            predictions = staged_array[-1]  # Final prediction
            
            # Estimate uncertainty from prediction stability
            prediction_std = np.std(staged_array[-10:], axis=0) if len(staged_array) >= 10 else np.zeros_like(predictions)
        
        else:
            # Fallback to regular prediction
            predictions = self.best_model.predict(X_scaled)
            prediction_std = np.zeros_like(predictions)
        
        return predictions, prediction_std
    
    def get_feature_importance(self, top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance, or None if not available
        """
        if not self.is_trained or not hasattr(self.best_model, 'feature_importances_'):
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def _print_feature_importance(self, top_n: int = 5):
        """Print top feature importance for model interpretation."""
        importance_df = self.get_feature_importance(top_n)
        
        if importance_df is not None:
            print(f"\n   📈 TOP FEATURE IMPORTANCE:")
            for _, row in importance_df.iterrows():
                print(f"      {row['feature']}: {row['importance']:.3f}")
    
    def calculate_residuals(self, X: pd.DataFrame, y: pd.Series) -> Optional[np.ndarray]:
        """
        Calculate residuals (actual - predicted) for the trained model.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Residuals array, or None if model not trained
        """
        if not self.is_trained:
            return None
        
        predictions = self.predict(X)
        if predictions is None:
            return None
        
        return y.values - predictions
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Optional[Dict[str, float]]:
        """
        Evaluate model performance on given data.
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_trained:
            return None
        
        predictions = self.predict(X)
        if predictions is None:
            return None
        
        metrics = {
            'mse': mean_squared_error(y, predictions),
            'mae': mean_absolute_error(y, predictions),
            'r2': r2_score(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions))
        }
        
        return metrics
    
    def save_model(self, filepath: str):
        """
        Save the trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        import joblib
        
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a trained model from file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded HeatAdaptationMLModel instance
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls()
        instance.best_model = model_data['best_model']
        instance.best_model_name = model_data['best_model_name']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.performance_metrics = model_data['performance_metrics']
        instance.is_trained = True
        
        print(f"Model loaded from {filepath}")
        return instance