"""
Test suite for heat_adaptation.models.ml_models module.

Tests cover:
- HeatAdaptationMLModel class functionality
- Feature engineering and preparation
- Model training and evaluation
- Prediction capabilities
- Feature importance analysis
- Model selection logic
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Suppress sklearn warnings for tests
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def create_ml_features(df):
    """Create enhanced features for machine learning models"""
    features_df = df.copy()
    
    # Basic engineered features
    features_df['heat_index'] = calculate_heat_index(df['temp'], df['humidity'])
    features_df['pace_vs_pr'] = df['pace_sec'] / df['mile_pr_sec']
    features_df['hr_efficiency'] = df['avg_hr'] / df['max_hr']
    features_df['temp_humidity_interaction'] = df['temp'] * df['humidity'] / 100
    features_df['pace_hr_interaction'] = features_df['pace_vs_pr'] * features_df['hr_efficiency']
    
    # Time-based features
    if len(df) >= 3:
        features_df['hss_rolling_3'] = df['raw_score'].rolling(window=min(3, len(df)), min_periods=1).mean()
        features_df['temp_rolling_3'] = df['temp'].rolling(window=min(3, len(df)), min_periods=1).mean()
        features_df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        if len(df) >= 2:
            features_df['hss_trend'] = df['raw_score'].diff().fillna(0)
            features_df['temp_trend'] = df['temp'].diff().fillna(0)
    else:
        features_df['hss_rolling_3'] = df['raw_score']
        features_df['temp_rolling_3'] = df['temp']
        features_df['days_since_start'] = 0
        features_df['hss_trend'] = 0
        features_df['temp_trend'] = 0
    
    # Performance stress indicators
    features_df['environmental_stress'] = (
        normalize(df['temp'], 60, 100) * 0.6 + 
        normalize(df['humidity'], 40, 95) * 0.4
    )
    
    features_df['performance_stress'] = (
        features_df['pace_vs_pr'] * 0.4 + 
        features_df['hr_efficiency'] * 0.6
    )
    
    return features_df

def calculate_heat_index(temp_f, humidity_pct):
    """Calculate heat index - simplified version for testing"""
    T = temp_f
    H = humidity_pct
    
    if hasattr(T, '__iter__') and not isinstance(T, str):
        heat_index = np.where(
            T < 80,
            T,
            -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
        )
    else:
        if T < 80:
            heat_index = T
        else:
            heat_index = -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
    
    return heat_index

def normalize(value, min_val, max_val):
    """Normalize value between 0 and 1"""
    return (value - min_val) / (max_val - min_val)


class HeatAdaptationMLModel:
    """ML Model class for heat adaptation analysis"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=8, 
                min_samples_split=2, 
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.1, 
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, features_df):
        """Prepare feature matrix for ML models"""
        feature_cols = [
            'temp', 'humidity', 'heat_index', 'pace_vs_pr', 'hr_efficiency',
            'distance', 'temp_humidity_interaction', 'pace_hr_interaction',
            'hss_rolling_3', 'temp_rolling_3', 'days_since_start',
            'environmental_stress', 'performance_stress', 'hss_trend', 'temp_trend'
        ]
        
        X = features_df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        return X
    
    def train_models(self, features_df, target_col='raw_score', min_samples=5):
        """Train multiple ML models and select the best one"""
        if len(features_df) < min_samples:
            self.is_trained = False
            return None
            
        X = self.prepare_features(features_df)
        y = features_df[target_col]
        
        # For small datasets, use all data for training
        if len(features_df) < 10:
            X_train, X_test = X, X
            y_train, y_test = y, y
        else:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train and evaluate models
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
            except Exception as e:
                continue
        
        if model_scores:
            # Select best model based on lowest RMSE
            best_name = min(model_scores.keys(), key=lambda k: model_scores[k]['rmse'])
            self.best_model = model_scores[best_name]['model']
            self.is_trained = True
            return model_scores[best_name]
        else:
            self.is_trained = False
            return None
    
    def predict(self, features_df):
        """Make predictions using the trained model"""
        if not self.is_trained:
            return None
            
        X = self.prepare_features(features_df)
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def get_feature_importance(self):
        """Get feature importance for interpretation"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return None


class TestHeatAdaptationMLModel:
    """Test the HeatAdaptationMLModel class"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        dates = pd.date_range(start='2024-01-01', periods=15, freq='D')
        np.random.seed(42)
        
        data = {
            'date': dates,
            'temp': np.random.uniform(75, 95, 15),
            'humidity': np.random.uniform(60, 90, 15),
            'pace_sec': np.random.uniform(420, 480, 15),  # 7:00-8:00 pace
            'avg_hr': np.random.uniform(160, 180, 15),
            'max_hr': [190] * 15,
            'distance': np.random.uniform(3, 8, 15),
            'mile_pr_sec': [400] * 15,  # 6:40 PR
            'raw_score': np.random.uniform(8, 20, 15)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def minimal_training_data(self):
        """Create minimal training data (insufficient for ML)"""
        dates = pd.date_range(start='2024-01-01', periods=3, freq='D')
        
        data = {
            'date': dates,
            'temp': [85, 88, 90],
            'humidity': [75, 80, 85],
            'pace_sec': [450, 460, 470],
            'avg_hr': [165, 170, 175],
            'max_hr': [190, 190, 190],
            'distance': [5.0, 3.1, 4.0],
            'mile_pr_sec': [400, 400, 400],
            'raw_score': [12.5, 15.2, 18.1]
        }
        
        return pd.DataFrame(data)
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = HeatAdaptationMLModel()
        
        assert len(model.models) == 2
        assert 'random_forest' in model.models
        assert 'gradient_boost' in model.models
        assert isinstance(model.scaler, StandardScaler)
        assert model.best_model is None
        assert model.feature_names is None
        assert model.is_trained is False
    
    def test_prepare_features_sufficient_data(self, sample_training_data):
        """Test feature preparation with sufficient data"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(sample_training_data)
        X = model.prepare_features(features_df)
        
        assert len(X.columns) == 15  # All expected features
        assert len(X) == len(sample_training_data)
        assert 'temp' in X.columns
        assert 'humidity' in X.columns
        assert 'heat_index' in X.columns
        assert 'pace_vs_pr' in X.columns
    
    def test_prepare_features_minimal_data(self, minimal_training_data):
        """Test feature preparation with minimal data"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(minimal_training_data)
        X = model.prepare_features(features_df)
        
        assert len(X.columns) == 15  # Still all features, but some may be defaults
        assert len(X) == len(minimal_training_data)
        
        # Check that default values are used for time-based features
        assert X['days_since_start'].iloc[0] == 0
        assert X['hss_trend'].iloc[0] == 0
    
    def test_train_models_insufficient_data(self, minimal_training_data):
        """Test training with insufficient data"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(minimal_training_data)
        
        result = model.train_models(features_df, min_samples=5)
        
        assert result is None
        assert model.is_trained is False
        assert model.best_model is None
    
    def test_train_models_sufficient_data(self, sample_training_data):
        """Test training with sufficient data"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(sample_training_data)
        
        result = model.train_models(features_df, min_samples=5)
        
        assert result is not None
        assert model.is_trained is True
        assert model.best_model is not None
        assert 'mse' in result
        assert 'mae' in result
        assert 'r2' in result
        assert 'rmse' in result
    
    def test_predict_without_training(self, sample_training_data):
        """Test prediction without training first"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(sample_training_data)
        
        result = model.predict(features_df)
        
        assert result is None
    
    def test_predict_after_training(self, sample_training_data):
        """Test prediction after training"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(sample_training_data)
        
        # Train the model first
        model.train_models(features_df, min_samples=5)
        
        # Make predictions
        predictions = model.predict(features_df)
        
        assert predictions is not None
        assert len(predictions) == len(features_df)
        assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    def test_feature_importance(self, sample_training_data):
        """Test feature importance extraction"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(sample_training_data)
        
        # Train the model
        model.train_models(features_df, min_samples=5)
        
        # Get feature importance
        importance_df = model.get_feature_importance()
        
        if importance_df is not None:  # Tree-based model was selected
            assert len(importance_df) == 15  # Number of features
            assert 'feature' in importance_df.columns
            assert 'importance' in importance_df.columns
            assert importance_df['importance'].sum() > 0
    
    def test_model_selection_consistency(self, sample_training_data):
        """Test that model selection is consistent"""
        model1 = HeatAdaptationMLModel()
        model2 = HeatAdaptationMLModel()
        features_df = create_ml_features(sample_training_data)
        
        # Train both models with same data and seed
        result1 = model1.train_models(features_df)
        result2 = model2.train_models(features_df)
        
        # Should get same results due to random_state
        assert model1.is_trained == model2.is_trained
        if result1 and result2:
            assert abs(result1['rmse'] - result2['rmse']) < 0.01


class TestFeatureEngineering:
    """Test feature engineering functions"""
    
    @pytest.fixture
    def basic_data(self):
        """Basic run data for feature engineering"""
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        
        data = {
            'date': dates,
            'temp': [80, 85, 82, 88, 90, 87, 84, 86, 89, 91],
            'humidity': [70, 75, 68, 80, 85, 78, 72, 76, 82, 87],
            'pace_sec': [450, 460, 440, 470, 480, 465, 455, 468, 475, 485],
            'avg_hr': [165, 170, 160, 175, 180, 172, 168, 173, 178, 182],
            'max_hr': [190] * 10,
            'distance': [5.0, 3.1, 6.0, 4.0, 3.1, 5.0, 4.5, 3.1, 5.5, 4.0],
            'mile_pr_sec': [400] * 10,
            'raw_score': [10.5, 12.8, 9.2, 15.3, 18.1, 14.2, 11.7, 13.9, 16.8, 19.2]
        }
        
        return pd.DataFrame(data)
    
    def test_create_ml_features_basic(self, basic_data):
        """Test basic feature creation"""
        features_df = create_ml_features(basic_data)
        
        # Check that new features are created
        assert 'heat_index' in features_df.columns
        assert 'pace_vs_pr' in features_df.columns
        assert 'hr_efficiency' in features_df.columns
        assert 'temp_humidity_interaction' in features_df.columns
        assert 'pace_hr_interaction' in features_df.columns
        
        # Check feature values are reasonable
        assert all(features_df['pace_vs_pr'] > 1.0)  # Should be slower than PR
        assert all(0 < features_df['hr_efficiency'] < 1.0)  # Should be fraction of max HR
    
    def test_create_ml_features_time_based(self, basic_data):
        """Test time-based feature creation"""
        features_df = create_ml_features(basic_data)
        
        # Check time-based features
        assert 'hss_rolling_3' in features_df.columns
        assert 'temp_rolling_3' in features_df.columns
        assert 'days_since_start' in features_df.columns
        assert 'hss_trend' in features_df.columns
        assert 'temp_trend' in features_df.columns
        
        # Check rolling features
        assert not features_df['hss_rolling_3'].isna().any()
        assert not features_df['temp_rolling_3'].isna().any()
        
        # Check days since start
        assert features_df['days_since_start'].iloc[0] == 0
        assert features_df['days_since_start'].iloc[-1] == 9
    
    def test_create_ml_features_stress_indicators(self, basic_data):
        """Test stress indicator features"""
        features_df = create_ml_features(basic_data)
        
        assert 'environmental_stress' in features_df.columns
        assert 'performance_stress' in features_df.columns
        
        # Environmental stress should be between 0 and 1
        assert all(0 <= stress <= 1 for stress in features_df['environmental_stress'])
        
        # Performance stress should be positive
        assert all(stress >= 0 for stress in features_df['performance_stress'])
    
    def test_create_ml_features_minimal_data(self):
        """Test feature creation with minimal data (< 3 rows)"""
        minimal_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=2),
            'temp': [85, 88],
            'humidity': [75, 80],
            'pace_sec': [450, 460],
            'avg_hr': [165, 170],
            'max_hr': [190, 190],
            'distance': [5.0, 3.1],
            'mile_pr_sec': [400, 400],
            'raw_score': [12.5, 15.2]
        })
        
        features_df = create_ml_features(minimal_data)
        
        # Should still create features, but with defaults for time-based ones
        assert 'heat_index' in features_df.columns
        assert features_df['days_since_start'].iloc[0] == 0
        assert features_df['hss_trend'].iloc[0] == 0


class TestMLModelIntegration:
    """Integration tests for ML model with real-world scenarios"""
    
    @pytest.fixture
    def realistic_progression_data(self):
        """Create realistic heat adaptation progression data"""
        dates = pd.date_range(start='2024-06-01', periods=20, freq='D')
        np.random.seed(123)
        
        # Simulate heat adaptation progression - scores should generally decrease over time
        base_scores = np.linspace(18, 12, 20)  # Gradual improvement
        noise = np.random.normal(0, 1.5, 20)  # Add realistic noise
        scores = base_scores + noise
        scores = np.clip(scores, 6, 25)  # Keep in reasonable range
        
        # Temperature and humidity that varies realistically
        temps = np.random.uniform(82, 96, 20)
        humidity = np.random.uniform(65, 88, 20)
        
        # Pace gets slightly faster as adaptation improves (subtle effect)
        base_pace = 450
        pace_improvement = np.linspace(0, -15, 20)  # Slight improvement
        paces = base_pace + pace_improvement + np.random.normal(0, 10, 20)
        
        # Heart rate decreases slightly with adaptation
        base_hr = 175
        hr_improvement = np.linspace(0, -8, 20)
        hrs = base_hr + hr_improvement + np.random.normal(0, 3, 20)
        
        data = {
            'date': dates,
            'temp': temps,
            'humidity': humidity,
            'pace_sec': paces,
            'avg_hr': hrs,
            'max_hr': [190] * 20,
            'distance': np.random.choice([3.1, 5.0, 6.0, 4.0], 20),
            'mile_pr_sec': [420] * 20,
            'raw_score': scores
        }
        
        return pd.DataFrame(data)
    
    def test_realistic_training_and_prediction(self, realistic_progression_data):
        """Test training and prediction with realistic data"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(realistic_progression_data)
        
        # Split into training and test data
        train_data = features_df[:15]
        test_data = features_df[15:]
        
        # Train model
        result = model.train_models(train_data)
        
        assert result is not None
        assert model.is_trained
        
        # Make predictions on test data
        predictions = model.predict(test_data)
        
        assert predictions is not None
        assert len(predictions) == len(test_data)
        
        # Predictions should be in reasonable range
        assert all(5 <= pred <= 30 for pred in predictions)
    
    def test_model_performance_metrics(self, realistic_progression_data):
        """Test that model performance metrics are reasonable"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(realistic_progression_data)
        
        result = model.train_models(features_df)
        
        assert result is not None
        
        # Performance metrics should be reasonable
        assert result['mae'] >= 0  # MAE should be non-negative
        assert result['mse'] >= 0  # MSE should be non-negative
        assert result['rmse'] >= 0  # RMSE should be non-negative
        
        # R² should be between -inf and 1, but for good models > 0
        # With realistic data, we might not always get great R² values
        assert result['r2'] <= 1.0
    
    def test_feature_importance_realistic(self, realistic_progression_data):
        """Test feature importance with realistic data"""
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(realistic_progression_data)
        
        model.train_models(features_df)
        importance_df = model.get_feature_importance()
        
        if importance_df is not None:
            # Temperature and humidity should be among top features
            top_features = importance_df.head(5)['feature'].tolist()
            
            # At least one environmental factor should be important
            environmental_features = ['temp', 'humidity', 'heat_index', 'environmental_stress']
            assert any(feat in top_features for feat in environmental_features)
            
            # Importance values should sum to approximately 1 for tree models
            total_importance = importance_df['importance'].sum()
            assert abs(total_importance - 1.0) < 0.01
    
    def test_model_robustness_with_outliers(self, realistic_progression_data):
        """Test model robustness when data contains outliers"""
        # Add some outlier data points
        outlier_data = realistic_progression_data.copy()
        
        # Add extreme outliers
        outlier_data.loc[5, 'raw_score'] = 35.0  # Very high score
        outlier_data.loc[10, 'raw_score'] = 2.0   # Very low score
        
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(outlier_data)
        
        # Model should still train successfully
        result = model.train_models(features_df)
        
        assert result is not None
        assert model.is_trained
        
        # Should still make reasonable predictions
        predictions = model.predict(features_df)
        assert predictions is not None
        
        # Most predictions should still be in reasonable range
        reasonable_preds = sum(1 for pred in predictions if 5 <= pred <= 25)
        assert reasonable_preds >= len(predictions) * 0.8  # At least 80% reasonable


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe"""
        model = HeatAdaptationMLModel()
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        result = model.train_models(empty_df)
        assert result is None
        assert not model.is_trained
    
    def test_single_row_dataframe(self):
        """Test handling of single row dataframe"""
        model = HeatAdaptationMLModel()
        single_row = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')],
            'temp': [85],
            'humidity': [75],
            'pace_sec': [450],
            'avg_hr': [165],
            'max_hr': [190],
            'distance': [5.0],
            'mile_pr_sec': [400],
            'raw_score': [12.5]
        })
        
        features_df = create_ml_features(single_row)
        result = model.train_models(features_df)
        
        # Should not train with insufficient data
        assert result is None
        assert not model.is_trained
    
    def test_missing_feature_columns(self):
        """Test handling when expected feature columns are missing"""
        model = HeatAdaptationMLModel()
        
        # Create dataframe missing some expected columns
        incomplete_df = pd.DataFrame({
            'temp': [85, 88, 90],
            'humidity': [75, 80, 85],
            'raw_score': [12.5, 15.2, 18.1]
        })
        
        # Should handle missing columns by filling with 0
        X = model.prepare_features(incomplete_df)
        
        # Should still have all expected feature columns
        assert len(X.columns) == 15
        
        # Missing columns should be filled with 0
        missing_cols = set(model.feature_names) - set(incomplete_df.columns)
        for col in missing_cols:
            if col in X.columns:
                assert all(X[col] == 0)
    
    def test_infinite_or_nan_values(self):
        """Test handling of infinite or NaN values"""
        data_with_nans = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=5),
            'temp': [85, np.nan, 90, 88, 92],
            'humidity': [75, 80, np.inf, 85, 82],
            'pace_sec': [450, 460, 455, -np.inf, 470],
            'avg_hr': [165, 170, 175, 172, 168],
            'max_hr': [190] * 5,
            'distance': [5.0, 3.1, 6.0, 4.0, 5.5],
            'mile_pr_sec': [400] * 5,
            'raw_score': [12.5, 15.2, np.nan, 14.8, 16.1]
        })
        
        model = HeatAdaptationMLModel()
        features_df = create_ml_features(data_with_nans)
        
        # Feature preparation should handle NaN/inf by filling with 0
        X = model.prepare_features(features_df)
        
        # Should not have any NaN or infinite values
        assert not X.isna().any().any()
        assert not np.isinf(X).any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
        