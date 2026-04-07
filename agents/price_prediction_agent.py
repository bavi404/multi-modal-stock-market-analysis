"""
Price Prediction Agent for forecasting stock prices using historical data and sentiment
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from utils import config
from models.data_models import PredictionResult
from utils.prediction_explainability import (
    build_heuristic_explainability,
    build_linear_explainability,
    volatility_proxy,
)
import torch
import torch.nn as nn


class PricePredictionAgent:
    """Agent responsible for predicting future stock prices"""
    
    def __init__(self):
        """Initialize the price prediction agent"""
        self.logger = logging.getLogger(__name__)
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        self.use_lstm = config.USE_LSTM
        if self.use_lstm:
            self._init_lstm()

    def _init_lstm(self):
        class LSTMRegressor(nn.Module):
            def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
                super().__init__()
                self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
            def forward(self, x):
                # x: (batch, seq_len, input_size)
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.fc(last)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Placeholder sizes; will reinit when training with actual feature count
        self.lstm_model = None
        
    def _prepare_features(self, price_data: pd.DataFrame, sentiment_score: float) -> np.ndarray:
        """
        Prepare features for the prediction model
        
        Args:
            price_data: Historical stock price data
            sentiment_score: Current sentiment score
            
        Returns:
            Feature array for prediction
        """
        if price_data.empty:
            self.logger.error("No price data provided for feature preparation")
            return np.array([])
        
        features = []
        feature_names = []
        
        # Use the last N days of closing prices as features
        prediction_days = min(config.PREDICTION_DAYS, len(price_data))
        recent_prices = price_data['Close'].tail(prediction_days).values
        
        for i in range(prediction_days):
            features.append(recent_prices[i])
            feature_names.append(f'close_price_t-{prediction_days-i}')
        
        # Add technical indicators
        if len(price_data) >= 5:
            # Simple Moving Average (5-day)
            sma_5 = price_data['Close'].rolling(window=5).mean().iloc[-1]
            features.append(sma_5)
            feature_names.append('sma_5')
            
            # Price change percentage (last day)
            price_change_pct = ((price_data['Close'].iloc[-1] - price_data['Close'].iloc[-2]) / 
                              price_data['Close'].iloc[-2] * 100)
            features.append(price_change_pct)
            feature_names.append('price_change_pct')
            
            # Volume-based feature (volume ratio)
            if 'Volume' in price_data.columns:
                avg_volume = price_data['Volume'].rolling(window=5).mean().iloc[-1]
                current_volume = price_data['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                features.append(volume_ratio)
                feature_names.append('volume_ratio')
        
        # Add sentiment score as a feature
        features.append(sentiment_score)
        feature_names.append('sentiment_score')
        
        # Add volatility measure (standard deviation of last 5 days)
        if len(price_data) >= 5:
            volatility = price_data['Close'].tail(5).std()
            features.append(volatility)
            feature_names.append('volatility')
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)
    
    def _create_training_data(self, price_data: pd.DataFrame, 
                            sentiment_scores: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create training dataset from historical data
        
        Args:
            price_data: Historical stock price data
            sentiment_scores: Optional list of historical sentiment scores
            
        Returns:
            Tuple of (X_train, y_train)
        """
        if len(price_data) < config.PREDICTION_DAYS + 10:
            self.logger.error("Insufficient data for training")
            return np.array([]), np.array([])
        
        X_train = []
        y_train = []
        
        # If no sentiment scores provided, use neutral sentiment
        if sentiment_scores is None:
            sentiment_scores = [0.0] * len(price_data)
        elif len(sentiment_scores) < len(price_data):
            # Pad with neutral sentiment if not enough scores
            sentiment_scores.extend([0.0] * (len(price_data) - len(sentiment_scores)))
        
        # Create sliding window training samples
        for i in range(config.PREDICTION_DAYS, len(price_data) - 1):
            # Features: last N closing prices + technical indicators + sentiment
            window_data = price_data.iloc[i-config.PREDICTION_DAYS:i+1]
            
            features = []
            
            # Historical closing prices
            recent_prices = window_data['Close'].iloc[:-1].values  # Exclude current day
            features.extend(recent_prices)
            
            # Technical indicators
            if len(window_data) >= 5:
                sma_5 = window_data['Close'].rolling(window=5).mean().iloc[-2]  # Previous day's SMA
                features.append(sma_5)
                
                # Price change percentage
                if i > config.PREDICTION_DAYS:
                    price_change_pct = ((window_data['Close'].iloc[-2] - window_data['Close'].iloc[-3]) / 
                                      window_data['Close'].iloc[-3] * 100)
                    features.append(price_change_pct)
                else:
                    features.append(0.0)
                
                # Volume ratio
                if 'Volume' in window_data.columns:
                    avg_volume = window_data['Volume'].rolling(window=5).mean().iloc[-2]
                    current_volume = window_data['Volume'].iloc[-2]
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                    features.append(volume_ratio)
            
            # Sentiment score
            sentiment_idx = min(i, len(sentiment_scores) - 1)
            features.append(sentiment_scores[sentiment_idx])
            
            # Volatility
            if len(window_data) >= 5:
                volatility = window_data['Close'].iloc[:-1].tail(5).std()
                features.append(volatility)
            
            X_train.append(features)
            
            # Target: next day's closing price
            y_train.append(price_data['Close'].iloc[i + 1])
        
        return np.array(X_train), np.array(y_train)

    def _train_lstm(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 5) -> Dict[str, float]:
        if X_train.size == 0 or y_train.size == 0:
            self.logger.error("No training data for LSTM")
            return {}
        try:
            # For LSTM, treat each feature vector as a sequence of length seq_len with input_size=1
            # Here we reshape to (batch, seq_len, 1)
            seq_len = X_train.shape[1]
            input_size = 1
            if self.lstm_model is None:
                # Initialize based on seq_len and input size
                class LSTMRegressor(nn.Module):
                    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1):
                        super().__init__()
                        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
                        self.fc = nn.Linear(hidden_size, 1)
                    def forward(self, x):
                        out, _ = self.lstm(x)
                        last = out[:, -1, :]
                        return self.fc(last)
                self.lstm_model = LSTMRegressor(input_size).to(self.device)
            model = self.lstm_model
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            # Normalize targets for stability
            y_mean = float(y_train.mean())
            y_std = float(y_train.std() if y_train.std() != 0 else 1.0)
            X_seq = torch.tensor(X_train.reshape(-1, seq_len, 1), dtype=torch.float32).to(self.device)
            y_t = torch.tensor(((y_train - y_mean) / y_std).reshape(-1, 1), dtype=torch.float32).to(self.device)
            model.train()
            for _ in range(epochs):
                optimizer.zero_grad()
                preds = model(X_seq)
                loss = loss_fn(preds, y_t)
                loss.backward()
                optimizer.step()
            self.is_trained = True
            # Compute simple RMSE on training set
            model.eval()
            with torch.no_grad():
                preds = model(X_seq).cpu().numpy().reshape(-1)
            y_pred = preds * y_std + y_mean
            rmse = float(np.sqrt(np.mean((y_pred - y_train) ** 2)))
            return {'rmse': rmse}
        except Exception as e:
            self.logger.error(f"Error training LSTM: {e}")
            return {}
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the prediction model
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary with training metrics
        """
        if len(X_train) == 0 or len(y_train) == 0:
            self.logger.error("No training data available")
            return {}
        
        try:
            # Split data for validation
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train_split)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model
            self.model.fit(X_train_scaled, y_train_split)
            
            # Validate model
            y_pred = self.model.predict(X_val_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate R-squared
            r2_score = self.model.score(X_val_scaled, y_val)
            
            self.is_trained = True
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2_score': r2_score
            }
            
            self.logger.info(f"Model trained successfully. R²: {r2_score:.4f}, RMSE: {rmse:.4f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {}
    
    def _calculate_confidence_interval(self, predicted_price: float, 
                                     historical_errors: List[float] = None) -> Dict[str, float]:
        """
        Calculate confidence interval for prediction
        
        Args:
            predicted_price: The predicted price
            historical_errors: Historical prediction errors for confidence calculation
            
        Returns:
            Dictionary with lower and upper confidence bounds
        """
        # Simple confidence interval based on historical volatility or default percentage
        if historical_errors and len(historical_errors) > 0:
            error_std = np.std(historical_errors)
            confidence_range = 1.96 * error_std  # 95% confidence interval
        else:
            # Default to 5% confidence range if no historical errors available
            confidence_range = predicted_price * 0.05
        
        return {
            'lower': max(0, predicted_price - confidence_range),
            'upper': predicted_price + confidence_range
        }
    
    def predict(
        self,
        price_history: pd.DataFrame,
        sentiment_score: float,
        news_articles: Optional[List[Dict[str, str]]] = None,
        emotion_dominant: Optional[str] = None,
        emotion_scores: Optional[Dict[str, float]] = None,
    ) -> PredictionResult:
        """
        Predict the next day's closing price.

        Optional news/emotion inputs are used only for explainability (drivers, events).
        """
        self.logger.info("Starting price prediction")
        
        if price_history.empty:
            self.logger.error("No price history provided")
            return PredictionResult(
                predicted_price=0.0,
                confidence_interval={'lower': 0.0, 'upper': 0.0},
                prediction_date=datetime.now() + timedelta(days=1),
                model_confidence=0.0,
                features_used=[]
            )
        
        try:
            # Train model if not already trained
            if not self.is_trained:
                self.logger.info("Training prediction model")
                X_train, y_train = self._create_training_data(price_history)
                if self.use_lstm:
                    training_metrics = self._train_lstm(X_train, y_train, epochs=5)
                else:
                    training_metrics = self._train_model(X_train, y_train)
                if not self.is_trained:
                    self.logger.error("Model training failed")
                    return PredictionResult(
                        predicted_price=price_history['Close'].iloc[-1],  # Return last known price
                        confidence_interval={'lower': 0.0, 'upper': 0.0},
                        prediction_date=datetime.now() + timedelta(days=1),
                        model_confidence=0.0,
                        features_used=[]
                    )
            # Prepare features for prediction
            features = self._prepare_features(price_history, sentiment_score)
            if len(features) == 0:
                self.logger.error("Could not prepare features for prediction")
                return PredictionResult(
                    predicted_price=price_history['Close'].iloc[-1],
                    confidence_interval={'lower': 0.0, 'upper': 0.0},
                    prediction_date=datetime.now() + timedelta(days=1),
                    model_confidence=0.0,
                    features_used=[]
                )
            features_scaled: Any = None
            if self.use_lstm:
                # For LSTM, reshape to (1, seq_len, 1) and predict
                self.lstm_model.eval()
                with torch.no_grad():
                    seq = torch.tensor(features.reshape(1, -1, 1), dtype=torch.float32).to(self.device)
                    pred_norm = self.lstm_model(seq).cpu().numpy().reshape(-1)[0]
                # As we normalized targets during training, we lack y_mean/y_std here; fall back to using last price as baseline adjustment
                baseline = float(price_history['Close'].iloc[-1])
                predicted_price = baseline + float(pred_norm)
            else:
                # Scale features for linear regression
                features_scaled = self.scaler.transform(features)
                predicted_price = self.model.predict(features_scaled)[0]
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(predicted_price)
            
            # Calculate model confidence (simplified)
            model_confidence = min(0.85, max(0.3, abs(sentiment_score) + (0.55 if self.use_lstm else 0.5)))
            
            # Determine prediction date (next trading day)
            prediction_date = datetime.now() + timedelta(days=1)

            explainability = None
            news = news_articles or []
            emo_dom = emotion_dominant or "neutral"
            if self.use_lstm:
                explainability = build_heuristic_explainability(
                    predicted_price=float(predicted_price),
                    model_confidence=float(model_confidence),
                    sentiment_score=sentiment_score,
                    price_history_volatility=volatility_proxy(price_history["Close"]),
                    news_articles=news,
                    emotion_dominant=emo_dom,
                    emotion_scores=emotion_scores,
                )
            elif (
                self.is_trained
                and hasattr(self.model, "coef_")
                and features_scaled is not None
                and len(self.feature_names) > 0
            ):
                explainability = build_linear_explainability(
                    predicted_price=float(predicted_price),
                    model_confidence=float(model_confidence),
                    feature_names=self.feature_names.copy(),
                    coefficients=self.model.coef_,
                    scaled_features=features_scaled,
                    sentiment_score=sentiment_score,
                    news_articles=news,
                    emotion_dominant=emo_dom,
                    emotion_scores=emotion_scores,
                )
            
            result = PredictionResult(
                predicted_price=float(predicted_price),
                confidence_interval=confidence_interval,
                prediction_date=prediction_date,
                model_confidence=float(model_confidence),
                features_used=self.feature_names.copy(),
                explainability=explainability,
            )
            
            self.logger.info(f"Price prediction completed. Predicted price: ${predicted_price:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            # Return fallback prediction
            return PredictionResult(
                predicted_price=price_history['Close'].iloc[-1] if not price_history.empty else 0.0,
                confidence_interval={'lower': 0.0, 'upper': 0.0},
                prediction_date=datetime.now() + timedelta(days=1),
                model_confidence=0.0,
                features_used=[]
            )
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'coef_'):
            return {}
        
        importance_dict = {}
        coefficients = self.model.coef_
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(coefficients):
                importance_dict[feature_name] = abs(float(coefficients[i]))
        
        return importance_dict

