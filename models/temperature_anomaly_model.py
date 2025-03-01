import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib
import mlflow
import mlflow.sklearn
import json
from datetime import datetime
import os
from typing import Dict, Any
from pathlib import Path

from preprocess.preprocess import TemperaturePreprocessor
from utils.path_utils import get_project_root, get_results_dir

class TemperatureAnomalyDetector:
    def __init__(self, contamination=0.1):
        self.preprocessor = TemperaturePreprocessor()
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.feature_columns = [
            'temperature', 
            'temp_change',
            'rolling_mean_temp',
            'rolling_std_temp',
            'time_since_last'
        ]

    def prepare_features(self, data):
        """Prepare feature matrix from preprocessed data"""
       
        features = data[self.feature_columns].copy()
        
        features = features.ffill().bfill()
        
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled

    def fit(self, data_path: str):
        """Train the model and evaluate performance"""
     
        processed_data = self.preprocessor.process(data_path).get_processed_data()
        node_temps = processed_data['node_temperatures']
        
        X = self.prepare_features(node_temps)
        
        self.model.fit(X)
        
        predictions = self.model.predict(X)
        scores = self.model.score_samples(X)
        
        metrics = self.evaluate_predictions(predictions, scores, node_temps)
    
        print("\nTemperature Anomaly Detection Report")
        print("===================================")
        print("\nData Summary:")
        print(f"Total temperature readings analyzed: {metrics['summary']['total_readings']:,}")
        print(f"Normal temperature readings: {metrics['summary']['normal_readings']:,}")
        print(f"Anomalous temperature readings: {metrics['summary']['anomalous_readings']:,}")
        print(f"Percentage of readings flagged as anomalous: {metrics['summary']['anomaly_percentage']:.1f}%")
        
        print("\nTemperature Ranges:")
        print("\nNormal Temperatures:")
        print(f"  Range: {metrics['temperature_ranges']['normal']['min']:.1f}°C to {metrics['temperature_ranges']['normal']['max']:.1f}°C")
        print(f"  Average: {metrics['temperature_ranges']['normal']['mean']:.1f}°C ± {metrics['temperature_ranges']['normal']['std']:.1f}°C")
        
        if metrics['temperature_ranges']['anomalous']['mean'] is not None:
            print("\nAnomalous Temperatures:")
            print(f"  Range: {metrics['temperature_ranges']['anomalous']['min']:.1f}°C to {metrics['temperature_ranges']['anomalous']['max']:.1f}°C")
            print(f"  Average: {metrics['temperature_ranges']['anomalous']['mean']:.1f}°C ± {metrics['temperature_ranges']['anomalous']['std']:.1f}°C")
        
        self.metrics = metrics
        
        return self

    def predict(self, data_path=None, node_temps=None):
        """
        Predict anomalies in temperature data
        
        Args:
            data_path (str, optional): Path to raw log data
            node_temps (pd.DataFrame, optional): Preprocessed temperature data
            
        Returns:
            pd.DataFrame: Original data with anomaly predictions
        """
        try:
            if data_path is not None:

                processed_data = self.preprocessor.process(data_path).get_processed_data()
                node_temps = processed_data['node_temperatures']
            
            if node_temps is None:
                raise ValueError("Either data_path or node_temps must be provided")
            
            X = self.prepare_features(node_temps)
            
            predictions = self.model.predict(X)
            scores = self.model.score_samples(X)
            
            results = node_temps.copy()
            results['anomaly'] = predictions == -1 
            results['anomaly_score'] = scores
            
            return results
            
        except Exception as e:
            raise Exception(f"Error during prediction: {str(e)}")

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluate model performance
        
        Args:
            true_labels (array-like): True anomaly labels
            predicted_labels (array-like): Predicted anomaly labels
            
        Returns:
            dict: Performance metrics
        """
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
        
        metrics = {
            'precision': precision_score(true_labels, predicted_labels),
            'recall': recall_score(true_labels, predicted_labels),
            'f1': f1_score(true_labels, predicted_labels),
            'confusion_matrix': confusion_matrix(true_labels, predicted_labels).tolist()
        }
        
        return metrics

    def save_model(self, model_name: str, metadata: Dict[str, Any] = None) -> str:
        """Save the model and its metadata"""
        try:
            results_dir = get_results_dir()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = results_dir / f"{model_name}_{timestamp}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            model_info = {
                'model_version': '1.0.0',
                'creation_timestamp': datetime.now().isoformat(),
                'feature_columns': self.feature_columns,
                'model_parameters': {k: convert_to_serializable(v) 
                                   for k, v in self.model.get_params().items()},
                'preprocessing_params': {
                    'scaler_mean_': convert_to_serializable(self.scaler.mean_) if hasattr(self.scaler, 'mean_') else None,
                    'scaler_scale_': convert_to_serializable(self.scaler.scale_) if hasattr(self.scaler, 'scale_') else None
                }
            }
            
            if metadata:
                metadata = {k: convert_to_serializable(v) for k, v in metadata.items()}
                model_info.update(metadata)
            
            model_path = model_dir / 'model.joblib'
            joblib.dump(self.model, model_path)
            
            scaler_path = model_dir / 'scaler.joblib'
            joblib.dump(self.scaler, scaler_path)
            
            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return str(model_dir)
                
        except Exception as e:
            raise Exception(f"Error saving model: {str(e)}")

    def load_model(self, path: str) -> 'TemperatureAnomalyDetector':
        """
        Load the model from production storage
        
        Args:
            path (str): Path to the model directory
            
        Returns:
            TemperatureAnomalyDetector: Loaded model instance
        """
        try:
            if not os.path.exists(path):
                raise ValueError(f"Model path does not exist: {path}")
            
            config_path = os.path.join(path, 'metadata.json')
            if not os.path.exists(config_path):
                raise ValueError(f"Model configuration not found at {config_path}")
                
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_info = config.get('model_info', {})
            if model_info.get('model_version', '0.0.0') < '1.0.0':
                raise ValueError("Incompatible model version")
            
            model_path = os.path.join(path, 'model.joblib')
            self.model = joblib.load(model_path)
            
            scaler_path = os.path.join(path, 'scaler.joblib')
            self.scaler = joblib.load(scaler_path)
            
            self.feature_columns = config['feature_columns']
            
            return self
            
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            dict: Model information including version, parameters, and metrics
        """
        return {
            'model_version': '1.0.0',
            'model_type': type(self.model).__name__,
            'feature_columns': self.feature_columns,
            'model_parameters': self.model.get_params(),
            'quality_metrics': self.metrics,
            'creation_timestamp': datetime.now().isoformat()
        }

    def evaluate_predictions(self, predictions: np.ndarray, scores: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate model predictions and generate human-readable performance metrics
        
        Args:
            predictions: Array of binary predictions (-1 for anomalies, 1 for normal)
            scores: Array of anomaly scores
            data: Original DataFrame with temperature readings
            
        Returns:
            dict: Dictionary of performance metrics
        """
        n_samples = len(predictions)
        anomaly_mask = predictions == -1
        n_anomalies = sum(anomaly_mask)
        
        # Get temperature statistics for anomalies
        anomaly_temps = data.loc[anomaly_mask, 'temperature']
        normal_temps = data.loc[~anomaly_mask, 'temperature']
        
        metrics = {
            'summary': {
                'total_readings': n_samples,
                'normal_readings': n_samples - n_anomalies,
                'anomalous_readings': n_anomalies,
                'anomaly_percentage': (n_anomalies / n_samples) * 100
            },
            'temperature_ranges': {
                'normal': {
                    'min': float(normal_temps.min()),
                    'max': float(normal_temps.max()),
                    'mean': float(normal_temps.mean()),
                    'std': float(normal_temps.std())
                },
                'anomalous': {
                    'min': float(anomaly_temps.min()) if not anomaly_temps.empty else None,
                    'max': float(anomaly_temps.max()) if not anomaly_temps.empty else None,
                    'mean': float(anomaly_temps.mean()) if not anomaly_temps.empty else None,
                    'std': float(anomaly_temps.std()) if not anomaly_temps.empty else None
                }
            }
        }
        
        return metrics 