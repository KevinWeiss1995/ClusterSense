import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TemperaturePreprocessor:
    def __init__(self):
        self.data = None
        self.temp_data = None
        self.node_temps = None
        self.component_temps = None
        self.required_columns = [
            'LineId', 'LogId', 'Node', 'Component', 'State',
            'Time', 'Flag', 'Content', 'EventId', 'EventTemplate'
        ]
        self.quality_metrics = {}
        self.summary_stats = {}

    def validate_input_data(self, df):
        """Validate input dataframe structure and content"""
     
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
     
        if df.empty:
            raise ValueError("Input dataframe is empty")
            
        
        critical_cols = ['Time', 'Node', 'Content', 'EventTemplate']
        null_counts = df[critical_cols].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found in critical columns: {null_counts[null_counts > 0]}")
            
        return True

    def load_data(self, data_path):
        """Load and initialize the data with validation"""
        try:
           
            self.data = pd.read_csv(data_path, names=self.required_columns, skiprows=1)
            self.validate_input_data(self.data)
            
            logger.info(f"Sample of raw Time values: {self.data['Time'].head()}")
            logger.info(f"Time column dtype: {self.data['Time'].dtype}")
            
            try:
                self.data['Time'] = pd.to_numeric(self.data['Time'], errors='coerce')
                
                self.data['Time'] = pd.to_datetime(self.data['Time'], unit='s')
               
                nat_count = self.data['Time'].isna().sum()
                if nat_count > 0:
                    logger.warning(f"Found {nat_count} invalid timestamps")
                
                    self.data = self.data.dropna(subset=['Time'])
                
                self.data = self.data.sort_values('Time').reset_index(drop=True)
                
                # Analyze time distribution
                time_diffs = self.data['Time'].diff()
                large_gaps = time_diffs[time_diffs > pd.Timedelta(days=1)]
                
                if not large_gaps.empty:
                    logger.warning(f"Found {len(large_gaps)} gaps larger than 1 day")
                    logger.warning("Largest gaps:")
                    for idx in large_gaps.nlargest(5).index:
                        gap = time_diffs.loc[idx]
                        start_time = self.data.loc[idx-1, 'Time']
                        end_time = self.data.loc[idx, 'Time']
                        if end_time > start_time:  # Ensure gap is valid
                            logger.warning(f"Gap of {gap}: from {start_time} to {end_time}")
                            logger.warning(f"Records around gap:")
                            logger.warning(f"Before: {self.data.loc[idx-1].to_dict()}")
                            logger.warning(f"After: {self.data.loc[idx].to_dict()}")
                
                # Log time range statistics
                logger.info(f"Time range: from {self.data['Time'].min()} to {self.data['Time'].max()}")
                logger.info(f"Median time between readings: {time_diffs.median()}")
                logger.info(f"Average time between readings: {time_diffs.mean()}")
                
            except (ValueError, TypeError) as e:
                logger.error(f"Original timestamp values causing error: {self.data['Time'].head()}")
                raise ValueError(f"Error converting timestamps: {str(e)}")
            
            return self
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("Data file is empty")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def extract_temperature_events(self):
        """Extract temperature-related events with validation"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        # Filter temperature events
        temp_mask = (
            self.data['EventTemplate'].str.contains('Temperature', na=False) |
            self.data['EventTemplate'].str.contains('ambient', na=False)
        )
        self.temp_data = self.data[temp_mask].copy()
        
        if self.temp_data.empty:
            raise ValueError("No temperature events found in data")
            
        # Separate and validate node/component temperatures
        self.node_temps = self.temp_data[
            self.temp_data['EventTemplate'].str.contains('ambient', na=False)
        ].copy()
        
        self.component_temps = self.temp_data[
            self.temp_data['EventTemplate'].str.contains('Temperature', na=False)
        ].copy()
        
        if self.node_temps.empty:
            raise ValueError("No ambient temperature readings found")
            
        return self

    def extract_temperature_values(self):
        """Extract numerical temperature values with validation"""
        if self.node_temps is None or self.component_temps is None:
            raise ValueError("Temperature events not extracted. Call extract_temperature_events first.")
            
        # Extract and validate ambient temperatures
        self.node_temps['temperature'] = pd.to_numeric(
            self.node_temps['Content'].str.extract(r'ambient=(\d+)')[0],
            errors='coerce'
        )
        
        # Validate temperature values
        invalid_temps = self.node_temps['temperature'].isnull().sum()
        if invalid_temps > 0:
            print(f"Warning: {invalid_temps} invalid temperature readings found and will be excluded")
        
        # Remove invalid temperatures
        self.node_temps = self.node_temps.dropna(subset=['temperature'])
        
        # Validate temperature range (e.g., reasonable range for computer hardware)
        temp_range = (0, 100)  # Celsius
        invalid_range = ~self.node_temps['temperature'].between(*temp_range)
        if invalid_range.any():
            print(f"Warning: {invalid_range.sum()} temperature readings outside expected range {temp_range}")
        
        return self

    def create_time_features(self):
        """Create time-based features"""
        for df in [self.node_temps, self.component_temps]:
            # Sort by time
            df.sort_values('Time', inplace=True)
            
            # Group by node/component
            grouped = df.groupby('Node')
            
            # Add time-based features
            df['time_since_last'] = grouped['Time'].diff().dt.total_seconds()
            df['hour_of_day'] = df['Time'].dt.hour
            df['day_of_week'] = df['Time'].dt.dayofweek
            
        return self

    def create_temperature_features(self):
        """Create temperature-related features"""
        # For node temperatures
        self.node_temps = self.node_temps.sort_values(['Node', 'Time'])
        grouped_nodes = self.node_temps.groupby('Node')
        
        # Calculate features and assign them back to the DataFrame
        self.node_temps['temp_change'] = grouped_nodes['temperature'].diff()
        
        # Calculate rolling statistics
        rolling_means = grouped_nodes['temperature'].rolling(
            window=3, 
            min_periods=1
        ).mean()
        self.node_temps['rolling_mean_temp'] = rolling_means.reset_index(level=0, drop=True)
        
        rolling_stds = grouped_nodes['temperature'].rolling(
            window=3, 
            min_periods=1
        ).std()
        self.node_temps['rolling_std_temp'] = rolling_stds.reset_index(level=0, drop=True)
        
        # For component temperatures (if numerical values available)
        if 'temperature' in self.component_temps.columns:
            self.component_temps = self.component_temps.sort_values(['Node', 'Time'])
            grouped_components = self.component_temps.groupby('Node')
            
            self.component_temps['temp_change'] = grouped_components['temperature'].diff()
            
            rolling_means = grouped_components['temperature'].rolling(
                window=3, 
                min_periods=1
            ).mean()
            self.component_temps['rolling_mean_temp'] = rolling_means.reset_index(level=0, drop=True)
            
            rolling_stds = grouped_components['temperature'].rolling(
                window=3, 
                min_periods=1
            ).std()
            self.component_temps['rolling_std_temp'] = rolling_stds.reset_index(level=0, drop=True)
        
        return self

    def label_anomalies(self):
        """Label temperature anomalies based on events"""
        # Label node temperature anomalies (can be customized based on domain knowledge)
        self.node_temps['is_anomaly'] = (
            self.node_temps['temperature'].rolling(window=5, min_periods=1).std() > 
            self.node_temps['temperature'].std() * 2
        )
        
        # Label component temperature anomalies based on warning/critical events
        self.component_temps['is_anomaly'] = self.component_temps['EventTemplate'].str.contains(
            'warning|critical', case=False
        )
        
        return self

    def compute_quality_metrics(self):
        """Compute comprehensive data quality metrics"""
        if self.node_temps is None or self.component_temps is None:
            raise ValueError("Data not processed. Call process first.")

        self.quality_metrics = {
            'node_temperatures': {
                'total_readings': len(self.node_temps),
                'unique_nodes': self.node_temps['Node'].nunique(),
                'missing_values': self.node_temps.isnull().sum().to_dict(),
                'duplicate_readings': self.node_temps.duplicated().sum(),
                'reading_frequency': {
                    'mean_time_between_readings': self.node_temps.groupby('Node')['time_since_last'].mean().mean(),
                    'max_time_gap': self.node_temps.groupby('Node')['time_since_last'].max().max()
                },
                'value_ranges': {
                    'min_temp': self.node_temps['temperature'].min(),
                    'max_temp': self.node_temps['temperature'].max(),
                    'outliers_count': len(self.detect_temperature_outliers(self.node_temps))
                },
                'data_coverage': {
                    'start_time': self.node_temps['Time'].min(),
                    'end_time': self.node_temps['Time'].max(),
                    'total_duration_hours': (self.node_temps['Time'].max() - 
                                          self.node_temps['Time'].min()).total_seconds() / 3600
                }
            },
            'component_temperatures': {
                'total_events': len(self.component_temps),
                'unique_components': self.component_temps['Node'].nunique(),
                'event_distribution': self.component_temps['EventTemplate'].value_counts().to_dict(),
                'warning_events': len(self.component_temps[
                    self.component_temps['EventTemplate'].str.contains('warning', case=False)
                ]),
                'critical_events': len(self.component_temps[
                    self.component_temps['EventTemplate'].str.contains('critical', case=False)
                ])
            }
        }
        
        return self

    def compute_summary_statistics(self):
        """Compute detailed summary statistics for temperature data"""
        if self.node_temps is None:
            raise ValueError("Data not processed. Call process first.")

        # Node temperature statistics
        node_temp_stats = self.node_temps.groupby('Node').agg({
            'temperature': ['count', 'mean', 'std', 'min', 'max'],
            'temp_change': ['mean', 'std'],
            'time_since_last': ['mean', 'max'],
            'is_anomaly': 'sum'
        })

        # Time-based statistics
        hourly_stats = self.node_temps.groupby('hour_of_day').agg({
            'temperature': ['mean', 'std'],
            'is_anomaly': 'sum'
        })

        daily_stats = self.node_temps.groupby('day_of_week').agg({
            'temperature': ['mean', 'std'],
            'is_anomaly': 'sum'
        })

        self.summary_stats = {
            'node_temperature_stats': node_temp_stats,
            'hourly_patterns': hourly_stats,
            'daily_patterns': daily_stats,
            'overall_stats': {
                'temperature': {
                    'mean': self.node_temps['temperature'].mean(),
                    'std': self.node_temps['temperature'].std(),
                    'skew': self.node_temps['temperature'].skew(),
                    'kurtosis': self.node_temps['temperature'].kurtosis(),
                    'quantiles': self.node_temps['temperature'].quantile([0.25, 0.5, 0.75]).to_dict()
                },
                'anomaly_rate': self.node_temps['is_anomaly'].mean(),
                'total_anomalies': self.node_temps['is_anomaly'].sum()
            }
        }
        
        return self

    def detect_temperature_outliers(self, df, threshold=3):
        """Detect statistical outliers in temperature readings"""
        temp_mean = df['temperature'].mean()
        temp_std = df['temperature'].std()
        outliers = df[abs(df['temperature'] - temp_mean) > threshold * temp_std]
        return outliers

    def get_data_quality_report(self):
        """Generate a comprehensive data quality report"""
        if not self.quality_metrics:
            self.compute_quality_metrics()
            
        report = {
            'data_completeness': {
                'missing_data_percentage': (self.quality_metrics['node_temperatures']['missing_values']['temperature'] / 
                                         self.quality_metrics['node_temperatures']['total_readings']) * 100,
                'coverage_hours': self.quality_metrics['node_temperatures']['data_coverage']['total_duration_hours']
            },
            'data_validity': {
                'temperature_range_validity': (
                    f"Min: {self.quality_metrics['node_temperatures']['value_ranges']['min_temp']:.2f}°C, "
                    f"Max: {self.quality_metrics['node_temperatures']['value_ranges']['max_temp']:.2f}°C"
                ),
                'outliers_percentage': (self.quality_metrics['node_temperatures']['value_ranges']['outliers_count'] / 
                                      self.quality_metrics['node_temperatures']['total_readings']) * 100
            },
            'data_consistency': {
                'duplicate_readings_percentage': (self.quality_metrics['node_temperatures']['duplicate_readings'] / 
                                               self.quality_metrics['node_temperatures']['total_readings']) * 100,
                'average_reading_interval': self.quality_metrics['node_temperatures']['reading_frequency']['mean_time_between_readings']
            },
            'anomaly_statistics': {
                'warning_event_rate': (self.quality_metrics['component_temperatures']['warning_events'] / 
                                     self.quality_metrics['component_temperatures']['total_events']) * 100,
                'critical_event_rate': (self.quality_metrics['component_temperatures']['critical_events'] / 
                                      self.quality_metrics['component_temperatures']['total_events']) * 100
            }
        }
        
        return report

    def process(self, data_path):
        """Run the full preprocessing pipeline with quality checks"""
        try:
            return (self
                    .load_data(data_path)
                    .extract_temperature_events()
                    .extract_temperature_values()
                    .create_time_features()
                    .create_temperature_features()
                    .label_anomalies()
                    .compute_quality_metrics()
                    .compute_summary_statistics())
        except Exception as e:
            raise Exception(f"Error in preprocessing pipeline: {str(e)}")

    def get_processed_data(self):
        """Return the processed datasets with validation"""
        if self.node_temps is None or self.component_temps is None:
            raise ValueError("Data not processed. Call process first.")
            
        return {
            'node_temperatures': self.node_temps,
            'component_temperatures': self.component_temps
        }
    