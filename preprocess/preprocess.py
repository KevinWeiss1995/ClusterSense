import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import re

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.path_utils import get_project_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HPCPreprocessor:
    def __init__(self):
        self.data = None
        self.required_columns = ['timestamp', 'node', 'component', 'state', 'value']
        
    def load_data(self, data_path):
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.data)} records")
            
            missing_cols = set(self.required_columns) - set(self.data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], unit='s')
            
            logger.info("\nInitial Dataset Overview:")
            logger.info(f"Time range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
            logger.info(f"Components distribution: {self.data['component'].value_counts().to_dict()}")
            logger.info(f"States distribution: {self.data['state'].value_counts().to_dict()}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def process(self):
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        try:    
            temp_data = self.data[self.data['state'] == 'temperature'].copy()
            
            temp_data['temp_value'] = temp_data['value'].str.split().str[1].astype(float)
            
            self.data['is_failure'] = self.data['state'].isin(['error', 'abort'])
            
            temp_stats = temp_data.groupby('node').agg({
                'temp_value': ['mean', 'std', 'max', 'min', 'count'],
                'timestamp': ['first', 'last']
            }).reset_index()
            
            temp_stats.columns = ['node', 'avg_temp', 'temp_std', 'max_temp', 
                                'min_temp', 'temp_readings', 'first_reading', 'last_reading']
            
            self.data = self.data.merge(temp_stats, on='node', how='left')
            
            self.data['hour'] = self.data['timestamp'].dt.hour
            self.data['time_since_last'] = self.data.groupby(['node', 'component'])['timestamp'].diff()
            
            logger.info("\nProcessing Results:")
            logger.info(f"Temperature readings processed: {len(temp_data)}")
            logger.info(f"Unique nodes with temperature data: {temp_data['node'].nunique()}")
            logger.info(f"Failure events found: {self.data['is_failure'].sum()}")
            
            logger.info("\nTemperature Statistics:")
            logger.info(f"Mean temperature: {temp_data['temp_value'].mean():.2f}")
            logger.info(f"Max temperature: {temp_data['temp_value'].max():.2f}")
            logger.info(f"Min temperature: {temp_data['temp_value'].min():.2f}")
            logger.info(f"Std temperature: {temp_data['temp_value'].std():.2f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def save_processed_data(self, output_path=None):
        if self.data is None:
            raise ValueError("No data processed. Call process first.")
            
        try:
            if output_path is None:
                output_path = get_project_root() / "data" / "processed_HPC.csv"
            
            self.data.to_csv(output_path, index=False)
            logger.info(f"Saved processed data to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

if __name__ == "__main__":
    preprocessor = HPCPreprocessor()
    data_path = get_project_root() / "data" / "HPC.csv"
    preprocessor.load_data(data_path).process().save_processed_data()
    