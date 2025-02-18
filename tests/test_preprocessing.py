import logging
from pathlib import Path
import sys
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from preprocess.preprocess import TemperaturePreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_time_gaps(data):
    """Analyze gaps in timestamp data"""
    # Sort data by time
    data_sorted = data.sort_values('Time')
    
    # Calculate time differences
    time_diffs = data_sorted['Time'].diff()
    
    # Find large gaps (more than 1 day)
    large_gaps = time_diffs[time_diffs > pd.Timedelta(days=1)]
    
    if not large_gaps.empty:
        logger.warning(f"\nFound {len(large_gaps)} large time gaps")
        logger.warning("\nTop 5 largest gaps:")
        for idx, gap in large_gaps.nlargest(5).items():
            start_time = data_sorted.loc[idx-1, 'Time']
            end_time = data_sorted.loc[idx, 'Time']
            logger.warning(f"Gap of {gap}: from {start_time} to {end_time}")
    
    # Time range statistics
    logger.info("\nTime Range Statistics:")
    logger.info(f"Start date: {data_sorted['Time'].min()}")
    logger.info(f"End date: {data_sorted['Time'].max()}")
    logger.info(f"Total time span: {data_sorted['Time'].max() - data_sorted['Time'].min()}")
    logger.info(f"Average time between readings: {time_diffs.mean()}")
    logger.info(f"Median time between readings: {time_diffs.median()}")

def test_preprocessing():
    """Test the preprocessing pipeline with focus on timestamp handling"""
    try:
        # Initialize preprocessor
        preprocessor = TemperaturePreprocessor()
        
        # Load and process data
        data_path = project_root / 'data' / 'data.csv'
        logger.info(f"Loading data from: {data_path}")
        
        # Process the data
        preprocessor.load_data(str(data_path))
        
        # Verify the data is properly sorted and timestamps are valid
        assert preprocessor.data is not None, "Data not loaded"
        assert preprocessor.data['Time'].is_monotonic_increasing, "Times not properly sorted"
        assert not preprocessor.data['Time'].isna().any(), "Found invalid timestamps"
        
        logger.info("Preprocessing test completed successfully")
            
    except Exception as e:
        logger.error(f"Error during preprocessing test: {str(e)}")
        raise

if __name__ == "__main__":
    test_preprocessing() 