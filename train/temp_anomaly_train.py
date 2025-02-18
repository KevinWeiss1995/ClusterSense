import argparse
import logging
from pathlib import Path
import json
from datetime import datetime
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.temperature_anomaly_model import TemperatureAnomalyDetector
from utils.path_utils import get_project_root

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train temperature anomaly detection model for HPC clusters'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/data.csv',
        help='Path to training data CSV file'
    )
    
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.1,
        help='Expected proportion of outliers in the data'
    )
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("Starting temperature anomaly model training")
    logger.info(f"Arguments: {args}")
    
    try:
        # Initialize model
        detector = TemperatureAnomalyDetector(contamination=args.contamination)
        
        # Resolve data path relative to project root
        data_path = project_root / args.data_path
        logger.info(f"Using data from: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        
        # Train model
        logger.info("Training model...")
        detector.fit(str(data_path))
        
        # Get quality report
        quality_report = detector.preprocessor.get_data_quality_report()
        logger.info("Data Quality Report:")
        logger.info(json.dumps(quality_report, indent=2))
        
        # Save model with metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'data_path': str(data_path),
            'contamination': args.contamination,
            'quality_metrics': quality_report
        }
        
        model_path = detector.save_model(
            model_name='temp_anomaly_detector',
            metadata=metadata
        )
        
        logger.info(f"Model successfully saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
