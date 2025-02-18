# HPC Temperature Anomaly Detection

Machine learning system for detecting temperature anomalies in HPC clusters.

## Features
- Automated temperature anomaly detection using machine learning
- Robust timestamp processing and gap detection
- Production-ready data preprocessing pipeline
- Comprehensive logging and error handling

## Installationbash

```

git clone https://github.com/KevinWeiss1995/ClusterSense.git
cd ClusterSense
pip install -r requirements.txt

```
## Usage

```
python train/temp_anomaly_train.py --data-path data/data.csv
```

## Project Structure

```
.
├── models/
│   └── temperature_anomaly_model.py
├── preprocess/
│   └── preprocess.py
├── utils/
│   └── path_utils.py
├── train/
│   └── temp_anomaly_train.py
└── tests/
    └── test_preprocessing.py
```



