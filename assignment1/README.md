# Assignment 1 - Advanced Data Analytics

## Overview

This project explores two major business challenges in advanced data analytics:

1. **Privacy-Preserving Analytics with Synthetic Data Generation**
2. **Mining NYC Taxi Trip Data using Big Data Technologies**

## Project Structure

```
assignment1/
├── Assignment 1 - Group 4-version4.ipynb  # Main notebook implementation
├── Assignment1.pdf                        # Assignment requirements
├── mapper.py                              # MapReduce mapper
├── reducer.py                             # MapReduce reducer
├── mapper_simple.py                       # Simplified mapper
├── borough_diagnostics.py                 # Borough analysis diagnostics
├── convert_to_csv.py                      # Data conversion utility
├── Hadoop_Spark Command Cheat Sheet_NYC Taxi Analysis.txt
├── data/
│   ├── taxi_zone_lookup.csv              # NYC taxi zone mapping
│   └── yellow_tripdata_2025-01.parquet   # NYC taxi trip data
└── README.md                              # This file
```

## Business Challenges

### Challenge 1: Privacy-Preserving Analytics with Synthetic Data

**Objective**: Generate synthetic healthcare data that preserves privacy while maintaining statistical utility for analysis.

**Tasks**:
- **Task I**: Exploratory Analysis of Real Healthcare Dataset
  - Analysis of HealthInsurance.csv dataset (8,802 records, 12 features)
  - Statistical profiling, distribution analysis, and correlation studies

- **Task II**: Baseline Synthetic Data Generation
  - Classical methods using Faker library and bootstrapping
  - Privacy-sensitive feature handling

- **Task III**: Advanced Synthetic Data Generation
  - Implementation using SDV (Synthetic Data Vault)
  - CTGAN (Conditional Generative Adversarial Network)
  - Gaussian Copula models

- **Task IV**: Evaluation and Validation
  - Statistical similarity testing (Kolmogorov-Smirnov tests)
  - Utility assessment (TSTR - Train on Synthetic, Test on Real)
  - Privacy risk evaluation
  - Quality metrics and diagnostic reports

**Key Results**:
- Achieved 100% diagnostic validity scores
- High statistical fidelity with synthetic models
- Successful privacy preservation with maintained utility

### Challenge 2: Mining NYC Taxi Trip Data

**Objective**: Analyze large-scale NYC taxi trip data using big data technologies (HDFS, MapReduce, PySpark).

**Tasks**:
- **Task I**: Big Data Setup & Exploration
  - Data loading from HDFS (Parquet format)
  - Schema analysis and data profiling
  - Data cleaning and outlier handling

- **Task II**: MapReduce Analysis
  - Custom mapper and reducer implementations
  - Fare aggregation by zone and borough
  - Performance comparison with Spark

- **Task III**: Frequent Travel Pattern Mining
  - PySpark FPGrowth algorithm implementation
  - Common route identification
  - Time-of-day pattern analysis

- **Task IV**: Rider Segmentation
  - K-Means clustering implementation
  - Customer segmentation based on trip characteristics
  - Behavioral pattern identification

## Technologies & Tools

### Python Libraries
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Synthetic Data**: SDV, CTGAN, Faker, sdmetrics
- **Statistical Analysis**: scipy.stats

### Big Data Technologies
- **PySpark**: DataFrame operations, SQL functions
- **PySpark ML**: FPGrowth, K-Means, VectorAssembler
- **Hadoop HDFS**: Distributed file storage
- **MapReduce**: Custom Python implementations

## Dataset Information

### Healthcare Dataset
- **Source**: Vincent Arel-Bundock's Rdatasets repository
- **File**: HealthInsurance.csv
- **Records**: 8,802 rows
- **Features**: 12 columns (age, family size, gender, education, region, health status, insurance status, etc.)

### NYC Taxi Trip Data
- **Source**: NYC TLC (Taxi and Limousine Commission)
- **Format**: Parquet files
- **Features**: Pickup/dropoff locations and timestamps, fares, distances, passenger counts
- **Additional Data**: NYC Taxi zone lookup data for location mapping

## Setup Instructions

### Prerequisites
- Python 3.9+
- Anaconda (recommended)
- Apache Spark 3.x
- Hadoop HDFS (for local development)

### Installation

1. **Navigate to project directory**:
   ```bash
   cd assignment1
   ```

2. **Install dependencies** (from root directory):
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Configure PySpark**:
   ```bash
   # Set environment variables
   export SPARK_LOCAL_IP=127.0.0.1
   export PYSPARK_PYTHON=python
   ```

4. **Setup HDFS** (if not already configured):
   - Follow Apache Hadoop installation guide for your OS
   - Parquet files are in `data/` directory

### Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the assignment notebook**:
   - Navigate to `Assignment 1 - Group 4-version4.ipynb`
   - Select the appropriate kernel (spark_env or advanced_analytics)

3. **Execute cells sequentially**:
   - Follow the notebook structure from top to bottom
   - Ensure data files are accessible before running analysis cells

## Key Findings

### Synthetic Data Generation
- CTGAN and Gaussian Copula models demonstrated high-quality synthetic data generation
- Privacy preservation achieved without sacrificing data utility
- Synthetic data suitable for sharing and analysis in healthcare applications

### NYC Taxi Data Analysis
- MapReduce vs. Spark performance comparisons validated for big data processing
- Identified frequent travel patterns and peak time behaviors
- Customer segmentation revealed distinct rider profiles based on trip characteristics

## Troubleshooting

### PySpark Session Hangs
If PySpark session creation hangs:
1. Set environment variables before starting:
   ```python
   import os
   os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
   os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'
   ```

2. Use minimal Spark configuration:
   ```python
   spark = SparkSession.builder \
           .master("local[*]") \
           .config("spark.driver.host", "127.0.0.1") \
           .config("spark.ui.enabled", "false") \
           .getOrCreate()
   ```

3. Consider using Google Colab or Databricks for cloud-based execution

### HDFS Access Issues
- Verify HDFS is running: `jps` (should show NameNode and DataNode)
- Check file paths use correct format: `hdfs://localhost:9000/path` or local paths for standalone mode

## Contributors

- Group 4 Members

## Acknowledgments

- NYC Taxi and Limousine Commission for providing trip data
- Vincent Arel-Bundock for the Rdatasets healthcare dataset
- SDV (Synthetic Data Vault) development team
