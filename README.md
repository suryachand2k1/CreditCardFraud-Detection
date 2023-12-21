# Credit Card Fraud Detection Using Machine Learning

## Project Overview
This project aims to detect credit card fraud using machine learning techniques. It leverages the European Credit Card dataset, which classifies transactions as either NORMAL or FRAUD.

### Dataset
- **European Credit Card Dataset**
- Dataset URL: [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Features
- **Upload Dataset:** Upload the dataset to the application.
- **Preprocess Dataset:** Remove missing values, drop unnecessary columns, and split the dataset for training and testing.
- **runKMEANS:** Cluster the train data into NORMAL or FRAUD categories and build a prediction model.
- **Upload Test Transaction & Evaluate Risk Zone:** Upload new test transactions and use fuzzy logic and KMEANS to evaluate and classify transactions.

## How to Run
1. Start the application by executing `run.bat`.
2. Upload the dataset and preprocess it.
3. Apply the KMeans algorithm to train the model.
4. Upload test transactions for fraud detection.

### Screenshots
Screenshots demonstrating each step are included in the project files.

## Requirements
- Python
- Machine Learning Libraries (refer to `requirements.txt` for the complete list)
