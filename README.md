# Fraud Detection System

## Project Overview
This Fraud Detection System is designed to identify fraudulent transactions using machine learning techniques. It reads transaction data in JSON format from a `.txt` file, preprocesses it, and applies feature engineering to extract meaningful insights and patterns. The data is then used to train a RandomForestClassifier to predict fraudulent activities.

## Technologies Used
- Python
- Pandas
- NumPy
- scikit-learn
- imbalanced-learn

## Features
- Data cleaning and preprocessing, including handling missing values and converting data types.
- Dynamic reading of JSON formatted data from a text file.
- Feature engineering to enhance model performance.
- Over-sampling using SMOTE to handle class imbalance.
- Machine learning model training and evaluation.
- Performance metrics evaluation including accuracy, precision, recall, and cross-validation scores.
