
# Water Quality Analysis Using Machine Learning

## Project Overview

This project focuses on the analysis and prediction of water quality using machine learning models. The goal is to evaluate various water quality parameters and classify the water as safe or unsafe based on these features.

## Dataset
- The dataset used for this project is available on Kaggle: [Water Potability Dataset](https://www.kaggle.com/adityakadiwal/water-potability).
- The dataset includes various features like pH, hardness, solids, chloramines, sulfate, conductivity, and more.

## Features

- **Dataset**: Includes attributes such as pH, hardness, solids, chloramines, sulfate, conductivity, and more, which are used to predict water quality.
- **Models**: Different machine learning models like Logistic Regression, Decision Trees, Random Forest, and Support Vector Machine are used for analysis.
- **Evaluation**: The models are evaluated using performance metrics such as accuracy, precision, recall, and F1 score.

## Installation Guide

### 1. Install Python
   - Download and install Python from: https://www.python.org/downloads/.
   - Ensure Python is added to your system's PATH.

### 2. Set Up a Virtual Environment (optional but recommended)
   - Run the following commands to create and activate a virtual environment:
     ```
     python -m venv env
     source env/bin/activate   # On macOS/Linux
     .\env\Scripts\activate    # On Windows
     ```

### 3. Install Dependencies
   - Install the necessary Python packages by running:
     ```
     pip install -r requirements.txt
     ```
   - The `requirements.txt` file should contain:
     ```
     pandas
     numpy
     scikit-learn
     matplotlib
     seaborn
     jupyterlab
     ```

### 4. Install JupyterLab (for Interactive Analysis)
   - To perform interactive analysis using Jupyter notebooks:
     ```
     pip install jupyterlab
     jupyter lab
     ```

### 5. Install MongoDB Compass (Optional for Database)
   - If you're using MongoDB for storing analysis results, download MongoDB Compass from: https://www.mongodb.com/products/compass.
   - Use this tool to connect to your MongoDB instance and manage the data.

## Project Structure

- **data/**: Contains the dataset used for training and testing the machine learning models.
- **notebooks/**: Jupyter notebooks for data exploration, model building, and analysis.
- **src/**: Python scripts for data preprocessing, model training, and evaluation.
- **results/**: Stores results of model evaluations and predictions.
- **requirements.txt**: Lists the required Python packages for the project.

## How to Run the Project

### 1. Data Preprocessing
   - Load the dataset, clean the data, handle missing values, and perform necessary feature engineering.

### 2. Train Machine Learning Models
   - Use various machine learning algorithms (Logistic Regression, Decision Tree, etc.) to train on the water quality dataset.
   - Evaluate models using cross-validation, and compare performance.

### 3. Evaluate Model Performance
   - Use performance metrics such as accuracy, precision, recall, and F1 score to evaluate the models.
   - Visualize the results using confusion matrices and ROC curves.

### 4. Predict Water Quality
   - After training the models, use them to make predictions on new water quality data.

## Example Usage

```python
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load dataset
data = pd.read_csv('data/water_quality.csv')

# Split data into features and target
X = data.drop('Potability', axis=1)
y = data['Potability']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

