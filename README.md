# Fraudulent Transaction Detection

Welcome to the Fraudulent Transaction Detection project repository. This project aims to detect fraudulent financial transactions using machine learning techniques, specifically implementing Logistic Regression, Random Forest, and XGBoost classifiers. Below is an overview of the project, including features, code snippets, and instructions for running the code.

---

<div align="center">
  <img src="./card.jpeg" alt="Credit Card Fraud" style="border:none;">
</div>

---

## Overview

This project focuses on identifying fraudulent financial transactions by analyzing various features in the dataset. The dataset contains information about transactions, and the goal is to accurately detect fraudulent ones through the application of machine learning models.

---

## Features

- **Data Exploration**: Understanding the data distribution, statistical information, and data types.
- **Data Cleaning**: Handling missing values, data type conversion, and ensuring data consistency.
- **Feature Engineering**: Creating new features to improve model performance, including error calculations.
- **Outlier Removal**: Identifying and removing outliers from numerical columns to enhance model accuracy.
- **Model Building**: Implementing Logistic Regression, Random Forest, and XGBoost classifiers to detect fraudulent transactions.
- **Model Evaluation**: Evaluating model performance using metrics like Confusion Matrix, Classification Report, Accuracy, Precision, Recall, and F1 Score.

---

## Contents

- `fraudulent_transaction_detection.ipynb`: Jupyter notebook containing the code implementation and analysis.
- `README.md`: This file, providing an overview of the project.
- `Fraud.csv`: Dataset used for training and testing the models.

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Fraudulent-Transaction-Detection.git
   cd Fraudulent-Transaction-Detection
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook**:
   ```bash
   fraudulent_detection.ipynb

---

## Data Exploration and Cleaning
- **Data Loading**: The dataset is read and the first few rows are displayed.
- **Data Types**: The variable types in each column are identified and converted as necessary.
- **Null Values**: Null or missing values in the dataset are verified.
- **Statistical Information**: Statistical information about the variables is displayed.
- **Class Distribution**: The percentage of non-fraud and fraud transactions is calculated.

---

## Feature Engineering
- **Categorical Encoding**: Converting categorical features to numerical values using one-hot encoding.
- **Error Calculation**: Creating new features by calculating errors in the original and destination balances.
- **Outlier Removal**: Identifying and removing outliers from numerical columns.

---

## Model Building and Evaluation
- **Data Splitting**: The dataset is split into training and testing sets using train_test_split.
- **Scaling**: Features are scaled using StandardScaler.
- **Model Training**: Logistic Regression, Random Forest, and XGBoost classifiers are trained on the scaled data.
- **Predictions**: The trained models make predictions on the test set.
- **Evaluation**: Model performance is measured using Confusion Matrix, Classification Report, and various metrics (Accuracy, Precision, Recall, F1 Score).

---

## Key Insights
- Identified significant features affecting fraudulent transactions.
- Trained and compared multiple classifiers to detect fraudulent transactions accurately.
- Evaluated model performance using a comprehensive set of metrics.

---

## Tools and Libraries
- `Pandas`: For data manipulation and analysis.
- `Matplotlib`: For creating static, animated, and interactive visualizations.
- `Seaborn`: For statistical data visualization.
- `scikit-learn`: For implementing machine learning models and evaluation metrics.
- `XGBoost`: For gradient boosting algorithms.
- `Scikit-plot`: For plotting confusion matrix and other model evaluation plots.

---

## Contributing
- If you have suggestions or improvements, feel free to open an issue or create a pull request.

---

## Thank you for visiting! If you find this project useful, please consider starring the repository. Happy coding!

---
