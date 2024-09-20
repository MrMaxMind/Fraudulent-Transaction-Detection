# **Fraudulent Transaction Detection**

Welcome to the Fraudulent Transaction Detection project repository. This project aims to detect fraudulent financial transactions using machine learning techniques, specifically implementing Logistic Regression, Random Forest, and XGBoost classifiers. Below is an overview of the project, including features, code snippets, and instructions for running the code.

---

<div align="center">
  <img src="./Fraud-Detection.png" alt="Fraud Detection" style="border:none;">
</div>

---

## 🚀 **Overview**

This project focuses on identifying fraudulent financial transactions by analyzing various features in the dataset. The dataset contains information about transactions, and the goal is to accurately detect fraudulent ones through the application of machine learning models.

---

## ✨ **Features**

- **Data Exploration**: Understanding the data distribution, statistical information, and data types.
- **Data Cleaning**: Handling missing values, data type conversion, and ensuring data consistency.
- **Feature Engineering**: Creating new features to improve model performance, including error calculations.
- **Outlier Removal**: Identifying and removing outliers from numerical columns to enhance model accuracy.
- **Model Building**: Implementing Logistic Regression, Random Forest, and XGBoost classifiers to detect fraudulent transactions.
- **Model Evaluation**: Evaluating model performance using metrics like Confusion Matrix, Classification Report, Accuracy, Precision, Recall, and F1 Score.

---

## 📂 **Contents**

- `fraudulent_detection.ipynb`: Jupyter notebook containing the code implementation and analysis.
- `requirements.txt`: Python dependencies required to run the project.

---

## 🛠️  **Getting Started**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MrMaxMind/Fraudulent-Transaction-Detection.git
   cd Fraudulent-Transaction-Detection
2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook**:
   ```bash
   fraudulent_detection.ipynb

---

## 🔍 **Data Exploration & Cleaning**

- **📥 Data Loading**: 
   - The dataset is loaded and the first few rows are displayed to give an overview of the data structure.
   
- **🔢 Data Types**: 
   - Each column's data type is identified and converted where necessary (e.g., converting object types to numeric or categorical).
   
- **🔍 Null Value Detection**: 
   - The dataset is checked for missing or null values, and appropriate handling techniques are applied (e.g., imputation or removal).
   
- **📊 Statistical Overview**: 
   - Key statistical information (mean, median, standard deviation, etc.) for numerical features is displayed to better understand the data distribution.
   
- **📈 Class Distribution**: 
   - The percentage of non-fraud and fraud transactions is calculated to detect class imbalance, which is critical for model performance.

---

## 🛠️ **Feature Engineering**

- **🔄 Categorical Encoding**: 
   - Categorical variables are converted into numerical values using **one-hot encoding** to ensure models can process them effectively.
   
- **⚖️ Error Calculation**: 
   - New features are created by calculating errors between original and destination balances (e.g., balance difference or percentage error).
   
- **🚫 Outlier Detection & Removal**: 
   - Outliers in numerical columns are identified and removed (or transformed) to reduce their impact on model performance.

---

## 🤖 **Model Building & Evaluation**

- **📤 Data Splitting**: 
   - The dataset is divided into training and testing sets using the **train_test_split** method to evaluate model performance on unseen data.
   
- **⚙️ Feature Scaling**: 
   - Numerical features are scaled using **StandardScaler** to standardize the data and improve model convergence, especially for models sensitive to feature scaling.
   
- **🏋️ Model Training**: 
   - Multiple classifiers, including **Logistic Regression**, **Random Forest**, and **XGBoost**, are trained on the preprocessed and scaled data to compare performance.
   
- **🔮 Predictions**: 
   - The trained models make predictions on the test set, providing insight into how well they generalize to unseen data.
   
- **📊 Evaluation**: 
   - Model performance is assessed using:
     - **Confusion Matrix**: To understand the classification accuracy.
     - **Classification Report**: Provides detailed metrics like **Precision**, **Recall**, **F1 Score**, and **Support** for each class.
     - **Accuracy, Precision, Recall, F1 Score**: To compare overall model performance and evaluate trade-offs between false positives and false negatives.

---


## 🔍 **Key Insights**

- Identified significant features affecting fraudulent transactions.
- Trained and compared multiple classifiers to detect fraudulent transactions accurately.
- Evaluated model performance using a comprehensive set of metrics.

---

## 🛠️ **Tools and Libraries**

- `Pandas`: For data manipulation and analysis.
- `Matplotlib`: For creating static, animated, and interactive visualizations.
- `Seaborn`: For statistical data visualization.
- `scikit-learn`: For implementing machine learning models and evaluation metrics.
- `XGBoost`: For gradient boosting algorithms.
- `Scikit-plot`: For plotting confusion matrix and other model evaluation plots.

---

## 🤝 **Contributing**
If you have suggestions or improvements, feel free to open an issue or create a pull request.

---

## ⭐ **Thank You!**

Thank you for visiting! If you find this project useful, please consider starring the repository. Happy coding!

