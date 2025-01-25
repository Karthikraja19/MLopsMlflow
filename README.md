# MLopsMlflow
M2: Process and Tooling

This code trains and evaluates three machine learning models (Random Forest, Decision Tree, and SVM) on housing data (Housing.csv) and logs metrics and models using MLflow.

Steps
1. Data Preprocessing
Loaded data and encoded categorical variables (yes/no, furnished).
Separated features (x) and target (y).
2. Data Splitting
Split data into 80% training and 20% testing sets.
3. Model Training and Evaluation
Random Forest: Trained with 20 estimators, logged parameters and metrics.
Decision Tree: Standard tree model, logged metrics.
SVM: Trained with default parameters, logged metrics.
4. Model Logging
Models and metrics logged in MLflow.
Models saved as .pkl files using Joblib.
5. Experiment Tracking
Logged each model under separate MLflow experiments:
housing-experiment-Random forest
housing-experiment - Decision Tree
housing-experiment - SVM

Requirements
Python 3.x
Libraries: pandas, scikit-learn, mlflow, joblib

Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt

Run
Execute the script to train models, evaluate performance, and track experiments in MLflow.
