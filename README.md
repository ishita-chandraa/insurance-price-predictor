# 💰 Insurance Price Predictor

A Machine Learning project that predicts individual insurance charges based on personal and lifestyle factors such as age, BMI, smoking status, and more.

---

## 📌 Project Overview

Health insurance costs are influenced by multiple factors like age, body mass index, smoking habits, and region.  
This project builds a regression-based machine learning model to estimate insurance charges based on user input features.

The project demonstrates an end-to-end ML workflow including data preprocessing, model training, and evaluation.

---

## 🧠 Problem Statement

Given the following user attributes:

- Age  
- Sex  
- BMI  
- Number of children  
- Smoking status  
- Region  

Predict the **insurance charges** accurately using a regression model.

---

## 🛠️ Tech Stack

- Python 🐍  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn (for EDA, if used)

---

## 📊 Machine Learning Workflow

1. Data loading and exploration  
2. Data preprocessing:
   - Handling categorical variables  
   - Encoding features  
3. Exploratory Data Analysis (EDA)  
4. Model training (Regression model)  
5. Model evaluation  

---

## 📁 Project Structure
insurance-price-predictor/
│
├── app.py                # Main application file 
├── model.pkl             # Trained ML model 
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
│
├── data/
│   └── insurance.csv     # Dataset 
│
├── notebooks/
│   └── model_training.ipynb   # Jupyter notebook 
│
└── src/
    ├── data_preprocessing.py  # Data cleaning & preprocessing
    ├── model.py               # Model training logic
    └── utils.py              # Helper functions

