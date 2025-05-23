# 🫀 Heart Disease Classification using Machine Learning

## 📌 Overview

This repository presents a comprehensive machine learning project aimed at predicting the presence of heart disease based on patient medical attributes. It uses multiple classification models, hyperparameter tuning, k-fold cross-validation, and model explainability techniques to ensure high performance and transparency.

**🔗 Dataset:** [Heart Disease Dataset – Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## 🎯 Project Objectives

* Predict the likelihood of heart disease using machine learning.
* Compare multiple models and select the best performer.
* Use explainable AI techniques (SHAP) for transparency.
* Build a reproducible pipeline for robust deployment.

## 🧠 Models Implemented

The notebook evaluates the performance of 7 models:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* XGBoost
* LightGBM

All models are optimized using:

* 🔍 Hyperparameter Tuning
* 🔄 K-Fold Cross Validation
* 📈 ROC-AUC as the primary evaluation metric

## 🧪 Performance Evaluation

Each model is evaluated using:

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* ROC-AUC Curve
* SHAP (SHapley Additive exPlanations) for explainability

## 🖥️ Features

* Interactive user interface for real-time predictions
* Feature importance visualizations
* Model interpretability using SHAP
* End-to-end ML pipeline from data preprocessing to model deployment

## 💾 Installation & Usage

### Requirements

* Python 3.8+
* Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook Heart\ Disease\ Classification.ipynb
```

## 🗂️ Project Structure

```bash
├── Heart Disease Classification.ipynb    # Main project notebook
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── assets/                               # Visuals and SHAP plots (if any)
```

## 🧠 Key Learnings

* Importance of early detection using ML in healthcare.
* Techniques for model selection and hyperparameter optimization.
* Using SHAP to build trust in model predictions.

## 🙌 Acknowledgments

* Dataset by John Smith on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
* Inspired by healthcare use cases in AI

## 📜 License

This project is licensed under the MIT License. See `LICENSE` for more information.
