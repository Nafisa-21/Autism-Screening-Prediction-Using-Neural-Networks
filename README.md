# Autism Screening Prediction Using Neural Networks

This project explores the use of machine learning, specifically a neural network, to predict whether a person might have Autism Spectrum Disorder (ASD) based on screening responses and demographic data. Autism is a developmental disorder that impacts behavior, communication, and social interaction, and early detection can be critical. Since clinical diagnosis often lacks accessible tools, this project demonstrates how predictive modeling can assist in early screening.

## Project Overview

The entire pipeline from raw data to trained model is implemented and explained step by step.

---

## 📌 Steps Covered

### 1. Data Loading and Inspection
- Loaded the autism screening dataset using `pandas`.
- Displayed the shape, column types, and summary statistics.
- Identified missing values and irregular categorical entries.

### 2. Data Cleaning
- Filled missing `age` values with the rounded mean.
- Replaced invalid values in `ethnicity` and `relation` with more meaningful defaults.
- Standardized inconsistent categorical entries (e.g., case mismatches like "Others" vs. "others").

### 3. Exploratory Data Analysis (EDA)
- Visualized the distribution of autism-positive vs. negative cases.
- Explored geographic distribution of ASD cases.
- Analyzed age distribution and applied log transformation to reduce skewness.
- Explored relationships between gender, ethnicity, and ASD using bar plots.

### 4. Feature Engineering and Encoding
- Removed irrelevant columns (e.g., `age_desc`).
- Separated features (`X`) and target labels (`Y`).
- Applied one-hot encoding to handle categorical variables.

### 5. Data Preprocessing
- Split the data into training and testing sets.
- Standardized the features using `StandardScaler`.
- Reshaped feature arrays to match the neural network input shape `(102, 1)`.

### 6. Model Building and Training
- Constructed a neural network using TensorFlow/Keras:
  - Flatten layer to preprocess input.
  - Dense layer with ReLU activation.
  - Output layer with softmax for binary classification.
- Used Adam optimizer and categorical crossentropy loss.
- Trained the model for 20 epochs with batch size 10.
- Tracked training and validation accuracy and loss over time.

### 7. Model Evaluation
- Evaluated test accuracy on the standardized and reshaped test set.
- Converted one-hot encoded labels back to class labels for evaluation.
- Generated a classification report including precision, recall, and F1-score.

### 8. Model Saving
- Saved the trained Keras model as `autism_prediction_model.h5`.
- Trained a Random Forest classifier for comparison and saved it as `rf_model.pkl`.

---

## ✅ Outcome

This notebook shows that a simple neural network can effectively learn patterns in the autism screening dataset and make meaningful predictions. The framework sets up a strong baseline for future enhancements such as more advanced models, hyperparameter tuning, feature selection, or deployment as a web app.

---
