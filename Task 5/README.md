# Credit Card Fraud Detection GUI

This repository contains a simple desktop application for credit card fraud detection using a Tkinter GUI and a machine learning pipeline built with `scikit-learn`.

## Files

- `app.py` - Main application file containing the full GUI, data pipeline, model training, evaluation, and prediction logic.
- `requirements.txt` - Python package dependencies required to run the application.
- `creditcard.csv` - Dataset file. Place it in the same folder as `app.py` before running the app.

## Dataset

The app expects the `creditcard.csv` dataset in the same directory. The dataset should include:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class` (0 = genuine, 1 = fraud)

The dataset has a strong class imbalance with very few fraud cases.

## Features

- Loads the dataset from `creditcard.csv`
- Standardizes `Amount` and `Time` using `StandardScaler`
- Splits the data into training and test sets with `stratify=y`
- Applies `SMOTE` only on the training set to address class imbalance
- Trains two models:
  - Logistic Regression
  - Random Forest
- Computes evaluation metrics for each model:
  - Precision
  - Recall
  - F1 score
  - ROC AUC score
  - Confusion matrix
- Displays metrics and the confusion matrix in a Tkinter GUI
- Allows prediction for custom inputs using the selected model

## GUI Overview

The application window includes:

- Top bar with title and `Load & Train` button
- Left panel:
  - Model selection dropdown
  - Metrics table showing Precision, Recall, F1, and AUC
  - Confusion matrix plot
- Right panel:
  - Input fields for `Amount` and `V1`–`V4`
  - `Predict` button
  - Prediction result label with fraud/genuine status
- Bottom status bar showing pipeline progress messages

## Installation

1. Install Python 3.11 or 3.13+.
2. Open a terminal in this folder.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the App

In the project folder, run:

```bash
python app.py
```

Then click the `Load & Train` button. Wait for training to complete before using the `Predict` button.

## Notes

- The app uses Tkinter for the GUI, so no web server or web framework is required.
- The prediction form currently only takes `V1`–`V4` as manual inputs and fills the remaining PCA features with `0.0`.
- If `creditcard.csv` is missing, the app shows an error message instructing the user to place the dataset in the same folder.
