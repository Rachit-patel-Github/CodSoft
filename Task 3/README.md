# Iris Flower Classifier

A friendly Python desktop application that predicts the species of an Iris flower using a trained machine learning model. The app launches a simple GUI with sliders for sepal and petal measurements and returns a prediction for Setosa, Versicolor, or Virginica.

## Features

- Interactive sliders for sepal length, sepal width, petal length, and petal width
- Automatically trains a Random Forest model when the app starts
- Displays dataset min/max measurement ranges from `IRIS.csv`
- Shows the predicted Iris species instantly after clicking the button
- Built with `tkinter` and `ttk` for a clean desktop interface

## Prerequisites

- Python 3.8 or newer
- `pandas`
- `scikit-learn`

## Installation & Setup

1. Make sure your project folder contains both `iris_classifier_gui.py` and `IRIS.csv`.
2. Open a terminal in the project directory:
   ```bash
   cd "c:\CodSoft\Task 3"
   ```
3. Install the required packages:
   ```bash
   pip install pandas scikit-learn
   ```

## Usage

1. Run the application from the project folder:
   ```bash
   python iris_classifier_gui.py
   ```
2. Adjust the sliders for sepal length, sepal width, petal length, and petal width.
3. Click the **Predict Species** button.
4. The app will display the predicted Iris species at the bottom of the window.

## Notes

- The model trains automatically each time the application starts using the local `IRIS.csv` file.
- If the file is missing or the data format is invalid, the app will show an error message.

## Project Structure

- `iris_classifier_gui.py` — main application script
- `IRIS.csv` — dataset used for training the classifier
- `README.md` — project documentation

Enjoy exploring the Iris dataset with an interactive GUI! Feel free to customize the model or add new features like save/load presets or dataset visualization. 