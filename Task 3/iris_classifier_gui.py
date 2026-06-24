import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

CSV_FILENAME = "IRIS.csv"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SLIDER_RANGE = (0.0, 8.0)
SLIDER_STEP = 0.1


def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing_cols = [col for col in FEATURES + ["species"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    return df


def train_model(df):
    X = df[FEATURES].values
    y = df["species"].values

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    return model, encoder


def build_input_slider(parent, label_text, var, row):
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=0, padx=8, pady=6, sticky="w")

    value_label = ttk.Label(parent, text=f"{var.get():.1f}")
    value_label.grid(row=row, column=2, padx=8, pady=6)

    slider = ttk.Scale(
        parent,
        variable=var,
        from_=SLIDER_RANGE[0],
        to=SLIDER_RANGE[1],
        orient="horizontal",
        command=lambda *_ , lbl=value_label, v=var: lbl.config(text=f"{v.get():.1f}"),
    )
    slider.grid(row=row, column=1, padx=8, pady=6, sticky="ew")
    slider.set((SLIDER_RANGE[0] + SLIDER_RANGE[1]) / 2)

    return slider


def main():
    try:
        csv_path = os.path.join(os.path.dirname(__file__), CSV_FILENAME)
        iris_df = load_dataset(csv_path)
        model, encoder = train_model(iris_df)
    except Exception as exc:
        messagebox.showerror("Data Load Error", f"Could not load and train model:\n{exc}")
        sys.exit(1)

    root = tk.Tk()
    root.title("Iris Flower Classifier")
    root.resizable(False, False)
    root.geometry("520x420")

    main_frame = ttk.Frame(root, padding=16)
    main_frame.grid(row=0, column=0, sticky="nsew")

    title_label = ttk.Label(
        main_frame,
        text="Iris Flower Classifier",
        font=(None, 18, "bold"),
    )
    title_label.grid(row=0, column=0, columnspan=3, pady=(0, 12))

    sliders_frame = ttk.LabelFrame(main_frame, text="Input features")
    sliders_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)
    sliders_frame.columnconfigure(1, weight=1)

    vars = [tk.DoubleVar(value=5.0) for _ in FEATURES]
    slider_widgets = []
    for idx, feature in enumerate(FEATURES):
        widget = build_input_slider(sliders_frame, feature.replace("_", " ").title(), vars[idx], idx)
        slider_widgets.append(widget)

    result_label = ttk.Label(
        main_frame,
        text="Enter measurements and click Predict Species.",
        wraplength=480,
        font=(None, 14),
        foreground="#333333",
    )
    result_label.grid(row=3, column=0, columnspan=3, pady=(18, 8))

    def predict_species():
        try:
            sample = [[var.get() for var in vars]]
            prediction = model.predict(sample)
            species = encoder.inverse_transform(prediction)[0]
            result_label.config(text=f"Predicted species: {species}", foreground="#1a6600")
        except Exception as exc:
            messagebox.showerror("Prediction Error", f"There was an error making the prediction:\n{exc}")

    predict_button = ttk.Button(main_frame, text="Predict Species", command=predict_species)
    predict_button.grid(row=2, column=0, columnspan=3, pady=12, ipadx=8)

    if not iris_df.empty:
        sample_range = iris_df[FEATURES].describe().loc[["min", "max"]].round(1)
        range_text = (
            f"Feature ranges from CSV:\n"
            f"Sepal Length {sample_range.at['min','sepal_length']}–{sample_range.at['max','sepal_length']}, "
            f"Sepal Width {sample_range.at['min','sepal_width']}–{sample_range.at['max','sepal_width']}, "
            f"Petal Length {sample_range.at['min','petal_length']}–{sample_range.at['max','petal_length']}, "
            f"Petal Width {sample_range.at['min','petal_width']}–{sample_range.at['max','petal_width']}"
        )
        info_label = ttk.Label(main_frame, text=range_text, wraplength=480, font=(None, 9), foreground="#555555")
        info_label.grid(row=4, column=0, columnspan=3, pady=(0, 6))

    root.mainloop()


if __name__ == "__main__":
    main()
