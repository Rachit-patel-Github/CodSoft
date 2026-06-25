import threading
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("TkAgg")

DATA_FILE = "creditcard.csv"
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
PREDICT_FEATURES = ["Amount"] + [f"V{i}" for i in range(1, 5)]

models = {}
metrics = {}
scaler_amount = None
scaler_time = None
trained = False


def update_status(message: str):
    status_var.set(message)


def load_and_train():
    def worker():
        global models, metrics, scaler_amount, scaler_time, trained
        try:
            root.after(0, lambda: update_status("Loading data…"))
            df = pd.read_csv(DATA_FILE)

            if "Class" not in df.columns:
                raise ValueError("Dataset does not contain a 'Class' column.")

            root.after(0, lambda: update_status("Scaling Amount and Time…"))
            scaler_amount = StandardScaler()
            scaler_time = StandardScaler()
            df["Amount_scaled"] = scaler_amount.fit_transform(df[["Amount"]])
            df["Time_scaled"] = scaler_time.fit_transform(df[["Time"]])
            df = df.drop(columns=["Amount", "Time"])

            X = df[[*FEATURE_COLUMNS[:-2], "Amount_scaled", "Time_scaled"]]
            y = df["Class"]

            root.after(0, lambda: update_status("Splitting data…"))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            root.after(0, lambda: update_status("Applying SMOTE…"))
            smote = SMOTE(random_state=42)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

            root.after(0, lambda: update_status("Training models…"))
            lr = LogisticRegression(max_iter=1000, random_state=42)
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )
            lr.fit(X_train_res, y_train_res)
            rf.fit(X_train_res, y_train_res)

            root.after(0, lambda: update_status("Evaluating models…"))
            for name, model in [("Logistic Regression", lr), ("Random Forest", rf)]:
                y_pred = model.predict(X_test)
                y_score = model.predict_proba(X_test)[:, 1]
                metrics[name] = {
                    "Precision": precision_score(y_test, y_pred, zero_division=0),
                    "Recall": recall_score(y_test, y_pred, zero_division=0),
                    "F1": f1_score(y_test, y_pred, zero_division=0),
                    "AUC": roc_auc_score(y_test, y_score),
                    "ConfusionMatrix": confusion_matrix(y_test, y_pred).tolist(),
                }
                models[name] = model

            trained = True
            root.after(0, update_status, "Training complete!")
            root.after(0, enable_predict)
            root.after(0, update_metrics_display)
            root.after(0, render_confusion_matrix)
        except FileNotFoundError:
            root.after(0, lambda: messagebox.showerror(
                "File Not Found",
                f"Could not find {DATA_FILE}. Please place creditcard.csv in the same folder as app.py and try again."
            ))
            root.after(0, update_status, "Dataset not found.")
        except Exception as exc:
            root.after(0, update_status, f"Error: {exc}")

    threading.Thread(target=worker, daemon=True).start()
    load_button.config(state="disabled")
    update_status("Started training in background…")


def enable_predict():
    predict_button.config(state="normal")
    load_button.config(state="normal")


def get_selected_model():
    return model_choice.get()


def update_metrics_display(*args):
    selected = get_selected_model()
    if selected not in metrics:
        return
    for item in metrics_table.get_children():
        metrics_table.delete(item)
    for key in ["Precision", "Recall", "F1", "AUC"]:
        metrics_table.insert("", "end", values=(key, f"{metrics[selected][key]:.4f}"))


def render_confusion_matrix(*args):
    selected = get_selected_model()
    if selected not in metrics:
        return
    matrix = np.array(metrics[selected]["ConfusionMatrix"])
    figure.clear()
    ax = figure.add_subplot(111)
    cax = ax.matshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, int(value), va="center", ha="center", color="white" if value > matrix.max() / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    figure.colorbar(cax, ax=ax)
    canvas.draw()


def predict_transaction():
    if not trained:
        messagebox.showwarning("Model Not Trained", "Please load and train the models before predicting.")
        return

    selected = get_selected_model()
    if selected not in models:
        messagebox.showwarning("Select Model", "Please choose a valid model for prediction.")
        return

    try:
        amount = float(amount_entry.get())
        feature_values = [float(amount)]
        for feat in ["V1", "V2", "V3", "V4"]:
            feature_values.append(float(feature_entries[feat].get()))
        for _ in range(24):
            feature_values.append(0.0)
        amount_scaled = scaler_amount.transform(np.array([[amount]]))[0, 0]
        time_scaled = 0.0
        feature_values.append(amount_scaled)
        feature_values.append(time_scaled)

        X_input = np.array(feature_values).reshape(1, -1)
        model = models[selected]
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0, 1]

        if prediction == 1:
            result_label.config(text=f"🚨 FRAUD DETECTED ({probability:.4f})", fg="#e74c3c")
        else:
            result_label.config(text=f"✅ GENUINE ({1 - probability:.4f})", fg="#2ecc71")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values for Amount and V1–V4.")
    except Exception as exc:
        messagebox.showerror("Prediction Error", str(exc))


root = tk.Tk()
root.title("Credit Card Fraud Detector")
root.geometry("920x650")
root.configure(bg="#1a1a2e")
root.resizable(False, False)

style = ttk.Style(root)
style.theme_use("clam")
style.configure("TFrame", background="#16213e")
style.configure("TLabel", background="#16213e", foreground="white", font=("Segoe UI", 10))
style.configure("Title.TLabel", background="#1a1a2e", foreground="white", font=("Segoe UI", 14, "bold"))
style.configure("TButton", background="#e94560", foreground="white", relief="flat", font=("Segoe UI", 10))
style.map("TButton", background=[("active", "#ff4d73")])
style.configure("Treeview", background="#16213e", foreground="white", fieldbackground="#16213e", rowheight=24)
style.configure("Treeview.Heading", background="#0f3460", foreground="white")

header_frame = ttk.Frame(root)
header_frame.place(x=20, y=20, width=880, height=60)

title_label = ttk.Label(header_frame, text="💳 Credit Card Fraud Detector", style="Title.TLabel")
title_label.pack(side="left", padx=(0, 20))

load_button = ttk.Button(header_frame, text="Load & Train", command=load_and_train)
load_button.pack(side="right")

main_frame = ttk.Frame(root)
main_frame.place(x=20, y=100, width=880, height=470)

metrics_frame = ttk.Frame(main_frame)
metrics_frame.place(x=0, y=0, width=430, height=470)
metrics_title = ttk.Label(metrics_frame, text="📊 Metrics", style="Title.TLabel")
metrics_title.pack(anchor="nw", pady=(0, 10))

model_choice = tk.StringVar(value="Logistic Regression")
model_menu = ttk.OptionMenu(metrics_frame, model_choice, model_choice.get(), "Logistic Regression", "Random Forest", command=lambda *_: [update_metrics_display(), render_confusion_matrix()])
model_menu.pack(anchor="nw", pady=(0, 10))

metrics_table = ttk.Treeview(metrics_frame, columns=("Metric", "Value"), show="headings", height=6)
metrics_table.heading("Metric", text="Metric")
metrics_table.heading("Value", text="Value")
metrics_table.column("Metric", width=150, anchor="w")
metrics_table.column("Value", width=150, anchor="center")
metrics_table.pack(anchor="nw", pady=(0, 10))

confusion_canvas_frame = ttk.Frame(metrics_frame)
confusion_canvas_frame.pack(anchor="nw", fill="both", expand=True)
figure = Figure(figsize=(4.0, 3.0), dpi=90)
canvas = FigureCanvasTkAgg(figure, master=confusion_canvas_frame)
canvas.get_tk_widget().pack(fill="both", expand=True)

predict_frame = ttk.Frame(main_frame)
predict_frame.place(x=450, y=0, width=430, height=470)
predict_title = ttk.Label(predict_frame, text="🔍 Predict Transaction", style="Title.TLabel")
predict_title.pack(anchor="nw", pady=(0, 10))

labels = ["Amount", "V1", "V2", "V3", "V4"]
feature_entries = {}

for label_text in labels:
    row = ttk.Frame(predict_frame)
    row.pack(anchor="nw", pady=5, fill="x")
    label = ttk.Label(row, text=label_text)
    label.pack(side="left", padx=(0, 10))
    entry = ttk.Entry(row)
    entry.pack(side="left", fill="x", expand=True)
    if label_text == "Amount":
        amount_entry = entry
    else:
        feature_entries[label_text] = entry

for feat in ["V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]:
    feature_entries[feat] = tk.StringVar(value="0.0")

predict_button = ttk.Button(predict_frame, text="Predict", command=predict_transaction, state="disabled")
predict_button.pack(anchor="nw", pady=(20, 10), fill="x")

result_label = ttk.Label(predict_frame, text="Prediction result will appear here", font=("Segoe UI", 10, "bold"))
result_label.pack(anchor="nw", pady=(10, 0))

status_frame = ttk.Frame(root)
status_frame.place(x=20, y=590, width=880, height=40)
status_var = tk.StringVar(value="Ready to load and train.")
status_label = ttk.Label(status_frame, textvariable=status_var, style="TLabel")
status_label.pack(anchor="w", padx=10)

root.mainloop()
