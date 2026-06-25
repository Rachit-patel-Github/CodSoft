import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class SalesPredictionApp:
    """
    A complete Sales Prediction Application using Linear Regression.
    This app predicts sales based on advertising budgets for TV, Radio, and Newspaper.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Prediction System")
        self.root.geometry("700x500")
        self.root.configure(bg="#ADD8E6")
        
        self.model = None
        self.accuracy = None
        self.X_train = None
        self.y_train = None
        
        self.load_and_train_model()
        self.create_gui()
    
    def load_and_train_model(self):
        try:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_directory, "advertising.csv")
            
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"advertising.csv not found at {csv_path}")
            
            data = pd.read_csv(csv_path)
            
            if data.columns[0] == 'Unnamed: 0':
                self.X_train = data[['TV', 'Radio', 'Newspaper']].values
            else:
                self.X_train = data[['TV', 'Radio', 'Newspaper']].values
            
            self.y_train = data['Sales'].values
            
            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            
            predictions = self.model.predict(self.X_train)
            self.accuracy = r2_score(self.y_train, predictions)
            
            print(f"✓ Model trained successfully!")
            print(f"✓ Accuracy (R² Score): {self.accuracy:.4f}")
            
        except FileNotFoundError as e:
            messagebox.showerror("Error", f"Cannot find file: {e}")
            self.root.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {e}")
            self.root.destroy()
    
    def create_gui(self):
        main_frame = tk.Frame(self.root, bg="#ADD8E6")
        main_frame.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        title_label = tk.Label(
            main_frame,
            text="Sales Prediction System",
            font=("Arial", 20, "bold"),
            bg="#ADD8E6",
            fg="#003366"
        )
        title_label.pack(pady=(0, 10))
        
        accuracy_text = f"Model Accuracy (R² Score): {self.accuracy:.4f}" if self.accuracy else "Model not trained"
        accuracy_label = tk.Label(
            main_frame,
            text=accuracy_text,
            font=("Arial", 10, "italic"),
            bg="#ADD8E6",
            fg="#005500"
        )
        accuracy_label.pack(pady=(0, 20))
        
        input_frame = tk.Frame(main_frame, bg="#ADD8E6")
        input_frame.pack(pady=10)
        
        tv_label = tk.Label(
            input_frame,
            text="TV Advertisement Budget ($):",
            font=("Arial", 11),
            bg="#ADD8E6"
        )
        tv_label.grid(row=0, column=0, sticky="w", pady=5)
        self.tv_entry = tk.Entry(input_frame, font=("Arial", 11), width=20)
        self.tv_entry.grid(row=0, column=1, padx=10, pady=5)
        
        radio_label = tk.Label(
            input_frame,
            text="Radio Advertisement Budget ($):",
            font=("Arial", 11),
            bg="#ADD8E6"
        )
        radio_label.grid(row=1, column=0, sticky="w", pady=5)
        self.radio_entry = tk.Entry(input_frame, font=("Arial", 11), width=20)
        self.radio_entry.grid(row=1, column=1, padx=10, pady=5)
        
        newspaper_label = tk.Label(
            input_frame,
            text="Newspaper Advertisement Budget ($):",
            font=("Arial", 11),
            bg="#ADD8E6"
        )
        newspaper_label.grid(row=2, column=0, sticky="w", pady=5)
        self.newspaper_entry = tk.Entry(input_frame, font=("Arial", 11), width=20)
        self.newspaper_entry.grid(row=2, column=1, padx=10, pady=5)
        
        button_frame = tk.Frame(main_frame, bg="#ADD8E6")
        button_frame.pack(pady=20)
        
        predict_button = tk.Button(
            button_frame,
            text="Predict Sales",
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=10,
            command=self.predict_sales,
            cursor="hand2"
        )
        predict_button.pack(side=tk.LEFT, padx=10)
        
        clear_button = tk.Button(
            button_frame,
            text="Clear",
            font=("Arial", 12, "bold"),
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=10,
            command=self.clear_inputs,
            cursor="hand2"
        )
        clear_button.pack(side=tk.LEFT, padx=10)
        
        view_graphs_button = tk.Button(
            button_frame,
            text="View Graphs",
            font=("Arial", 12, "bold"),
            bg="#2196F3",
            fg="white",
            padx=20,
            pady=10,
            command=self.view_analysis_graphs,
            cursor="hand2"
        )
        view_graphs_button.pack(side=tk.LEFT, padx=10)
        
        result_label = tk.Label(
            main_frame,
            text="Predicted Monthly Sales (in thousands):",
            font=("Arial", 11),
            bg="#ADD8E6"
        )
        result_label.pack(pady=(20, 5))
        
        self.result_display = tk.Label(
            main_frame,
            text="---",
            font=("Arial", 32, "bold"),
            bg="#FFFFFF",
            fg="#003366",
            relief=tk.SUNKEN,
            padx=20,
            pady=20,
            width=20
        )
        self.result_display.pack(pady=10)
    
    def validate_inputs(self, tv, radio, newspaper):
        try:
            tv_val = float(tv)
            radio_val = float(radio)
            newspaper_val = float(newspaper)
            
            if tv_val < 0 or radio_val < 0 or newspaper_val < 0:
                messagebox.showerror("Invalid Input", "Budget values cannot be negative!")
                return False
            
            return True, tv_val, radio_val, newspaper_val
        
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for all fields!")
            return False, None, None, None
    
    def view_analysis_graphs(self):
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Model Analysis Graphs")
        graph_window.geometry("1200x700")
        graph_window.configure(bg="#ADD8E6")
        
        header_label = tk.Label(
            graph_window,
            text="Model Performance & Data Distribution Analysis",
            font=("Arial", 14, "bold"),
            bg="#ADD8E6",
            fg="#003366"
        )
        header_label.pack(pady=10)
        
        fig = Figure(figsize=(12, 7), dpi=100)
        
        predictions = self.model.predict(self.X_train)
        
        ax1 = fig.add_subplot(231)
        ax1.scatter(self.y_train, predictions, alpha=0.5, color='blue')
        ax1.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Sales ($K)', fontsize=9, fontweight='bold')
        ax1.set_ylabel('Predicted Sales ($K)', fontsize=9, fontweight='bold')
        ax1.set_title('Actual vs Predicted Sales', fontsize=10, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(232)
        residuals = self.y_train - predictions
        ax2.scatter(predictions, residuals, alpha=0.5, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted Sales ($K)', fontsize=9, fontweight='bold')
        ax2.set_ylabel('Residuals ($K)', fontsize=9, fontweight='bold')
        ax2.set_title('Residual Plot', fontsize=10, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(233)
        ax3.hist(residuals, bins=20, color='purple', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Residuals ($K)', fontsize=9, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax3.set_title('Residual Distribution', fontsize=10, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = fig.add_subplot(234)
        ax4.hist(self.X_train[:, 0], bins=20, color='skyblue', edgecolor='black')
        ax4.set_xlabel('TV Budget ($K)', fontsize=9, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax4.set_title('TV Budget Distribution', fontsize=10, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        ax5 = fig.add_subplot(235)
        ax5.hist(self.X_train[:, 1], bins=20, color='lightcoral', edgecolor='black')
        ax5.set_xlabel('Radio Budget ($K)', fontsize=9, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax5.set_title('Radio Budget Distribution', fontsize=10, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        ax6 = fig.add_subplot(236)
        ax6.hist(self.X_train[:, 2], bins=20, color='lightgreen', edgecolor='black')
        ax6.set_xlabel('Newspaper Budget ($K)', fontsize=9, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=9, fontweight='bold')
        ax6.set_title('Newspaper Budget Distribution', fontsize=10, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def show_prediction_graphs(self, tv_val, radio_val, newspaper_val, prediction):
        graph_window = tk.Toplevel(self.root)
        graph_window.title("Sales Prediction Analysis")
        graph_window.geometry("1000x600")
        graph_window.configure(bg="#ADD8E6")
        
        header_label = tk.Label(
            graph_window,
            text="Sales Prediction Analysis Graphs",
            font=("Arial", 14, "bold"),
            bg="#ADD8E6",
            fg="#003366"
        )
        header_label.pack(pady=10)
        
        info_label = tk.Label(
            graph_window,
            text=f"Monthly Sales Prediction: ${prediction}K | Input: TV=${tv_val}K, Radio=${radio_val}K, Newspaper=${newspaper_val}K",
            font=("Arial", 10, "italic"),
            bg="#ADD8E6",
            fg="#005500"
        )
        info_label.pack(pady=5)
        
        fig = Figure(figsize=(10, 5), dpi=100)
        
        ax1 = fig.add_subplot(121)
        predictions = self.model.predict(self.X_train)
        ax1.scatter(self.y_train, predictions, alpha=0.5, color='blue', label='Training Data')
        ax1.scatter([prediction], [prediction], color='red', s=200, marker='*', label='Your Prediction', edgecolors='black', linewidth=2)
        ax1.plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'g--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual Sales ($K)', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Predicted Sales ($K)', fontsize=10, fontweight='bold')
        ax1.set_title('Model Performance: Actual vs Predicted', fontsize=11, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(122)
        budgets = ['TV', 'Radio', 'Newspaper']
        values = [tv_val, radio_val, newspaper_val]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax2.bar(budgets, values, color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Budget ($K)', fontsize=10, fontweight='bold')
        ax2.set_title('Your Advertisement Budget Allocation', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(values):
            ax2.text(i, v + 1, f'${v}K', ha='center', fontweight='bold')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def predict_sales(self):
        tv = self.tv_entry.get().strip()
        radio = self.radio_entry.get().strip()
        newspaper = self.newspaper_entry.get().strip()
        
        if not tv or not radio or not newspaper:
            messagebox.showerror("Error", "Please fill in all fields!")
            return
        
        validation_result = self.validate_inputs(tv, radio, newspaper)
        
        if validation_result is False or validation_result[0] is False:
            return
        
        _, tv_val, radio_val, newspaper_val = validation_result
        
        try:
            input_data = np.array([[tv_val, radio_val, newspaper_val]])
            prediction = self.model.predict(input_data)[0]
            prediction = round(prediction, 2)
            self.result_display.config(text=f"${prediction}K/Month")
            
            self.show_prediction_graphs(tv_val, radio_val, newspaper_val, prediction)
            
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error making prediction: {e}")
    
    def clear_inputs(self):
        self.tv_entry.delete(0, tk.END)
        self.radio_entry.delete(0, tk.END)
        self.newspaper_entry.delete(0, tk.END)
        self.result_display.config(text="---")


if __name__ == "__main__":
    root = tk.Tk()
    app = SalesPredictionApp(root)
    root.mainloop()
