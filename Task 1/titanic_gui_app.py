import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


def _extract_title(name: pd.Series) -> pd.Series:
    return name.str.extract(r' ([A-Za-z]+)\.', expand=False)


def _train_model(df: pd.DataFrame, random_state: int = 42):
    df = df.copy()

    df['Title'] = _extract_title(df['Name'])
    title_replacements = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['Title'] = df['Title'].replace(title_replacements)

    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Age'] = df['Age'].fillna(df['Age'].median())

    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

    df['AgeBin'] = pd.qcut(df['Age'], 4, labels=[0, 1, 2, 3]).astype(int)
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=[0, 1, 2, 3]).astype(int)

    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']
    df = df.drop(cols_to_drop, axis=1)

    label_encoders = {}
    for col in ['Sex', 'Embarked', 'Title']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }

    gb_model = GradientBoostingClassifier(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=gb_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred, digits=4)

    importances = pd.Series(getattr(best_model, 'feature_importances_', np.zeros(X.shape[1])), index=X.columns)
    importances = importances.sort_values(ascending=False)

    artifacts = {
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_columns': list(X.columns),
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'metrics': {
            'accuracy': float(acc),
            'roc_auc': float(auc),
            'report': report,
        },
        'importances': importances,
    }
    return artifacts


def _prepare_single_passenger(artifacts, passenger: dict) -> np.ndarray:
    row = passenger.copy()
    for k in ['Name', 'Ticket', 'Cabin', 'PassengerId']:
        row.setdefault(k, np.nan)

    df = pd.DataFrame([row])
    df['Title'] = _extract_title(df['Name'])
    title_replacements = {
        'Lady': 'Rare', 'Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare',
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'
    }
    df['Title'] = df['Title'].replace(title_replacements)

    # Age/Fare imputation: since we don't have the training distribution here,
    # we fall back to simple median-like defaults computed from the passenger itself.
    # For better consistency, we could store training medians; keep it simple for GUI.
    if pd.isna(df.loc[0, 'Age']):
        df.loc[0, 'Age'] = df['Age'].median() if not pd.isna(df['Age'].median()) else 30
    if pd.isna(df.loc[0, 'Fare']):
        df.loc[0, 'Fare'] = df['Fare'].median() if not pd.isna(df['Fare'].median()) else 30

    if pd.isna(df.loc[0, 'Embarked']):
        df.loc[0, 'Embarked'] = 'S'

    df['FamilySize'] = df['SibSp'].fillna(0) + df['Parch'].fillna(0) + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0

    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)

    age = float(df.loc[0, 'Age'])
    fare = float(df.loc[0, 'Fare'])
    df['AgeBin'] = pd.cut([age], bins=[-np.inf, 16, 32, 48, np.inf], labels=[0, 1, 2, 3])[0]
    df['FareBin'] = pd.cut([fare], bins=[-np.inf, 10, 30, 80, np.inf], labels=[0, 1, 2, 3])[0]
    df['AgeBin'] = int(df['AgeBin'])
    df['FareBin'] = int(df['FareBin'])

    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Fare']
    for c in cols_to_drop:
        if c in df.columns:
            df = df.drop(c, axis=1)

    for col in ['Sex', 'Embarked', 'Title']:
        le = artifacts['label_encoders'][col]
        val = str(df.loc[0, col])
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])[0]

    feature_columns = artifacts['feature_columns']
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_columns]

    scaler = artifacts['scaler']
    X_scaled = scaler.transform(df.values)
    return X_scaled


class TitanicGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('🚢 Titanic Survival Predictor')
        self.geometry('1050x900')
        self.resizable(True, True)

        self.artifacts = None

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=15)
        frm.pack(fill='both', expand=True)

        # Header
        header = ttk.Label(frm, text='🚢 Titanic Survival Prediction System', font=('Arial', 16, 'bold'))
        header.pack(pady=10)

        # Step 1: Train Model
        top = ttk.LabelFrame(frm, text='STEP 1: Train the Model', padding=15)
        top.pack(fill='x', pady=(0, 15))

        info_text = tk.Text(top, height=3, width=80, wrap='word')
        info_text.insert('1.0', 
            'First, click the button below to train the AI model using the Titanic dataset. '
            'This will analyze historical passenger data to learn who survived. '
            'This may take 2-3 minutes. You only need to do this once.')
        info_text.config(state='disabled')
        info_text.pack(fill='x', padx=5, pady=5)

        btn_frame = ttk.Frame(top)
        btn_frame.pack(fill='x', pady=10)

        self.train_btn = ttk.Button(btn_frame, text='▶ TRAIN MODEL', command=self.on_train)
        self.train_btn.pack(side='left', padx=5)

        self.dataset_label = ttk.Label(btn_frame, text='Status: Ready', foreground='green', font=('Arial', 10, 'bold'))
        self.dataset_label.pack(side='left', padx=20)

        # Step 2: Model Results
        metrics = ttk.LabelFrame(frm, text='STEP 2: Model Performance', padding=15)
        metrics.pack(fill='both', expand=False, pady=(0, 15))

        result_info = tk.Text(metrics, height=2, width=80, wrap='word')
        result_info.insert('1.0', 
            'After training, you\'ll see the model\'s accuracy and other performance metrics here.')
        result_info.config(state='disabled')
        result_info.pack(fill='x', padx=5, pady=(0, 5))

        self.metrics_text = tk.Text(metrics, height=8, bg='white', relief='sunken', bd=1)
        self.metrics_text.pack(fill='both', expand=True)

        # Step 3: Input passenger data
        inp = ttk.LabelFrame(frm, text='STEP 3: Enter Passenger Information', padding=15)
        inp.pack(fill='x', pady=(0, 15))

        input_info = tk.Text(inp, height=2, width=80, wrap='word')
        input_info.insert('1.0', 
            'Fill in the passenger details below. Fields marked with * are required. '
            'Use realistic values (e.g., Age: 25, Fare: 50)')
        input_info.config(state='disabled')
        input_info.pack(fill='x', padx=5, pady=(0, 10))

        grid = ttk.Frame(inp)
        grid.pack(fill='x')

        def add_row(r, label, widget, required=False):
            label_text = label + (' *' if required else '')
            ttk.Label(grid, text=label_text, font=('Arial', 10)).grid(row=r, column=0, sticky='w', padx=10, pady=8)
            widget.grid(row=r, column=1, sticky='w', padx=10, pady=8)

        self.var_pclass = tk.StringVar(value='3')
        self.var_sex = tk.StringVar(value='Male')
        self.var_age = tk.StringVar(value='25')
        self.var_sibsp = tk.StringVar(value='0')
        self.var_parch = tk.StringVar(value='0')
        self.var_fare = tk.StringVar(value='50')
        self.var_embarked = tk.StringVar(value='S')
        self.var_cabin = tk.StringVar(value='')
        self.var_name = tk.StringVar(value='John Doe')
        self.var_ticket = tk.StringVar(value='')
        self.var_passengerid = tk.StringVar(value='')

        add_row(0, 'Passenger Class (1/2/3)*', ttk.Combobox(grid, textvariable=self.var_pclass, values=['1', '2', '3'], width=15, state='readonly'), True)
        add_row(1, 'Gender*', ttk.Combobox(grid, textvariable=self.var_sex, values=['Male', 'Female'], width=15, state='readonly'), True)
        add_row(2, 'Age (years)*', ttk.Entry(grid, textvariable=self.var_age, width=15), True)
        add_row(3, 'Siblings/Spouse aboard', ttk.Spinbox(grid, from_=0, to=10, textvariable=self.var_sibsp, width=15))
        add_row(4, 'Parents/Children aboard', ttk.Spinbox(grid, from_=0, to=10, textvariable=self.var_parch, width=15))
        add_row(5, 'Ticket Fare ($)*', ttk.Entry(grid, textvariable=self.var_fare, width=15), True)
        add_row(6, 'Port Embarked (C/Q/S)*', ttk.Combobox(grid, textvariable=self.var_embarked, values=['C - Cherbourg', 'Q - Queenstown', 'S - Southampton'], width=30, state='readonly'), True)
        add_row(7, 'Cabin Number (optional)', ttk.Entry(grid, textvariable=self.var_cabin, width=30))
        add_row(8, 'Full Name (optional)', ttk.Entry(grid, textvariable=self.var_name, width=50))

        # Step 4: Prediction
        pred = ttk.LabelFrame(frm, text='STEP 4: Get Prediction', padding=15)
        pred.pack(fill='x', pady=(0, 15))

        self.pred_label = ttk.Label(pred, text='Fill in passenger information and click "Predict" to see the survival estimate.', 
                                    font=('Arial', 11), foreground='blue')
        self.pred_label.pack(anchor='w', pady=5)

        self.pred_btn = ttk.Button(pred, text='🔮 PREDICT SURVIVAL', command=self.on_predict)
        self.pred_btn.pack(anchor='w', pady=10)

        # Step 5: Feature importance
        fi = ttk.LabelFrame(frm, text='STEP 5: What Factors Matter Most?', padding=15)
        fi.pack(fill='both', expand=True, pady=0)

        fi_info = tk.Text(fi, height=2, width=80, wrap='word')
        fi_info.insert('1.0', 
            'These are the factors that the AI model found most important when deciding '
            'whether a passenger survived (higher numbers = more important).')
        fi_info.config(state='disabled')
        fi_info.pack(fill='x', padx=5, pady=(0, 5))

        self.fi_text = tk.Text(fi, height=8, bg='white', relief='sunken', bd=1)
        self.fi_text.pack(fill='both', expand=True)

    def _log(self, msg: str):
        self.metrics_text.insert('end', msg + '\n')
        self.metrics_text.see('end')
        self.update_idletasks()

    def on_train(self):
        try:
            self.metrics_text.delete('1.0', 'end')
            self.fi_text.delete('1.0', 'end')
            self.pred_label.config(text='Training in progress...')

            # Find the dataset using absolute path
            script_dir = Path(__file__).parent
            csv_path = script_dir / 'Titanic-Dataset.csv'
            
            if not csv_path.exists():
                messagebox.showerror('Dataset not found', f'Cannot find Titanic-Dataset.csv at {csv_path}\n\nMake sure the dataset file is in the same folder as this script.')
                self.pred_label.config(text='Training failed: Dataset not found.')
                return
                
            df = pd.read_csv(csv_path)
            self._log(f'✓ Loaded dataset ({len(df)} records)')
            self._log('⏳ Training... This may take a few minutes.')

            self.artifacts = _train_model(df, random_state=42)

            m = self.artifacts['metrics']
            self._log('\n' + '='*60)
            self._log('TRAINING COMPLETE!')
            self._log('='*60)
            self._log('\n📊 Best Hyperparameters:')
            self._log(str(self.artifacts['best_params']))
            self._log(f"\n✓ Accuracy: {m['accuracy']:.1%}")
            self._log(f"✓ ROC-AUC Score: {m['roc_auc']:.1%}")
            self._log('\n📋 Detailed Report:')
            self._log(m['report'])

            self.pred_label.config(text='✓ Model ready! Fill in passenger info and click "Predict"', foreground='green')
            self.dataset_label.config(text='✓ Training complete', foreground='green')

            top_imp = self.artifacts['importances'].head(10)
            self.fi_text.insert('end', top_imp.to_string())

        except Exception as e:
            messagebox.showerror('Training error', str(e))
            self.pred_label.config(text='Training failed.')

    def on_predict(self):
        if self.artifacts is None:
            messagebox.showwarning('Not trained', 'Please train the model first by clicking the "TRAIN MODEL" button.')
            return

        try:
            # Extract port code from full label
            embarked_str = self.var_embarked.get()
            embarked_code = embarked_str[0] if embarked_str else 'S'
            
            # Convert gender to lowercase for model
            sex_input = self.var_sex.get().strip().lower()

            passenger = {
                'Pclass': float(self.var_pclass.get()),
                'Sex': sex_input,
                'Age': float(self.var_age.get()),
                'SibSp': float(self.var_sibsp.get()),
                'Parch': float(self.var_parch.get()),
                'Fare': float(self.var_fare.get()),
                'Embarked': embarked_code,
                'Cabin': self.var_cabin.get().strip() or np.nan,
                'Name': self.var_name.get().strip(),
                'Ticket': self.var_ticket.get().strip() or np.nan,
                'PassengerId': float(self.var_passengerid.get()) if self.var_passengerid.get().strip() else np.nan,
                'Survived': 0,
            }

            X_scaled = _prepare_single_passenger(self.artifacts, passenger)
            model = self.artifacts['best_model']

            prob_survive = float(model.predict_proba(X_scaled)[0, 1])
            pred_class = int(model.predict(X_scaled)[0])

            result_text = f'{"✓ SURVIVED" if pred_class == 1 else "✗ DID NOT SURVIVE"}\n'
            result_text += f'Confidence: {prob_survive*100:.1f}%'
            
            self.pred_label.config(text=result_text, foreground='green' if pred_class == 1 else 'red',
                                  font=('Arial', 12, 'bold'))

        except ValueError as e:
            messagebox.showerror('Input Error', 'Please fill in all required fields with valid numbers.\n\nAge and Fare should be numbers, for example: 25 or 50.5')
        except Exception as e:
            messagebox.showerror('Prediction error', f'Error: {str(e)}')


if __name__ == '__main__':
    app = TitanicGUI()
    app.mainloop()

