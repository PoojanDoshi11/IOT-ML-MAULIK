
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
import joblib  # For saving models
from sklearn.model_selection import StratifiedKFold, cross_val_score

class AttackPredictorApp:
    def __init__(self, root):
        """
        Initializes the application, sets up UI components, and prepares placeholders for data and models.
        """
        self.root = root
        self.root.title("Attack Type Prediction")
        self.data = None  # Placeholder for the dataset
        self.models = {}  # Dictionary to store trained models
        self.accuracies = {}  # Dictionary to store model accuracies
        self.label_encoder = LabelEncoder()  # For encoding categorical target variable
        self.scaler = StandardScaler()  # For feature scaling
        self.pca = PCA(n_components=5)  # Applying PCA to reduce dimensions to 5
        self.best_model = None  # Variable to store the best model
        self.setup_ui()  # Set up the user interface

    def setup_ui(self):
        """Sets up the user interface (UI) using Tkinter."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Data loading section
        ttk.Label(main_frame, text="1. Load Dataset").grid(row=0, column=0, sticky=tk.W)
        ttk.Button(main_frame, text="Browse CSV", command=self.load_data).grid(row=1, column=0, pady=5)

        # Input section for entering feature values
        ttk.Label(main_frame, text="2. Enter Features (comma-separated):").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.input_entry = ttk.Entry(main_frame, width=80)
        self.input_entry.grid(row=3, column=0, pady=5)
        ttk.Button(main_frame, text="Predict", command=self.predict_attack).grid(row=4, column=0, pady=5)

        # Results section to display predictions
        ttk.Label(main_frame, text="3. Predictions:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.result_text = tk.Text(main_frame, height=10, width=80)
        self.result_text.grid(row=6, column=0, pady=5)

        # Model accuracies section
        ttk.Label(main_frame, text="Model Accuracies:").grid(row=7, column=0, sticky=tk.W, pady=(10, 0))
        self.accuracy_text = tk.Text(main_frame, height=8, width=80)
        self.accuracy_text.grid(row=8, column=0, pady=5)

    def load_data(self):
        """Loads the dataset, preprocesses it, and trains models."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            self.data = pd.read_csv(file_path)
            self.preprocess_data()
            self.train_models()
            self.show_accuracies()
            messagebox.showinfo("Success", "Dataset loaded and models trained successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def show_accuracies(self):
        """Displays the accuracies of the trained models."""
        self.accuracy_text.delete(1.0, tk.END)
        for name, acc in self.accuracies.items():
            self.accuracy_text.insert(tk.END, f"{name}: {acc:.4f}\n")

    def preprocess_data(self):
        """Preprocesses the dataset by handling missing values, encoding categorical columns, scaling, and applying PCA."""
        if self.data is None:
            return

        # Handle missing values
        self.data.fillna(self.data.mean(), inplace=True)

        # Remove non-numeric columns except the target 'Attack_type'
        target_column = "Attack_type"
        non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
        non_numeric_cols = [col for col in non_numeric_cols if col != target_column]
        self.data.drop(columns=non_numeric_cols, inplace=True, errors='ignore')

        # Encode target column
        if target_column not in self.data.columns:
            messagebox.showerror("Error", f"Target column '{target_column}' not found in dataset.")
            return
        self.data[target_column] = self.label_encoder.fit_transform(self.data[target_column])

        # Prepare features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Apply PCA to reduce dimensions
        X_pca = self.pca.fit_transform(X_scaled)

        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)


    def train_models(self):
        """Trains multiple machine learning models using K-Fold Cross-Validation and tracks the highest score and fold number."""
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Machine": SVC(probability=True),
            "k-Nearest Neighbors": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Naïve Bayes": GaussianNB(),
            "XGBoost": xgb.XGBClassifier(),
            "LightGBM": lgb.LGBMClassifier(force_col_wise=True)
        }

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold Cross-Validation

        self.fold_scores = {}  # Store fold-wise scores for each model

        for name, model in self.models.items():
            scores = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring='accuracy')
            self.accuracies[name] = np.mean(scores)  # Store mean accuracy

            # Track highest accuracy and the fold where it occurred
            max_score = np.max(scores)
            max_score_fold = np.argmax(scores)  # Fold index starts from 0, so add 1
            print(max_score, max_score_fold)

            self.fold_scores[name] = (max_score, max_score_fold)

        # Identify the best model based on average accuracy
        best_model_name = max(self.accuracies, key=self.accuracies.get)
        self.best_model = self.models[best_model_name]

        # Train the best model on the full training set
        self.best_model.fit(self.X_train, self.y_train)

        # Save the best model
        joblib.dump(self.best_model, 'best_model.pkl')

        # Display fold-wise maximum accuracy scores
        self.show_fold_accuracies()

    def predict_attack(self):
        """Predicts the attack type based on user input."""
        if self.data is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return

        try:
            # Get user input as a comma-separated string
            input_str = self.input_entry.get()
            input_data = [float(x.strip()) for x in input_str.split(",")]

            # Ensure user provides the correct number of features (85 before PCA)
            expected_features = self.scaler.n_features_in_  # Should be 85
            if len(input_data) != expected_features:
                messagebox.showerror("Error", f"Invalid input length. Expected {expected_features} features.")
                return

            # Scale the input data
            scaled_input = self.scaler.transform([input_data])

            # Apply PCA transformation
            pca_input = self.pca.transform(scaled_input)  # This reduces to 5 components

            # Make prediction
            pred = self.best_model.predict(pca_input)
            result = self.label_encoder.inverse_transform(pred)[0]

            # Display result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Best Model Prediction: {result}\n")
        
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")

    
    def show_fold_accuracies(self):
        """Displays the maximum K-Fold accuracy and the fold number where it occurred."""
        self.accuracy_text.delete(1.0, tk.END)
        self.accuracy_text.insert(tk.END, "Model Accuracies (Mean Accuracy | Max Fold Score & Fold No.):\n\n")
        
        for name, acc in self.accuracies.items():
            max_score, max_fold = self.fold_scores[name]
            self.accuracy_text.insert(tk.END, f"{name}: {acc:.4f} | Max: {max_score:.4f} (Fold {max_fold})\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = AttackPredictorApp(root)
    root.mainloop()
