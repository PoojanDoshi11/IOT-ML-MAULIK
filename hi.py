# # import pandas as pd
# # import numpy as np
# # import tkinter as tk
# # from tkinter import ttk, filedialog, messagebox
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.preprocessing import StandardScaler, LabelEncoder
# # from sklearn.metrics import accuracy_score
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.svm import SVC
# # from sklearn.neighbors import KNeighborsClassifier
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.naive_bayes import GaussianNB
# # import xgboost as xgb
# # import lightgbm as lgb
# # import joblib  # Import joblib for saving models

# # # GUI Application class for Attack Type Prediction
# # class AttackPredictorApp:
# #     def __init__(self, root):
# #         """
# #         Initializes the main application with the given root window.
# #         Sets up the data, models, and other necessary parameters for the prediction process.
# #         """
# #         self.root = root
# #         self.root.title("Attack Type Prediction")  # Set the window title
# #         self.data = None  # Placeholder for the dataset
# #         self.models = {}  # Dictionary to store machine learning models
# #         self.tuned_models = {}  # Dictionary to store models with hyperparameter tuning
# #         self.accuracies = {}  # Dictionary to store model accuracies
# #         self.best_params = {}  # Dictionary to store the best parameters from tuning
# #         self.label_encoder = LabelEncoder()  # For encoding the target variable
# #         self.scaler = StandardScaler()  # For feature scaling
# #         self.best_model = None  # Variable to store the best model
# #         self.setup_ui()  # Call to set up the user interface

# #     def setup_ui(self):
# #         """
# #         Sets up the user interface (UI) for the application using Tkinter.
# #         """
# #         # Create main frame
# #         main_frame = ttk.Frame(self.root, padding="10")
# #         main_frame.pack(fill=tk.BOTH, expand=True)

# #         # Data loading section
# #         ttk.Label(main_frame, text="1. Load Dataset").grid(row=0, column=0, sticky=tk.W)
# #         ttk.Button(main_frame, text="Browse CSV", command=self.load_data).grid(row=1, column=0, pady=5)

# #         # Hyperparameter tuning checkbox
# #         self.tune_var = tk.BooleanVar(value=False)
# #         ttk.Checkbutton(main_frame, text="Enable Hyperparameter Tuning", variable=self.tune_var).grid(row=2, column=0, pady=5)

# #         # Input section for entering feature values
# #         ttk.Label(main_frame, text="2. Enter Features (comma-separated):").grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
# #         self.input_entry = ttk.Entry(main_frame, width=80)
# #         self.input_entry.grid(row=4, column=0, pady=5)
# #         ttk.Button(main_frame, text="Predict", command=self.predict_attack).grid(row=5, column=0, pady=5)

# #         # Results section to display predictions
# #         ttk.Label(main_frame, text="3. Predictions:").grid(row=6, column=0, sticky=tk.W, pady=(10, 0))
# #         self.result_text = tk.Text(main_frame, height=10, width=80)
# #         self.result_text.grid(row=7, column=0, pady=5)

# #         # Model accuracies section
# #         ttk.Label(main_frame, text="Model Accuracies:").grid(row=8, column=0, sticky=tk.W, pady=(10, 0))
# #         self.accuracy_text = tk.Text(main_frame, height=8, width=80)
# #         self.accuracy_text.grid(row=9, column=0, pady=5)

# #         # Display models used
# #         ttk.Label(main_frame, text="Models Used:").grid(row=10, column=0, sticky=tk.W, pady=(10, 0))
# #         self.models_text = tk.Text(main_frame, height=8, width=80)
# #         self.models_text.grid(row=11, column=0, pady=5)

# #         # Hyperparameter tuning status
# #         self.tuning_status = ttk.Label(main_frame, text="Hyperparameter Tuning: Disabled", foreground="blue")
# #         self.tuning_status.grid(row=12, column=0, sticky=tk.W, pady=(5, 0))

# #         # Best model status
# #         self.best_model_status = ttk.Label(main_frame, text="Best Model: None", foreground="green")
# #         self.best_model_status.grid(row=13, column=0, sticky=tk.W, pady=(5, 0))

# #     def load_data(self):
# #         """
# #         Opens a file dialog to load the dataset (CSV format), preprocesses it,
# #         and trains models on the data.
# #         """
# #         # Open file dialog to select a CSV file
# #         file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
# #         if not file_path:
# #             return  # If no file is selected, exit the function

# #         try:
# #             # Load dataset into a pandas DataFrame
# #             self.data = pd.read_csv(file_path)
# #             print("Dataset loaded successfully!")
# #             print(f"Dataset shape: {self.data.shape}")
# #             self.preprocess_data()  # Preprocess the data before training models
# #             self.train_models()  # Train machine learning models on the dataset
# #             self.show_accuracies()  # Show the accuracies of the models
# #             self.show_models()  # Show which models were used
# #             tuning_status = "Enabled" if self.tune_var.get() else "Disabled"
# #             self.tuning_status.config(text=f"Hyperparameter Tuning: {tuning_status}")
# #             messagebox.showinfo("Success", "Dataset loaded and models trained successfully!")
# #         except Exception as e:
# #             messagebox.showerror("Error", f"Failed to load data: {str(e)}")

# #     def preprocess_data(self):
# #         """
# #         Preprocesses the dataset, handles missing values, encodes categorical columns,
# #         and splits the data into training and testing sets.
# #         """
# #         if self.data is None:
# #             print("No data to preprocess.")
# #             return

# #         print("Preprocessing data...")

# #         # Handle missing values by filling with the mean of each column
# #         self.data.fillna(self.data.mean(), inplace=True)
# #         print("Missing values filled with mean.")

# #         # Remove non-numeric columns except the target 'Attack_type'
# #         non_numeric_cols = self.data.select_dtypes(exclude=[np.number]).columns
# #         non_numeric_cols = [col for col in non_numeric_cols if col != 'Attack_type']

# #         if len(non_numeric_cols) > 0:
# #             print(f"Removing non-numeric columns: {non_numeric_cols}")
# #             self.data = self.data.drop(columns=non_numeric_cols)

# #         # Encode target column 'Attack_type'
# #         target_column = "Attack_type"
# #         if target_column not in self.data.columns:
# #             print(f"Error: '{target_column}' column not found.")
# #             messagebox.showerror("Error", f"Target column '{target_column}' not found in the dataset.")
# #             return

# #         # Label encode the 'Attack_type' column
# #         self.data[target_column] = self.label_encoder.fit_transform(self.data[target_column])
        
# #         # Prepare feature variables (X) and target variable (y)
# #         X = self.data.drop(columns=[target_column])
# #         y = self.data[target_column]

# #         # Scale the feature variables
# #         X_scaled = self.scaler.fit_transform(X)

# #         # Split the dataset into training and testing sets
# #         self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
# #             X_scaled, y, test_size=0.2, random_state=42
# #         )
# #         print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")

# #     def train_models(self):
# #         """
# #         Initializes various machine learning models, trains them, and performs hyperparameter tuning (if enabled).
# #         """
# #         # Define the models to be used, including boosting models
# #         self.models = {
# #             "Logistic Regression": LogisticRegression(max_iter=1000),
# #             "Support Vector Machine": SVC(probability=True),
# #             "k-Nearest Neighbors": KNeighborsClassifier(),
# #             "Decision Tree": DecisionTreeClassifier(),
# #             "Random Forest": RandomForestClassifier(),
# #             "Naïve Bayes": GaussianNB(),
# #             "XGBoost": xgb.XGBClassifier(),
# #             "LightGBM": lgb.LGBMClassifier()
# #         }

# #         # Hyperparameter grids for tuning, including grids for boosting models
# #         self.param_grids = {
# #             "Logistic Regression": {"C": [0.1, 1, 10], "solver": ["liblinear", "lbfgs"]},
# #             "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
# #             "k-Nearest Neighbors": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
# #             "Decision Tree": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]},
# #             "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
# #             "XGBoost": {"learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]},
# #             "LightGBM": {"learning_rate": [0.01, 0.1, 0.3], "n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]}
# #         }

# #         # Train each model
# #         for name, model in self.models.items():
# #             try:
# #                 if self.tune_var.get() and name in self.param_grids:
# #                     print(f"Tuning {name}...")
# #                     # Perform hyperparameter tuning with GridSearchCV
# #                     grid_search = GridSearchCV(model, self.param_grids[name], cv=3, scoring="accuracy")
# #                     grid_search.fit(self.X_train, self.y_train)
# #                     self.tuned_models[name] = grid_search.best_estimator_
# #                     self.accuracies[name] = grid_search.best_score_
# #                     self.best_params[name] = grid_search.best_params_
# #                     print(f"Best parameters for {name}: {grid_search.best_params_}")
# #                 else:
# #                     # Train the model without tuning
# #                     model.fit(self.X_train, self.y_train)
# #                     self.accuracies[name] = accuracy_score(self.y_test, model.predict(self.X_test))
# #                 print(f"{name} trained successfully!")
# #             except Exception as e:
# #                 print(f"Error training {name}: {str(e)}")

# #         # Identify the best model based on accuracy
# #         best_model_name = max(self.accuracies, key=self.accuracies.get)
# #         self.best_model = self.models[best_model_name] if best_model_name not in self.tuned_models else self.tuned_models[best_model_name]

# #         # Save the best model using joblib
# #         joblib.dump(self.best_model, 'best_model.pkl')

# #         # Update the UI with the best model
# #         self.best_model_status.config(text=f"Best Model: {best_model_name}")
# #         print(f"Best model: {best_model_name} has been saved.")

# #     def show_accuracies(self):
# #         """
# #         Displays the accuracies of the trained models, including tuning information if applicable.
# #         """
# #         self.accuracy_text.delete(1.0, tk.END)
# #         for name, acc in self.accuracies.items():
# #             tuning_info = f" (Tuned)" if name in self.best_params else ""
# #             self.accuracy_text.insert(tk.END, f"{name}{tuning_info}: {acc:.4f}\n")
# #             if name in self.best_params:
# #                 self.accuracy_text.insert(tk.END, f"Best Params: {self.best_params[name]}\n")

# #     def show_models(self):
# #         """
# #         Displays the names of the models used in the training process.
# #         """
# #         self.models_text.delete(1.0, tk.END)
# #         for name in self.models.keys():
# #             self.models_text.insert(tk.END, f"{name}\n")

# #     def predict_attack(self):
# #         """
# #         Takes user input for prediction, processes it, and displays the predicted attack type for each model.
# #         """
# #         if self.data is None:
# #             messagebox.showwarning("Warning", "Please load a dataset first!")
# #             return

# #         try:
# #             # Get user input and convert it to a list of floats
# #             input_str = self.input_entry.get()
# #             input_data = [float(x.strip()) for x in input_str.split(",")]

# #             if len(input_data) != self.X_train.shape[1]:
# #                 messagebox.showerror("Error", f"Invalid input length. Expected {self.X_train.shape[1]} features.")
# #                 return

# #             # Scale the input data to match the training set scale
# #             scaled_input = self.scaler.transform(np.array(input_data).reshape(1, -1))

# #             # Make predictions using all models
# #             predictions = {}
# #             for name, model in self.models.items():
# #                 if self.tune_var.get() and name in self.tuned_models:
# #                     pred = self.tuned_models[name].predict(scaled_input)
# #                 else:
# #                     pred = model.predict(scaled_input)
# #                 predictions[name] = self.label_encoder.inverse_transform(pred)[0]

# #             # Display the predictions in the results section
# #             self.result_text.delete(1.0, tk.END)
# #             for model, pred in predictions.items():
# #                 self.result_text.insert(tk.END, f"{model}: {pred}\n")

# #         except ValueError:
# #             messagebox.showerror("Error", "Invalid input format. Please enter comma-separated numbers.")
# #         except Exception as e:
# #             messagebox.showerror("Error", f"Prediction failed: {str(e)}")

# # # Main program execution starts here
# # if __name__ == "__main__":
# #     root = tk.Tk()
# #     app = AttackPredictorApp(root)  # Initialize the application
# #     root.mainloop()  # Run the Tkinter event loop
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
