# gui/app.py

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import time
import logging
import pandas as pd
import csv
import pandas as pd

from gui.dialogs import PreprocessingOptionsDialog
from ml.data_processing import preprocess_data, preprocess_new_data
from ml.model_training import train_model_thread, compare_models_thread
from ml.model_evaluation import evaluate_model_thread
from visualization.plotting import visualize_data
from utils.utils import create_tooltip, update_progress

class MLModelBuilderApp:
    def __init__(self, root):
        """
        Initialize the MLModelBuilderApp with default settings and create the UI.
        """
        self.root = root
        self.root.title("Advanced Machine Learning Model Builder")
        self.root.geometry("1200x800")

        ctk.set_appearance_mode("dark")

        # Initialize attributes
        self.model = None
        self.data = None
        self.X = None
        self.y = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.problem_type = None
        self.task_in_progress = False
        self.task_start_time = None
        self.best_params = None
        self.last_evaluation_metrics = None
        self.preprocessing_objects = {}

        # Create the UI
        self.create_ui()

    def create_ui(self):
        """
        Create the user interface components.
        """
        # File Import Section
        self.file_frame = ctk.CTkFrame(self.root)
        self.file_frame.pack(pady=10, padx=10, fill="x")

        self.import_button = ctk.CTkButton(self.file_frame, text="Import Dataset", command=self.import_dataset)
        self.import_button.pack(side="left", padx=20, pady=5)
        create_tooltip(self.import_button, "Import a CSV dataset.")

        # Target Variable Selection
        self.target_var_label = ctk.CTkLabel(self.file_frame, text="Select Target Variable:")
        self.target_var_label.pack(side="left", padx=10)

        self.target_var = tk.StringVar()
        self.target_var_menu = ctk.CTkOptionMenu(self.file_frame, variable=self.target_var, values=[])
        self.target_var_menu.pack(side="left", padx=10)

        self.target_var.trace('w', self.on_target_var_change)

        # Model Selection Section
        self.model_frame = ctk.CTkFrame(self.root)
        self.model_frame.pack(pady=10, padx=10, fill="x")

        self.model_label = ctk.CTkLabel(self.model_frame, text="Select Model:")
        self.model_label.pack(side="left", padx=10)

        self.model_var = tk.StringVar(value="")
        self.model_menu = ctk.CTkOptionMenu(self.model_frame, variable=self.model_var, values=[])
        self.model_menu.pack(side="left", padx=10)

        self.param_button = ctk.CTkButton(self.model_frame, text="Set Parameters", command=self.set_parameters)
        self.param_button.pack(side="left", padx=10)
        create_tooltip(self.param_button, "Set model parameters and hyperparameter tuning options.")

        # Training and Evaluation Section
        self.action_frame = ctk.CTkFrame(self.root)
        self.action_frame.pack(pady=10, padx=10, fill="x")

        self.train_button = ctk.CTkButton(self.action_frame, text="Train Model", command=self.train_model)
        self.train_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.train_button, "Train the model with the current settings.")

        self.evaluate_button = ctk.CTkButton(self.action_frame, text="Evaluate Model", command=self.evaluate_model)
        self.evaluate_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.evaluate_button, "Evaluate the trained model.")

        self.compare_button = ctk.CTkButton(self.action_frame, text="Compare Models", command=self.compare_models)
        self.compare_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.compare_button, "Compare different models using cross-validation.")

        # Visualization Section
        self.visualization_frame = ctk.CTkFrame(self.root)
        self.visualization_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.visualize_button = ctk.CTkButton(self.visualization_frame, text="Visualize Dataset", command=self.visualize_data)
        self.visualize_button.pack(side="top", pady=10)
        create_tooltip(self.visualize_button, "Visualize the dataset or selected variables.")

        # Variable Selection for Visualization
        self.visual_vars_label = ctk.CTkLabel(self.visualization_frame, text="Select Variables for Visualization:")
        self.visual_vars_label.pack(side="top", padx=10)

        self.visual_vars = tk.StringVar()
        self.visual_vars_menu = ctk.CTkOptionMenu(self.visualization_frame, variable=self.visual_vars, values=[])
        self.visual_vars_menu.pack(side="top", padx=10)

        # Dataset Information Section
        self.dataset_info_frame = ctk.CTkFrame(self.visualization_frame)
        self.dataset_info_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.dataset_info_text = tk.Text(self.dataset_info_frame, height=15, wrap="word")
        self.dataset_info_text.pack(fill="both", padx=10, pady=10)

        # Advanced Feature Buttons
        self.advanced_frame = ctk.CTkFrame(self.root)
        self.advanced_frame.pack(pady=10, padx=10, fill="x")

        self.predict_button = ctk.CTkButton(self.advanced_frame, text="Predict on New Data", command=self.predict_new_data)
        self.predict_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.predict_button, "Use the trained model to predict on new data.")

        self.cv_visualize_button = ctk.CTkButton(self.advanced_frame, text="Visualize CV Scores", command=self.visualize_cv_scores)
        self.cv_visualize_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.cv_visualize_button, "Visualize cross-validation scores.")

        self.feature_engineer_button = ctk.CTkButton(self.advanced_frame, text="Feature Engineering", command=self.feature_engineering)
        self.feature_engineer_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.feature_engineer_button, "Apply feature engineering techniques.")

        self.export_report_button = ctk.CTkButton(self.advanced_frame, text="Export Evaluation Report", command=self.export_evaluation_report)
        self.export_report_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.export_report_button, "Export the evaluation report to a CSV file.")

        self.save_model_button = ctk.CTkButton(self.advanced_frame, text="Save Model", command=self.save_model)
        self.save_model_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.save_model_button, "Save the trained model and preprocessing steps.")

        self.load_model_button = ctk.CTkButton(self.advanced_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side="left", padx=10, pady=10)
        create_tooltip(self.load_model_button, "Load a saved model and preprocessing steps.")

        # Status Text Box and Progress Bar
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.pack(pady=10, padx=10, fill="x")

        self.status_text = tk.Text(self.status_frame, height=8, wrap="word")
        self.status_text.pack(fill="both", padx=10, pady=(10, 5))

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(pady=(0, 10))
        self.progress_label = ctk.CTkLabel(self.status_frame, text="Progress: 0%")
        self.progress_label.pack()

    # GUI Event Methods

    def import_dataset(self):
        """
        Import a dataset from a CSV file.
        """
        try:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                with open(file_path, 'r') as f:
                    # Use csv.Sniffer to detect the delimiter
                    sample = f.read(2048)
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter

                # Now load the dataset with the detected delimiter
                self.data = pd.read_csv(file_path, delimiter=delimiter)

                if not self.data.empty:
                    # Efficient Data Handling: Sample large datasets
                    if self.data.shape[0] > 10000:
                        result = messagebox.askyesno("Large Dataset", "The dataset is large. Would you like to sample 10,000 rows?")
                        if result:
                            self.data = self.data.sample(n=10000, random_state=42)
                            self.status_text.insert("end", "Dataset sampled to 10,000 rows.\n")

                    # Update target variable selection
                    columns = list(self.data.columns)
                    self.target_var_menu.configure(values=columns)
                    self.target_var.set(columns[-1])  # Default to last column

                    # Update visualization variable selection
                    self.visual_vars_menu.configure(values=["All"] + columns)
                    self.visual_vars.set("All")

                    self.status_text.insert("end", f"Dataset loaded successfully. Shape: {self.data.shape}\n")

                    # Update dataset info display
                    self.update_dataset_info()
                else:
                    self.status_text.insert("end", "Failed to load dataset. The file may be empty or corrupt.\n")
        except Exception as e:
            self.status_text.insert("end", f"Error loading dataset: {e}\n")
            logging.error("Error loading dataset", exc_info=True)

    def update_dataset_info(self):
        """
        Update the dataset information displayed in the GUI.
        """
        if self.data is not None:
            info_text = f"Dataset Shape: {self.data.shape}\n"
            info_text += f"Missing Values:\n{self.data.isnull().sum()}\n"
            info_text += f"Basic Statistics:\n{self.data.describe()}\n"
            self.dataset_info_text.delete("1.0", tk.END)  # Clear previous text
            self.dataset_info_text.insert("end", info_text)

    def on_target_var_change(self, *args):
        """
        Callback function when the target variable selection changes.
        """
        self.determine_problem_type()
        self.update_model_options()

    def determine_problem_type(self):
        """
        Determine whether the problem is regression or classification based on the target variable.
        """
        target_column = self.target_var.get()
        if target_column and self.data is not None:
            self.y = self.data[target_column]
            if pd.api.types.is_numeric_dtype(self.y):
                self.problem_type = 'regression'
            else:
                self.problem_type = 'classification'
                self.y = self.y.astype(str)
            self.status_text.insert("end", f"Problem type determined: {self.problem_type}\n")

            # Check class distribution
            if self.problem_type == 'classification':
                class_counts = self.y.value_counts()
                self.status_text.insert("end", f"Class distribution:\n{class_counts}\n")
        else:
            self.problem_type = None

    def update_model_options(self):
        """
        Update the model options based on the problem type.
        """
        if self.problem_type == 'classification':
            model_options = ["Logistic Regression", "Random Forest", "Support Vector Machine",
                             "Gradient Boosting", "K-Nearest Neighbors", "AutoML"]
        elif self.problem_type == 'regression':
            model_options = ["Linear Regression", "Random Forest Regressor", "Support Vector Regressor",
                             "Gradient Boosting Regressor", "K-Nearest Neighbors Regressor", "AutoML"]
        else:
            model_options = []

        if model_options:
            self.model_menu.configure(values=model_options)
            self.model_var.set(model_options[0])  # Set default model
        else:
            self.model_menu.configure(values=[])
            self.model_var.set('')

    def set_parameters(self):
        """
        Set model parameters and hyperparameter tuning options.
        """
        model_name = self.model_var.get()
        param_grid = None
        if model_name == "Logistic Regression":
            self.model = 'Logistic Regression'
            param_grid = {'C': [0.1, 1, 10]}
        elif model_name == "Random Forest":
            self.model = 'Random Forest'
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        elif model_name == "Support Vector Machine":
            self.model = 'Support Vector Machine'
            param_grid = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
        elif model_name == "Gradient Boosting":
            self.model = 'Gradient Boosting'
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        elif model_name == "K-Nearest Neighbors":
            self.model = 'K-Nearest Neighbors'
            param_grid = {'n_neighbors': [3, 5]}
        elif model_name == "Linear Regression":
            self.model = 'Linear Regression'
            param_grid = {}
        elif model_name == "Random Forest Regressor":
            self.model = 'Random Forest Regressor'
            param_grid = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        elif model_name == "Support Vector Regressor":
            self.model = 'Support Vector Regressor'
            param_grid = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
        elif model_name == "Gradient Boosting Regressor":
            self.model = 'Gradient Boosting Regressor'
            param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]}
        elif model_name == "K-Nearest Neighbors Regressor":
            self.model = 'K-Nearest Neighbors Regressor'
            param_grid = {'n_neighbors': [3, 5]}
        elif model_name == "AutoML":
            self.model = 'AutoML'
            param_grid = None

        # Save the parameter grid for later use
        self.param_grid = param_grid

        self.status_text.insert("end", f"Parameters set for {model_name}.\n")

    def train_model(self):
        """
        Train the model using the preprocessed data.
        """
        if self.data is not None:
            if self.task_in_progress:
                messagebox.showwarning("Task in Progress", "Please wait for the current task to complete.")
                return

            # Collect preprocessing options before starting the training thread
            options_dialog = PreprocessingOptionsDialog(self.root)
            self.root.wait_window(options_dialog)

            if not options_dialog.result:
                self.status_text.insert("end", "Training canceled by user.\n")
                return

            # Get options from the dialog
            options = {
                'strategy': options_dialog.strategy.get(),
                'fill_value': options_dialog.fill_value.get() if options_dialog.strategy.get() == "constant" else None,
                'encoding_method': options_dialog.encoding_method.get(),
                'outlier_method': options_dialog.outlier_method.get(),
                'scaling_method': options_dialog.scaling_method.get(),
                'apply_poly': options_dialog.apply_poly.get(),
                'degree': options_dialog.degree.get() if options_dialog.apply_poly.get() else None,
                'apply_feature_selection': options_dialog.apply_feature_selection.get(),
                'feature_selection_method': options_dialog.feature_selection_method.get() if options_dialog.apply_feature_selection.get() else None,
                'threshold': options_dialog.threshold.get() if options_dialog.feature_selection_method.get() == "Correlation Threshold" else None,
                'n_features': options_dialog.n_features.get() if options_dialog.feature_selection_method.get() == "Recursive Feature Elimination" else None,
                'param_grid': self.param_grid,
                'model_name': self.model_var.get(),
            }

            # Disable buttons
            self.task_in_progress = True
            self.disable_buttons()

            # Start training thread
            threading.Thread(target=train_model_thread, args=(self, options)).start()
        else:
            messagebox.showwarning("No Data", "Please import a dataset first.")

    def evaluate_model(self):
        """
        Evaluate the trained model and display performance metrics and visualizations.
        """
        if self.task_in_progress:
            messagebox.showwarning("Task in Progress", "Please wait for the current task to complete.")
            return

        if self.model and self.X_test is not None and self.y_test is not None:
            # Disable buttons
            self.task_in_progress = True
            self.disable_buttons()

            threading.Thread(target=evaluate_model_thread, args=(self,)).start()
        else:
            messagebox.showwarning("No Model", "Please train a model first.")

    def compare_models(self):
        """
        Compare different models using cross-validation.
        """
        if self.task_in_progress:
            messagebox.showwarning("Task in Progress", "Please wait for the current task to complete.")
            return

        if self.X is None or self.y is None:
            messagebox.showwarning("No Data", "Please import and preprocess a dataset first.")
            return

        # Disable buttons
        self.task_in_progress = True
        self.disable_buttons()

        threading.Thread(target=compare_models_thread, args=(self,)).start()

    def visualize_data(self):
        """
        Visualize the dataset or selected variables.
        """
        if self.data is not None:
            selected_vars = self.visual_vars.get()
            threading.Thread(target=visualize_data, args=(self.data, selected_vars, self.status_text)).start()
        else:
            messagebox.showwarning("No Data", "Please import a dataset first.")

    def predict_new_data(self):
        """
        Use the trained model to predict on new data.
        """
        if self.model:
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if file_path:
                try:
                    new_data = pd.read_csv(file_path)
                    new_data_processed = preprocess_new_data(new_data, self.preprocessing_objects)
                    predictions = self.model.predict(new_data_processed)
                    result = pd.DataFrame(predictions, columns=["Predictions"])
                    save_path = filedialog.asksaveasfilename(defaultextension=".csv",
                                                             filetypes=[("CSV files", "*.csv")])
                    if save_path:
                        result.to_csv(save_path, index=False)
                        self.status_text.insert("end", f"Predictions saved to {save_path}\n")
                except Exception as e:
                    self.status_text.insert("end", f"Error predicting new data: {e}\n")
                    logging.error("Error predicting new data", exc_info=True)
        else:
            messagebox.showwarning("No Model", "Please train or load a model first.")

    def feature_engineering(self):
        """
        Apply feature engineering techniques such as polynomial features and scaling options.
        """
        # For simplicity, let's just display a message
        messagebox.showinfo("Feature Engineering", "Feature Engineering options are applied during training.")

    def export_evaluation_report(self):
        """
        Export the evaluation report to a CSV file.
        """
        if self.last_evaluation_metrics:
            try:
                report_df = pd.DataFrame([self.last_evaluation_metrics])
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if file_path:
                    report_df.to_csv(file_path, index=False)
                    self.status_text.insert("end", "Evaluation report exported successfully.\n")
            except Exception as e:
                self.status_text.insert("end", f"Error exporting evaluation report: {e}\n")
                logging.error("Error exporting evaluation report", exc_info=True)
        else:
            messagebox.showwarning("No Evaluation", "Please evaluate a model first.")

    def save_model(self):
        """
        Save the trained model and preprocessing steps to a file.
        """
        if self.model:
            try:
                file_path = filedialog.asksaveasfilename(defaultextension=".joblib",
                                                         filetypes=[("Joblib files", "*.joblib")])
                if file_path:
                    import joblib
                    model_data = {
                        'model': self.model,
                        'preprocessing_objects': self.preprocessing_objects,
                        'problem_type': self.problem_type,
                        'best_params': self.best_params,
                        'metrics': self.last_evaluation_metrics
                    }
                    joblib.dump(model_data, file_path)
                    self.status_text.insert("end", "Model and metadata saved successfully.\n")
            except Exception as e:
                self.status_text.insert("end", f"Error saving model: {e}\n")
                logging.error("Error saving model", exc_info=True)
        else:
            messagebox.showwarning("No Model", "Please train a model first.")

    def load_model(self):
        """
        Load a saved model and preprocessing steps from a file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Joblib files", "*.joblib")])
        if file_path:
            try:
                import joblib
                saved_objects = joblib.load(file_path)
                self.model = saved_objects['model']
                self.preprocessing_objects = saved_objects['preprocessing_objects']
                self.problem_type = saved_objects['problem_type']
                self.best_params = saved_objects.get('best_params', None)
                self.last_evaluation_metrics = saved_objects.get('metrics', None)
                self.status_text.insert("end", "Model and preprocessing steps loaded successfully.\n")
            except Exception as e:
                self.status_text.insert("end", f"Error loading model: {e}\n")
                logging.error("Error loading model", exc_info=True)
        else:
            messagebox.showwarning("No Model", "Please select a valid model file.")

    def visualize_cv_scores(self):
        """
        Visualize cross-validation scores for the selected model.
        """
        # For simplicity, we'll display a message
        messagebox.showinfo("Visualize CV Scores", "Cross-validation scores visualization not implemented.")

    def disable_buttons(self):
        """
        Disable buttons during a long-running task.
        """
        self.train_button.configure(state='disabled')
        self.evaluate_button.configure(state='disabled')
        self.compare_button.configure(state='disabled')
        self.param_button.configure(state='disabled')
        self.predict_button.configure(state='disabled')
        self.save_model_button.configure(state='disabled')
        self.load_model_button.configure(state='disabled')

    def enable_buttons(self):
        """
        Enable buttons after a long-running task.
        """
        self.train_button.configure(state='normal')
        self.evaluate_button.configure(state='normal')
        self.compare_button.configure(state='normal')
        self.param_button.configure(state='normal')
        self.predict_button.configure(state='normal')
        self.save_model_button.configure(state='normal')
        self.load_model_button.configure(state='normal')

