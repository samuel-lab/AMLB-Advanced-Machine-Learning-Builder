# ml/model_training.py


import threading
from sklearn.model_selection import (train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, KFold)
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, SGDRegressor
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import logging
import numpy as np
import pandas as pd
import time
from collections import Counter

from ml.data_processing import preprocess_data
from utils.utils import update_progress

def train_model_thread(app_instance, options):
    """
    Perform model training in a separate thread with memory optimizations.
    """
    try:
        app_instance.root.after(0, lambda: [
            update_progress(app_instance, 0, 1, time.time()),
            app_instance.status_text.insert("end", "Training started. This may take some time...\n"),
            app_instance.progress_bar.configure(mode='indeterminate'),
            app_instance.progress_bar.start()
        ])
        app_instance.task_start_time = time.time()

        # Prepare data
        target_column = app_instance.target_var.get()
        if target_column not in app_instance.data.columns:
            app_instance.status_text.insert("end", "Selected target variable is not in the dataset.\n")
            return

        # Separate target and features
        X = app_instance.data.drop(columns=[target_column])
        y = app_instance.data[target_column]

        # Preprocess data with Dask, including encoding categorical columns
        X_processed, y_processed, preprocessing_objects = preprocess_data(X, y, options)
        app_instance.preprocessing_objects = preprocessing_objects

        # Filter out classes with fewer than 2 instances
        class_counts = Counter(y_processed)
        valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
        y_processed = y_processed[y_processed.isin(valid_classes)]
        X_processed = X_processed.loc[y_processed.index]

        # Update app instance variables
        app_instance.X = X_processed
        app_instance.y = y_processed

        # Train-test split
        if app_instance.problem_type == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42, stratify=y_processed)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_processed, test_size=0.3, random_state=42)

        app_instance.X_train = X_train
        app_instance.X_test = X_test
        app_instance.y_train = y_train
        app_instance.y_test = y_test

        # Use partial fit with SGD for incremental learning
        model = SGDClassifier() if app_instance.problem_type == 'classification' else SGDRegressor()
        batch_size = 10000
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train.iloc[i:i + batch_size]
            model.partial_fit(X_batch, y_batch, classes=np.unique(y_processed) if app_instance.problem_type == 'classification' else None)

        app_instance.model = model

        app_instance.root.after(0, lambda: [
            app_instance.status_text.insert("end", "Model trained successfully.\n"),
            app_instance.progress_bar.stop(),
            app_instance.progress_bar.configure(mode='determinate'),
            update_progress(app_instance, 1, 1, app_instance.task_start_time),
            app_instance.enable_buttons(),
            setattr(app_instance, 'task_in_progress', False)
        ])
    except Exception as e:
        app_instance.root.after(0, lambda: [
            app_instance.status_text.insert("end", f"Error during training: {e}\n"),
            app_instance.progress_bar.stop(),
            app_instance.progress_bar.configure(mode='determinate'),
            app_instance.enable_buttons(),
            setattr(app_instance, 'task_in_progress', False)
        ])
        logging.error("Error during training", exc_info=True)

def get_model_instance(model_name, problem_type):
    """
    Get an instance of the model based on the model name and problem type.

    Parameters:
    - model_name: Name of the model
    - problem_type: 'classification' or 'regression'

    Returns:
    - model: An instance of the model
    """
    if problem_type == 'classification':
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            return RandomForestClassifier()
        elif model_name == "Support Vector Machine":
            return SVC(probability=True)
        elif model_name == "Gradient Boosting":
            return GradientBoostingClassifier()
        elif model_name == "K-Nearest Neighbors":
            return KNeighborsClassifier()
    else:
        if model_name == "Linear Regression":
            return LinearRegression()
        elif model_name == "Random Forest Regressor":
            return RandomForestRegressor()
        elif model_name == "Support Vector Regressor":
            return SVR()
        elif model_name == "Gradient Boosting Regressor":
            return GradientBoostingRegressor()
        elif model_name == "K-Nearest Neighbors Regressor":
            return KNeighborsRegressor()
    return None

def automated_model_selection(X_train, y_train, problem_type):
    """
    Automatically select the best model based on cross-validation performance.

    Parameters:
    - X_train: Training features
    - y_train: Training target variable
    - problem_type: 'classification' or 'regression'

    Returns:
    - best_model: Trained best model
    - best_model_name: Name of the best model
    """
    if problem_type == 'classification':
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "Gradient Boosting": GradientBoostingClassifier(),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }
        scoring = 'accuracy'
        cv = StratifiedKFold(n_splits=3)
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "Support Vector Regressor": SVR(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor()
        }
        scoring = 'neg_mean_squared_error'
        cv = KFold(n_splits=3)

    best_score = -np.inf
    best_model_name = None
    best_model = None
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        avg_score = cv_scores.mean()
        if problem_type == 'regression':
            avg_score = -avg_score  # Convert to positive MSE
        if avg_score > best_score:
            best_score = avg_score
            best_model_name = model_name
            best_model = model

    best_model.fit(X_train, y_train)
    return best_model, best_model_name

def compare_models_thread(app_instance):
    """
    Compare different models using cross-validation in a separate thread.

    Parameters:
    - app_instance: Instance of the MLModelBuilderApp
    """
    try:
        if app_instance.problem_type == 'classification':
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
                "Support Vector Machine": SVC(probability=True),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier()
            }
            scoring = 'accuracy'
            cv = StratifiedKFold(n_splits=5)
        else:
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=10),
                "Support Vector Regressor": SVR(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "K-Nearest Neighbors Regressor": KNeighborsRegressor()
            }
            scoring = 'neg_mean_squared_error'
            cv = KFold(n_splits=5)

        app_instance.total_steps = len(models)
        app_instance.current_step = 0
        app_instance.task_start_time = time.time()

        app_instance.root.after(0, lambda: [
            update_progress(app_instance, 0, app_instance.total_steps, app_instance.task_start_time),
            app_instance.status_text.insert("end", "Model comparison started...\n"),
            app_instance.progress_bar.configure(mode='indeterminate'),
            app_instance.progress_bar.start()
        ])

        results = []
        for model_name, model in models.items():
            cv_scores = cross_val_score(model, app_instance.X, app_instance.y, cv=cv, scoring=scoring)
            avg_score = cv_scores.mean()
            if app_instance.problem_type == 'regression':
                avg_score = -avg_score  # Convert to positive MSE
            results.append((model_name, avg_score))
            app_instance.root.after(0, lambda mn=model_name, ascore=avg_score: app_instance.status_text.insert("end", f"{mn} - Cross-Validation Score: {ascore:.4f}\n"))
            app_instance.current_step += 1
            progress = app_instance.current_step / app_instance.total_steps
            app_instance.root.after(0, lambda: update_progress(app_instance, progress, 1, app_instance.task_start_time))

        results_df = pd.DataFrame(results, columns=["Model", "CV Score"])
        import plotly.express as px
        fig = px.bar(results_df, x="CV Score", y="Model", orientation='h', title="Model Comparison")
        fig.show()
        app_instance.root.after(0, lambda: [
            app_instance.progress_bar.stop(),
            app_instance.progress_bar.configure(mode='determinate'),
            update_progress(app_instance, 1, 1, app_instance.task_start_time),
            app_instance.enable_buttons(),
            setattr(app_instance, 'task_in_progress', False)
        ])
    except Exception as e:
        app_instance.root.after(0, lambda: [
            app_instance.status_text.insert("end", f"Error during model comparison: {e}\n"),
            app_instance.progress_bar.stop(),
            app_instance.progress_bar.configure(mode='determinate'),
            app_instance.enable_buttons(),
            setattr(app_instance, 'task_in_progress', False)
        ])
        logging.error("Error during model comparison", exc_info=True)
