# ml/model_evaluation.py

import threading
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_score)
import logging
import time
import numpy as np
from dask import dataframe as dd
from visualization.plotting import (plot_confusion_matrix, plot_roc_curve, visualize_regression_performance,
                                    plot_residuals, plot_feature_importance)
from utils.utils import update_progress

def evaluate_model_thread(app_instance, batch_size=10000):
    """
    Perform model evaluation in a separate thread using batch processing for memory efficiency.

    Parameters:
    - app_instance: Instance of the MLModelBuilderApp
    - batch_size: Size of each batch for evaluation to manage memory usage.
    """
    try:
        app_instance.task_start_time = time.time()
        app_instance.root.after(0, lambda: [
            update_progress(app_instance, 0, 1, app_instance.task_start_time),
            app_instance.status_text.insert("end", "Evaluation started...\n"),
            app_instance.progress_bar.configure(mode='indeterminate'),
            app_instance.progress_bar.start()
        ])

        # Convert to Dask DataFrame for batch processing
        dask_X_test = dd.from_pandas(app_instance.X_test, npartitions=(len(app_instance.X_test) // batch_size) + 1)
        dask_y_test = dd.from_pandas(app_instance.y_test, npartitions=(len(app_instance.y_test) // batch_size) + 1)

        y_pred_all = []

        # Evaluate in batches
        for X_batch, y_batch in zip(dask_X_test.to_delayed(), dask_y_test.to_delayed()):
            X_batch = X_batch.compute()
            y_batch = y_batch.compute()
            
            # Get predictions for batch
            y_pred = app_instance.model.predict(X_batch)
            y_pred_all.extend(y_pred)

        y_pred_all = np.array(y_pred_all)

        # Collect metrics
        if app_instance.problem_type == 'classification':
            accuracy = accuracy_score(app_instance.y_test, y_pred_all)
            report = classification_report(app_instance.y_test, y_pred_all, zero_division=1)
            cm = confusion_matrix(app_instance.y_test, y_pred_all)
            result = f"Accuracy: {accuracy:.2f}\n\nClassification Report:\n{report}"
            app_instance.last_evaluation_metrics = {'Accuracy': accuracy}
            
            # Save confusion matrix plot
            plot_confusion_matrix(cm, labels=np.unique(app_instance.y), save_path='confusion_matrix.png')
            app_instance.root.after(0, lambda: app_instance.status_text.insert("end", "Confusion matrix saved.\n"))

            # ROC AUC and ROC Curve
            if hasattr(app_instance.model, "predict_proba"):
                try:
                    y_score = app_instance.model.predict_proba(app_instance.X_test)
                    if len(app_instance.model.classes_) == 2:
                        roc_auc = roc_auc_score(app_instance.y_test, y_score[:, 1])
                        fpr, tpr, _ = roc_curve(app_instance.y_test, y_score[:, 1])
                        plot_roc_curve(fpr, tpr, save_path='roc_curve.png')
                        result += f"\nROC AUC: {roc_auc:.2f}"
                        app_instance.root.after(0, lambda: app_instance.status_text.insert("end", "ROC curve saved.\n"))
                except Exception as e:
                    result += f"\nError calculating ROC AUC: {e}"

        elif app_instance.problem_type == 'regression':
            mse = mean_squared_error(app_instance.y_test, y_pred_all)
            mae = mean_absolute_error(app_instance.y_test, y_pred_all)
            r2 = r2_score(app_instance.y_test, y_pred_all)
            result = f"Mean Squared Error: {mse:.2f}\nMean Absolute Error: {mae:.2f}\nR^2 Score: {r2:.2f}"
            app_instance.last_evaluation_metrics = {'MSE': mse, 'MAE': mae, 'R2': r2}
            
            # Save regression performance plots
            visualize_regression_performance(app_instance.y_test, y_pred_all, save_path='regression_performance.png')
            plot_residuals(app_instance.y_test, y_pred_all, save_path='residuals.png')
            app_instance.root.after(0, lambda: app_instance.status_text.insert("end", "Regression plots saved.\n"))

        # Feature Importance or SHAP Explanation
        if hasattr(app_instance.model, 'feature_importances_'):
            plot_feature_importance(app_instance.model.feature_importances_, app_instance.X.columns, save_path='feature_importance.png')
            app_instance.root.after(0, lambda: app_instance.status_text.insert("end", "Feature importance plot saved.\n"))
        else:
            shap_explainability_thread(app_instance)

        app_instance.root.after(0, lambda: [
            app_instance.status_text.insert("end", f"Evaluation Result:\n{result}\n"),
            app_instance.progress_bar.stop(),
            app_instance.progress_bar.configure(mode='determinate'),
            update_progress(app_instance, 1, 1, app_instance.task_start_time),
            app_instance.enable_buttons(),
            setattr(app_instance, 'task_in_progress', False)
        ])
    except Exception as e:
        app_instance.root.after(0, lambda: [
            app_instance.status_text.insert("end", f"Error during evaluation: {e}\n"),
            app_instance.progress_bar.stop(),
            app_instance.progress_bar.configure(mode='determinate'),
            app_instance.enable_buttons(),
            setattr(app_instance, 'task_in_progress', False)
        ])
        logging.error("Error during evaluation", exc_info=True)

def shap_explainability_thread(app_instance):
    """
    Compute SHAP values for model explainability in a separate thread.
    """
    threading.Thread(target=_shap_explainability, args=(app_instance,)).start()

def _shap_explainability(app_instance):
    try:
        app_instance.root.after(0, lambda: app_instance.status_text.insert("end", "Calculating SHAP values...\n"))
        import shap
        import matplotlib.pyplot as plt

        sample_size = min(100, len(app_instance.X_test))
        sample_X = app_instance.X_test.iloc[:sample_size]
        explainer = shap.Explainer(app_instance.model, app_instance.X_train)
        shap_values = explainer(sample_X)

        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, sample_X, show=False)
        plt.savefig('shap_summary_plot.png')
        plt.close()
        app_instance.root.after(0, lambda: app_instance.status_text.insert("end", "SHAP summary plot saved.\n"))
    except Exception as e:
        app_instance.root.after(0, lambda: app_instance.status_text.insert("end", f"Error generating SHAP plots: {e}\n"))
        logging.error("Error generating SHAP plots", exc_info=True)
