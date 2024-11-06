# visualization/plotting.py

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(cm, labels, save_path=None):
    """
    Plots the confusion matrix.

    Parameters:
    - cm: Confusion matrix array.
    - labels: List of labels for the classes.
    - save_path: Optional path to save the plot as an image. If None, display the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory
    else:
        plt.show()

def plot_roc_curve(fpr, tpr):
    """
    Plot the ROC curve.

    Parameters:
    - fpr: False positive rates
    - tpr: True positive rates
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash')))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    fig.show()

def visualize_regression_performance(y_true, y_pred):
    """
    Visualize regression model performance.

    Parameters:
    - y_true: True target values
    - y_pred: Predicted target values
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions'))
    fig.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()],
                             mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Regression: Actual vs Predicted',
                      xaxis_title='Actual Values',
                      yaxis_title='Predicted Values')
    fig.show()

def plot_residuals(y_true, y_pred):
    """
    Plot residuals for regression models.

    Parameters:
    - y_true: True target values
    - y_pred: Predicted target values
    """
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
    fig.update_layout(title='Residuals vs Predicted Values',
                      xaxis_title='Predicted Values',
                      yaxis_title='Residuals')
    fig.show()

def plot_feature_importance(importances, features):
    """
    Plot feature importances for tree-based models.

    Parameters:
    - importances: Array of feature importances
    - features: List of feature names
    """
    fig = px.bar(x=importances, y=features, orientation='h', title='Feature Importances')
    fig.show()

def visualize_data(data, selected_vars, status_text_widget):
    """
    Visualize the dataset or selected variables.

    Parameters:
    - data: DataFrame of the dataset
    - selected_vars: Selected variable(s) to visualize
    - status_text_widget: Text widget to display status messages
    """
    try:
        if selected_vars == "All":
            numeric_data = data.select_dtypes(include=['number'])
        else:
            if selected_vars not in data.columns:
                status_text_widget.insert("end", "Selected variable is not in the dataset.\n")
                return
            numeric_data = data[[selected_vars]]

        if numeric_data.shape[1] > 1:
            corr = numeric_data.corr()
            fig = px.imshow(corr,
                            text_auto=True,
                            labels=dict(x="Features", y="Features", color="Correlation"),
                            title="Feature Correlation Heatmap")
            fig.show()
            status_text_widget.insert("end", "Correlation heatmap displayed.\n")
        elif numeric_data.shape[1] == 1:
            fig = px.histogram(numeric_data, x=numeric_data.columns[0],
                               title=f'Distribution of {numeric_data.columns[0]}')
            fig.show()
            status_text_widget.insert("end", f"Histogram of {numeric_data.columns[0]} displayed.\n")
        else:
            status_text_widget.insert("end", "No numeric columns available for visualization.\n")
    except Exception as e:
        status_text_widget.insert("end", f"Error visualizing data: {e}\n")
        import logging
        logging.error("Error visualizing data", exc_info=True)
