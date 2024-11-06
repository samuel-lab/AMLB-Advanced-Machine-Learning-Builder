# ml/__init__.py

# Initialize the ml package
from .data_processing import preprocess_data, preprocess_new_data
from .model_training import train_model_thread, compare_models_thread, automated_model_selection
from .model_evaluation import evaluate_model_thread, shap_explainability_thread
from .feature_engineering import apply_polynomial_features, choose_scaling_method
from .model_training import train_model_thread, compare_models_thread, automated_model_selection
