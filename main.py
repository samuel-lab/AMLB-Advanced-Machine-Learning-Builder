# main.py

import customtkinter as ctk
from gui.app import MLModelBuilderApp
from utils.logging_config import setup_logging
from ml.feature_engineering import select_features
from ml.model_training import train_model_thread

def start_training_with_feature_selection(app_instance):
    """
    Apply feature selection and start model training.
    """
    # Check if data is loaded
    if not hasattr(app_instance, "X") or not hasattr(app_instance, "y"):
        app_instance.status_text.insert("end", "Please load a dataset before training.\n")
        return
    
    # Apply feature selection
    app_instance.status_text.insert("end", "Selecting top features...\n")
    X_selected, feature_selector = select_features(app_instance.X, app_instance.y, k=20)
    app_instance.X = X_selected  # Update with selected features

    # Start the training thread with updated data
    options = {}  # Add any options for training or preprocessing if needed
    train_model_thread(app_instance, options)

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    root = ctk.CTk()
    app = MLModelBuilderApp(root)

    # Link the training function to the app (e.g., via a button press or a menu item in your GUI)
    app.start_training = lambda: start_training_with_feature_selection(app)

    root.mainloop()

