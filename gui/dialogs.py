# gui/dialogs.py

import tkinter as tk

class PreprocessingOptionsDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Preprocessing Options")
        self.grab_set()  # Make the window modal

        # Initialize variables to hold the options
        self.strategy = tk.StringVar(value="mean")
        self.fill_value = tk.DoubleVar(value=0)
        self.encoding_method = tk.StringVar(value="Label Encoding")
        self.outlier_method = tk.StringVar(value="None")
        self.scaling_method = tk.StringVar(value="StandardScaler")
        self.apply_poly = tk.BooleanVar(value=False)
        self.degree = tk.IntVar(value=2)
        self.apply_feature_selection = tk.BooleanVar(value=False)
        self.feature_selection_method = tk.StringVar(value="Correlation Threshold")
        self.threshold = tk.DoubleVar(value=0.9)
        self.n_features = tk.IntVar(value=10)

        # Create UI elements
        self.create_widgets()

        # Result: True if OK pressed, False otherwise
        self.result = False

    def create_widgets(self):
        # Use frames to organize the options
        # Missing Value Handling
        mv_frame = tk.LabelFrame(self, text="Missing Value Handling")
        mv_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(mv_frame, text="Strategy:").pack(side="left", padx=5, pady=5)
        missing_value_options = ["mean", "median", "most_frequent", "constant"]
        tk.OptionMenu(mv_frame, self.strategy, *missing_value_options, command=self.on_strategy_change).pack(side="left", padx=5, pady=5)
        self.fill_value_label = tk.Label(mv_frame, text="Fill Value:")
        self.fill_value_entry = tk.Entry(mv_frame, textvariable=self.fill_value)
        # Initially hide fill_value_entry and label unless strategy == "constant"
        if self.strategy.get() == "constant":
            self.fill_value_label.pack(side="left", padx=5, pady=5)
            self.fill_value_entry.pack(side="left", padx=5, pady=5)

        # Encoding Method
        enc_frame = tk.LabelFrame(self, text="Encoding Method")
        enc_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(enc_frame, text="Method:").pack(side="left", padx=5, pady=5)
        encoding_methods = ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding"]
        tk.OptionMenu(enc_frame, self.encoding_method, *encoding_methods).pack(side="left", padx=5, pady=5)

        # Outlier Detection
        outlier_frame = tk.LabelFrame(self, text="Outlier Detection")
        outlier_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(outlier_frame, text="Method:").pack(side="left", padx=5, pady=5)
        outlier_methods = ["None", "Z-Score", "IQR"]
        tk.OptionMenu(outlier_frame, self.outlier_method, *outlier_methods).pack(side="left", padx=5, pady=5)

        # Scaling Method
        scaling_frame = tk.LabelFrame(self, text="Scaling Method")
        scaling_frame.pack(fill="x", padx=10, pady=5)
        tk.Label(scaling_frame, text="Method:").pack(side="left", padx=5, pady=5)
        scaling_methods = ["StandardScaler", "MinMaxScaler", "RobustScaler"]
        tk.OptionMenu(scaling_frame, self.scaling_method, *scaling_methods).pack(side="left", padx=5, pady=5)

        # Polynomial Features
        poly_frame = tk.LabelFrame(self, text="Polynomial Features")
        poly_frame.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(poly_frame, text="Apply Polynomial Features", variable=self.apply_poly, command=self.on_apply_poly_change).pack(side="left", padx=5, pady=5)
        self.degree_label = tk.Label(poly_frame, text="Degree:")
        self.degree_entry = tk.Entry(poly_frame, textvariable=self.degree)
        # Initially hide degree entry unless apply_poly is True
        if self.apply_poly.get():
            self.degree_label.pack(side="left", padx=5, pady=5)
            self.degree_entry.pack(side="left", padx=5, pady=5)

        # Feature Selection
        fs_frame = tk.LabelFrame(self, text="Feature Selection")
        fs_frame.pack(fill="x", padx=10, pady=5)
        tk.Checkbutton(fs_frame, text="Apply Feature Selection", variable=self.apply_feature_selection, command=self.on_apply_feature_selection_change).pack(side="left", padx=5, pady=5)
        self.fs_method_label = tk.Label(fs_frame, text="Method:")
        feature_selection_methods = ["Correlation Threshold", "Recursive Feature Elimination"]
        self.fs_method_optionmenu = tk.OptionMenu(fs_frame, self.feature_selection_method, *feature_selection_methods, command=self.on_fs_method_change)
        # Additional options depending on method
        self.threshold_label = tk.Label(fs_frame, text="Threshold:")
        self.threshold_entry = tk.Entry(fs_frame, textvariable=self.threshold)
        self.n_features_label = tk.Label(fs_frame, text="Number of Features:")
        self.n_features_entry = tk.Entry(fs_frame, textvariable=self.n_features)
        # Initially hide method options unless apply_feature_selection is True
        if self.apply_feature_selection.get():
            self.fs_method_label.pack(side="left", padx=5, pady=5)
            self.fs_method_optionmenu.pack(side="left", padx=5, pady=5)
            self.update_fs_method_options()

        # Buttons
        button_frame = tk.Frame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        tk.Button(button_frame, text="OK", command=self.on_ok).pack(side="right", padx=5)
        tk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side="right", padx=5)

    def on_strategy_change(self, value):
        if value == "constant":
            self.fill_value_label.pack(side="left", padx=5, pady=5)
            self.fill_value_entry.pack(side="left", padx=5, pady=5)
        else:
            self.fill_value_label.pack_forget()
            self.fill_value_entry.pack_forget()

    def on_apply_poly_change(self):
        if self.apply_poly.get():
            self.degree_label.pack(side="left", padx=5, pady=5)
            self.degree_entry.pack(side="left", padx=5, pady=5)
        else:
            self.degree_label.pack_forget()
            self.degree_entry.pack_forget()

    def on_apply_feature_selection_change(self):
        if self.apply_feature_selection.get():
            self.fs_method_label.pack(side="left", padx=5, pady=5)
            self.fs_method_optionmenu.pack(side="left", padx=5, pady=5)
            self.update_fs_method_options()
        else:
            self.fs_method_label.pack_forget()
            self.fs_method_optionmenu.pack_forget()
            self.threshold_label.pack_forget()
            self.threshold_entry.pack_forget()
            self.n_features_label.pack_forget()
            self.n_features_entry.pack_forget()

    def on_fs_method_change(self, value):
        self.update_fs_method_options()

    def update_fs_method_options(self):
        method = self.feature_selection_method.get()
        if method == "Correlation Threshold":
            self.n_features_label.pack_forget()
            self.n_features_entry.pack_forget()
            self.threshold_label.pack(side="left", padx=5, pady=5)
            self.threshold_entry.pack(side="left", padx=5, pady=5)
        elif method == "Recursive Feature Elimination":
            self.threshold_label.pack_forget()
            self.threshold_entry.pack_forget()
            self.n_features_label.pack(side="left", padx=5, pady=5)
            self.n_features_entry.pack(side="left", padx=5, pady=5)
        else:
            self.threshold_label.pack_forget()
            self.threshold_entry.pack_forget()
            self.n_features_label.pack_forget()
            self.n_features_entry.pack_forget()

    def on_ok(self):
        self.result = True
        self.destroy()

    def on_cancel(self):
        self.result = False
        self.destroy()
