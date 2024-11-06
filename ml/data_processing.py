# ml/data_processing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, OrdinalEncoder,
                                   PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler)
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from scipy import stats
import logging
import dask.dataframe as dd

def preprocess_data(X, y, options):
    """
    Preprocess the data including handling missing values, encoding, scaling, and feature selection.

    Parameters:
    - X: Features DataFrame
    - y: Target variable
    - options: Dictionary of preprocessing options

    Returns:
    - X_processed: Preprocessed features
    - y: Possibly modified target variable
    - preprocessing_objects: Dictionary of fitted preprocessing objects
    """
    preprocessing_objects = {}
    dask_X = dd.from_pandas(X, npartitions=10)  # Adjust partitions based on memory availability
    dask_y = dd.from_pandas(y, npartitions=10)

    # Handle missing values
    strategy = options.get('strategy', 'mean')
    fill_value = options.get('fill_value', None)
    if strategy == "constant":
        imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy=strategy)
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
    preprocessing_objects['imputer'] = imputer

    # Handle categorical features
    encoding_method = options.get('encoding_method', 'Label Encoding')
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if encoding_method == "Label Encoding":
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
        preprocessing_objects['encoders'] = encoders
    elif encoding_method == "One-Hot Encoding":
        onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_cats = onehot_encoder.fit_transform(X[categorical_cols])
        encoded_cols = onehot_encoder.get_feature_names_out(categorical_cols)
        X_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols)
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
        preprocessing_objects['onehot_encoder'] = onehot_encoder
    elif encoding_method == "Ordinal Encoding":
        ordinal_encoder = OrdinalEncoder()
        X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])
        preprocessing_objects['ordinal_encoder'] = ordinal_encoder

    # Outlier Detection and Removal
    outlier_method = options.get('outlier_method', 'None')
    if outlier_method == "Z-Score":
        z_scores = np.abs(stats.zscore(X[numeric_cols]))
        threshold = 3
        mask = (z_scores < threshold).all(axis=1)
        X = X[mask]
        y = y.loc[X.index]
    elif outlier_method == "IQR":
        Q1 = X[numeric_cols].quantile(0.25)
        Q3 = X[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((X[numeric_cols] < (Q1 - 1.5 * IQR)) | (X[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        X = X[mask]
        y = y.loc[X.index]

    # Scaling
    scaling_method = options.get('scaling_method', 'StandardScaler')
    if scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaling_method == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    if numeric_cols:
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    preprocessing_objects['scaler'] = scaler

    # Polynomial Features
    apply_poly = options.get('apply_poly', False)
    if apply_poly:
        degree = options.get('degree', 2)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X[numeric_cols])
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
        X = X.drop(columns=numeric_cols)
        X = pd.concat([X.reset_index(drop=True), X_poly_df.reset_index(drop=True)], axis=1)
        preprocessing_objects['poly'] = poly

    # Feature Selection
    apply_feature_selection = options.get('apply_feature_selection', False)
    if apply_feature_selection:
        feature_selection_method = options.get('feature_selection_method')
        if feature_selection_method == 'Correlation Threshold':
            threshold = options.get('threshold', 0.9)
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            X = X.drop(columns=to_drop)
        elif feature_selection_method == 'Recursive Feature Elimination':
            n_features = options.get('n_features', 10)
            problem_type = options.get('problem_type', 'classification')
            if problem_type == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                estimator = RandomForestClassifier()
            else:
                from sklearn.ensemble import RandomForestRegressor
                estimator = RandomForestRegressor()
            selector = RFE(estimator, n_features_to_select=n_features, step=1)
            selector = selector.fit(X, y)
            X = X.loc[:, selector.support_]
            preprocessing_objects['feature_selector'] = selector

    # Identify columns with non-numeric data
    categorical_columns = dask_X.select_dtypes(include=['object']).columns.tolist()

    # Apply one-hot encoding to categorical columns
    if categorical_columns:
        print(f"Encoding categorical columns: {categorical_columns}")
        dask_X = dd.get_dummies(dask_X, columns=categorical_columns, dummy_na=True)  # Include dummy for NaNs

    # Compute the processed Dask DataFrame to a Pandas DataFrame
    X_processed = dask_X.compute()
    y_processed = dask_y.compute()

    # Ensure all columns are numeric
    non_numeric_cols = X_processed.select_dtypes(include=['object']).columns
    if not non_numeric_cols.empty:
        raise ValueError(f"Some columns are still non-numeric: {non_numeric_cols}")

    # Convert any remaining non-numeric entries to NaN, then fill or drop as needed
    X_processed = X_processed.apply(pd.to_numeric, errors='coerce')
    X_processed = X_processed.fillna(0)  # Fill NaNs with 0 or an appropriate placeholder

    preprocessing_objects = {}  # Store any additional preprocessing information if needed
    return X_processed, y_processed, preprocessing_objects

def preprocess_new_data(new_data, preprocessing_objects):
    """
    Preprocess new data before prediction using saved preprocessing objects.

    Parameters:
    - new_data: DataFrame of new data
    - preprocessing_objects: Dictionary of fitted preprocessing objects

    Returns:
    - X_processed: Preprocessed new data
    """
    X = new_data.copy()
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Handle missing values
    imputer = preprocessing_objects.get('imputer')
    if imputer and numeric_cols:
        X[numeric_cols] = imputer.transform(X[numeric_cols])

    # Handle categorical features
    if 'encoders' in preprocessing_objects:
        encoders = preprocessing_objects['encoders']
        for col in categorical_cols:
            if col in encoders:
                le = encoders[col]
                X[col] = X[col].astype(str).map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)
            else:
                X[col] = -1  # Unseen categorical variable
    elif 'onehot_encoder' in preprocessing_objects:
        onehot_encoder = preprocessing_objects['onehot_encoder']
        encoded_cats = onehot_encoder.transform(X[categorical_cols])
        encoded_cols = onehot_encoder.get_feature_names_out(categorical_cols)
        X_encoded = pd.DataFrame(encoded_cats, columns=encoded_cols)
        X = X.drop(columns=categorical_cols)
        X = pd.concat([X.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    elif 'ordinal_encoder' in preprocessing_objects:
        ordinal_encoder = preprocessing_objects['ordinal_encoder']
        X[categorical_cols] = ordinal_encoder.transform(X[categorical_cols])

    # Scaling
    scaler = preprocessing_objects.get('scaler')
    if scaler and numeric_cols:
        X[numeric_cols] = scaler.transform(X[numeric_cols])

    # Polynomial Features
    if 'poly' in preprocessing_objects:
        poly = preprocessing_objects['poly']
        X_poly = poly.transform(X[numeric_cols])
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        X_poly_df = pd.DataFrame(X_poly, columns=poly_feature_names)
        X = X.drop(columns=numeric_cols)
        X = pd.concat([X.reset_index(drop=True), X_poly_df.reset_index(drop=True)], axis=1)

    # Feature Selection
    if 'feature_selector' in preprocessing_objects:
        selector = preprocessing_objects['feature_selector']
        X = X.loc[:, selector.support_]

    return X

