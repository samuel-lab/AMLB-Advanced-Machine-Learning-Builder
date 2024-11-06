# ml/feature_engineering.py
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler

def apply_polynomial_features(X, degree):
    """
    Apply polynomial feature transformation to the data.

    Parameters:
    - X: Features DataFrame
    - degree: Degree of polynomial features

    Returns:
    - X_poly: DataFrame with polynomial features
    - poly: Fitted PolynomialFeatures object
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly, poly

def choose_scaling_method(X, method):
    """
    Apply the selected scaling method to the data.

    Parameters:
    - X: Features DataFrame
    - method: Scaling method ('StandardScaler', 'MinMaxScaler', 'RobustScaler')

    Returns:
    - X_scaled: Scaled features DataFrame
    - scaler: Fitted scaler object
    """
    if method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def select_features(X, y, k=20):
    """
    Select top k features based on ANOVA F-value for the provided data.
    """
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector