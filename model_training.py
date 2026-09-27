"""

"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, roc_auc_score, confusion_matrix
)

TRAIN_FILE = "train.csv"
TARGET = "Price_INR_Lakhs"
HIGH_VALUE_THRESHOLD = 100.0  # ₹1 crore = 100 lakh

df = pd.read_csv(TRAIN_FILE)
feature_cols = [c for c in df.columns if c not in ["Property_ID", TARGET]]
cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
num_cols = [c for c in feature_cols if df[c].dtype != "object"]

X = df[feature_cols]
y = df[TARGET]

# ---------------- Linear regression ----------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

linear_preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
])

linear_model = Pipeline([
    ("preprocessor", linear_preprocessor),
    ("regressor", TransformedTargetRegressor(
        regressor=LinearRegression(),
        func=np.log1p,
        inverse_func=np.expm1
    )),
])

linear_model.fit(X_tr, y_tr)
pred = np.maximum(linear_model.predict(X_val), 0)

linear_metrics = {
    "MAE_lakhs": mean_absolute_error(y_val, pred),
    "RMSE_lakhs": mean_squared_error(y_val, pred) ** 0.5,
    "R2": r2_score(y_val, pred),
}

# Refit on all rows before deployment.
linear_model.fit(X, y)

# ---------------- Logistic regression ----------------
# The supplied data has no sanction/default/loan field, so we create
# an auditable property-value class: >= ₹1 crore.
y_class = (y >= HIGH_VALUE_THRESHOLD).astype(int)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y_class, test_size=0.20, random_state=42, stratify=y_class
)

log_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
])

logistic_model = Pipeline([
    ("preprocessor", log_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    )),
])

logistic_model.fit(X_tr, y_tr)
class_pred = logistic_model.predict(X_val)
prob = logistic_model.predict_proba(X_val)[:, 1]

logistic_metrics = {
    "accuracy": accuracy_score(y_val, class_pred),
    "ROC_AUC": roc_auc_score(y_val, prob),
    "confusion_matrix": confusion_matrix(y_val, class_pred).tolist(),
}

logistic_model.fit(X, (y >= HIGH_VALUE_THRESHOLD).astype(int))

metadata = {
    "target": TARGET,
    "feature_cols": feature_cols,
    "categorical_cols": cat_cols,
    "numeric_cols": num_cols,
    "high_value_threshold_lakhs": HIGH_VALUE_THRESHOLD,
    "linear_validation_metrics": linear_metrics,
    "logistic_validation_metrics": logistic_metrics,
    "training_rows": len(df),
}

ui_metadata = {
    "categorical_options": {
        c: sorted(df[c].astype(str).unique().tolist()) for c in cat_cols
    },
    "numeric_ranges": {
        c: {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "median": float(df[c].median()),
        }
        for c in num_cols
    },
    "metadata": metadata,
}

joblib.dump({"model": linear_model, "metadata": ui_metadata},
            "linear_regression_model.sav", compress=3)
joblib.dump({"model": logistic_model, "metadata": ui_metadata},
            "logistic_regression_model.sav", compress=3)
joblib.dump(ui_metadata, "model_metadata.sav", compress=3)

with open("validation_metrics.json", "w") as f:
    json.dump({
        "linear_regression": linear_metrics,
        "logistic_regression": logistic_metrics
    }, f, indent=2)

print("Models created successfully.")
print("Linear:", linear_metrics)
print("Logistic:", logistic_metrics)
