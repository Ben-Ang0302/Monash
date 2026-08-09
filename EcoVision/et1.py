import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ======================================================
# CONFIG
# ======================================================

CSV_PATH = "spectral_dataset_18ch_separate_lights.csv"
OUT_DIR = Path(".")

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPS = 1e-8

# AS7265x channels: 18 UV + 18 WHITE + 18 IR
WAVELENGTHS = [
    410, 435, 460, 485, 510, 535,
    560, 585, 610, 645, 680, 705,
    730, 760, 810, 860, 900, 940
]

# ======================================================
# FEATURE ENGINEERING
# IMPORTANT: use this exact same transformation during inference
# ======================================================

def build_features(X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 54 raw AS7265x readings into the final 108-feature input:
    - 54 raw features: uv_*, white_*, ir_*
    - 54 ratio features: ir/white, uv/white, ir/uv for each wavelength

    This matches the feature design that gave the best ExtraTrees result.
    """
    X_raw = X_raw.astype(np.float32).copy()

    if X_raw.shape[1] != 54:
        raise ValueError(f"Expected 54 raw spectral features, got {X_raw.shape[1]}")

    raw_cols = []
    for light in ["uv", "white", "ir"]:
        for wl in WAVELENGTHS:
            raw_cols.append(f"{light}_{wl}")

    X_raw.columns = raw_cols
    X = X_raw.copy()

    for i, wl in enumerate(WAVELENGTHS):
        uv = X_raw[f"uv_{wl}"]
        white = X_raw[f"white_{wl}"]
        ir = X_raw[f"ir_{wl}"]

        X[f"ir_over_white_{wl}"] = ir / (white + EPS)
        X[f"uv_over_white_{wl}"] = uv / (white + EPS)
        X[f"ir_over_uv_{wl}"] = ir / (uv + EPS)

    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    X.columns = X.columns.astype(str)
    return X.astype(np.float32)

# ======================================================
# LOAD CSV
# ======================================================

print("\nLoading dataset...")
df = pd.read_csv(
    CSV_PATH,
    header=None,
    sep=",",
    keep_default_na=False
)

print("\nDataset loaded")
print(df.head())
print(f"\nDataset shape: {df.shape}")

# ======================================================
# FEATURES + LABELS
# ======================================================

sample_id = df.iloc[:, 0]
image_path = df.iloc[:, 1]
y = df.iloc[:, 2].astype(str)
X_raw = df.iloc[:, 3:]

print("\nClass counts:")
print(y.value_counts())

print("\nRaw feature shape:", X_raw.shape)
X = build_features(X_raw)
print("Feature shape after ratio features:", X.shape)

# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Test samples :", len(X_test))

# ======================================================
# EXTRA TREES MODEL - BEST CURRENT PARAMETERS
# ======================================================
class_weight = {
    "glass": 1.0,
    "metal": 2.5,
    "null": 1.5,
    "paper": 1.0,
    "plastic": 1.0
}

model = ExtraTreesClassifier(
    n_estimators=4000,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.7,
    bootstrap=False,
    class_weight=class_weight,
    criterion="entropy",
    random_state=42,
    n_jobs=-1
)

print("\nTraining ExtraTrees best model...")
model.fit(X_train, y_train)
print("Training complete")

# ======================================================
# HOLD-OUT EVALUATION
# ======================================================

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
report_text = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

print("\n====================")
print("HOLD-OUT ACCURACY")
print("====================")
print(f"{accuracy * 100:.2f}%")

print("\n====================")
print("CLASSIFICATION REPORT")
print("====================")
print(report_text)

print("\n====================")
print("CONFUSION MATRIX")
print("====================")
print(cm)
print("Labels:", model.classes_)

# ======================================================
# SHUFFLED STRATIFIED CROSS VALIDATION
# Better than cv=5 when CSV rows are ordered by class/time.
# ======================================================

print("\nRunning 5-fold stratified shuffled cross validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=1
)

print("\nCV Scores:")
print(cv_scores)
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Std Dev: {cv_scores.std() * 100:.2f}%")

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

importance_df = pd.DataFrame({
    "feature": X.columns.astype(str),
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\n====================")
print("TOP 20 FEATURES")
print("====================")
print(importance_df.head(20))

# ======================================================
# SAVE OUTPUTS
# ======================================================

# Main files for app.py / deployment
joblib.dump(model, OUT_DIR / "random_forest_recycle.pkl")
joblib.dump(list(X.columns), OUT_DIR / "random_forest_feature_columns.pkl")

# Versioned backup files
joblib.dump(model, OUT_DIR / "extratrees_recycle_best.pkl")
joblib.dump(list(X.columns), OUT_DIR / "extratrees_feature_columns_best.pkl")

importance_df.to_csv(OUT_DIR / "extratrees_feature_importance_best.csv", index=False)
pd.DataFrame(report_dict).transpose().to_csv(OUT_DIR / "extratrees_classification_report_best.csv")
pd.DataFrame(cm, index=model.classes_, columns=model.classes_).to_csv(OUT_DIR / "extratrees_confusion_matrix_best.csv")

metadata = {
    "model_type": "ExtraTreesClassifier",
    "dataset_csv": CSV_PATH,
    "raw_feature_count": 54,
    "final_feature_count": int(X.shape[1]),
    "feature_design": "54 raw AS7265x features + 54 UV/white/IR ratio features",
    "classes": list(model.classes_),
    "class_counts": y.value_counts().to_dict(),
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "params": model.get_params(),
    "holdout_accuracy": float(accuracy),
    "cv_scores": [float(s) for s in cv_scores],
    "cv_mean_accuracy": float(cv_scores.mean()),
    "cv_std_accuracy": float(cv_scores.std())
}

with open(OUT_DIR / "extratrees_model_metadata_best.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\n====================")
print("MODEL SAVED")
print("====================")
print("random_forest_recycle.pkl")
print("random_forest_feature_columns.pkl")
print("extratrees_recycle_best.pkl")
print("extratrees_feature_columns_best.pkl")
print("extratrees_feature_importance_best.csv")
print("extratrees_classification_report_best.csv")
print("extratrees_confusion_matrix_best.csv")
print("extratrees_model_metadata_best.json")

# ======================================================
# APP.PY INFERENCE REMINDER
# ======================================================

print("\nIMPORTANT:")
print("During real-time inference, send the 54 raw spectral values in this exact order:")
print("UV 18 channels -> WHITE 18 channels -> IR 18 channels")
print("Then apply the same ratio feature transformation before model.predict_proba().")
