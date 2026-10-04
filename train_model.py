"""
train_model.py
--------------
Stage 3 (Data Preprocessing) + Stage 4 (Model Training) in one file.

What this script does:
  1. Loads data/traffic_data.csv
  2. Checks the data (missing values, duplicates, invalid numbers)
  3. Separates features (X) and target (y), encodes the target
  4. Splits into training (80%) and testing (20%) sets
  5. Trains a RandomForestClassifier
  6. Evaluates it on the TEST set (accuracy, precision, recall, F1, confusion matrix)
  7. Saves the model to model/traffic_model.pkl using Joblib

Run:  python train_model.py
(Run generate_data.py first so the CSV exists.)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "traffic_data.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model", "traffic_model.pkl")

# IMPORTANT: this order is saved with the model and reused in predict.py,
# so training and prediction always use the same feature order.
FEATURES = [
    "bandwidth_usage",
    "packets_per_second",
    "bytes_per_second",
    "latency",
    "packet_loss",
    "network_utilization",
    "connection_count",
]
TARGET = "congestion_level"

# Class names in order of severity: Normal = 0 ... Congestion = 3
CLASS_NAMES = ["Normal", "Medium", "High", "Congestion"]
CLASS_TO_NUMBER = {name: number for number, name in enumerate(CLASS_NAMES)}

TEST_SIZE = 0.20      # 20% of the data is kept aside for testing
RANDOM_STATE = 42     # makes results repeatable


# ------------------------------------------------------------------
# STEP 1: Load and clean the data
# ------------------------------------------------------------------
def load_and_clean_data():
    if not os.path.exists(DATA_FILE):
        sys.exit(f"ERROR: {DATA_FILE} not found. Run 'python generate_data.py' first.")

    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns.")

    # Check that every needed column exists
    missing_columns = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_columns:
        sys.exit(f"ERROR: Missing columns in CSV: {missing_columns}")

    # Check missing values
    missing_count = int(df[FEATURES + [TARGET]].isnull().sum().sum())
    print(f"Missing values found: {missing_count}")
    if missing_count > 0:
        df = df.dropna(subset=FEATURES + [TARGET])
        print(f"  -> Dropped rows with missing values. Rows left: {len(df)}")

    # Remove duplicates
    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"  -> Removed duplicates. Rows left: {len(df)}")

    # Validate that all feature columns are numeric and not negative
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # bad text becomes NaN
    df = df.dropna(subset=FEATURES)
    negative_rows = int((df[FEATURES] < 0).any(axis=1).sum())
    print(f"Rows with negative values: {negative_rows}")
    if negative_rows > 0:
        df = df[(df[FEATURES] >= 0).all(axis=1)]
        print(f"  -> Removed them. Rows left: {len(df)}")

    # Check that the target only has the 4 allowed classes
    unknown = set(df[TARGET].unique()) - set(CLASS_NAMES)
    if unknown:
        sys.exit(f"ERROR: Unknown class names in {TARGET}: {unknown}")

    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# STEP 2: Features, target, train/test split
# ------------------------------------------------------------------
def prepare_data(df):
    X = df[FEATURES]                              # input columns (fixed order)
    y = df[TARGET].map(CLASS_TO_NUMBER)           # text label -> number

    # stratify=y keeps the same class proportions in train and test sets
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)


# ------------------------------------------------------------------
# STEP 3: Train the model
# ------------------------------------------------------------------
def train_model(X_train, y_train):
    # No feature scaling needed: Random Forest uses simple "greater than /
    # less than" decisions, so the size of the numbers does not matter.
    model = RandomForestClassifier(
        n_estimators=100,       # number of decision trees
        max_depth=10,           # limit tree depth to avoid memorizing data
        min_samples_leaf=3,     # each leaf needs at least 3 samples
        class_weight="balanced",# give rare classes (Congestion) fair weight
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# STEP 4: Evaluate on the unseen test data (real metrics, nothing faked)
# ------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("\n================ MODEL EVALUATION (test data) ================")
    print(f"Accuracy           : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision (weighted): {precision_score(y_test, predictions, average='weighted'):.4f}")
    print(f"Recall (weighted)   : {recall_score(y_test, predictions, average='weighted'):.4f}")
    print(f"F1-score (weighted) : {f1_score(y_test, predictions, average='weighted'):.4f}")

    print("\nPer-class report:")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, digits=3))

    print("Confusion matrix (rows = actual, columns = predicted):")
    cm = confusion_matrix(y_test, predictions)
    print(pd.DataFrame(cm, index=[f"actual_{c}" for c in CLASS_NAMES],
                       columns=[f"pred_{c}" for c in CLASS_NAMES]).to_string())

    print("\nFeature importance (which inputs matter most):")
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    print(importance.sort_values(ascending=False).round(3).to_string())


# ------------------------------------------------------------------
# STEP 5: Save the model with Joblib
# ------------------------------------------------------------------
def save_model(model):
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    # We save the model together with the feature order and class names,
    # so predict.py never has to guess them.
    bundle = {
        "model": model,
        "features": FEATURES,
        "class_names": CLASS_NAMES,
    }
    joblib.dump(bundle, MODEL_FILE)
    print(f"\nModel saved to: {MODEL_FILE}")


def main():
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"\nTraining rows: {len(X_train)} | Testing rows: {len(X_test)}")

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    print("\nTraining finished successfully.")


if __name__ == "__main__":
    main()
