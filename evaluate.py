"""
evaluate.py — loads trained URL and email models and reports clean,
consistent metrics for both.
"""
import os

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from feature_extraction import extract_url_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def print_section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)


def report_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n{label} — held-out test set metrics")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 score:  {f1:.4f}")
    print()
    print("  Classification report:")
    print(classification_report(y_true, y_pred, target_names=["benign", "phishing"]))
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# 1. Evaluate the URL model
print_section("URL MODEL")

url_model = joblib.load(os.path.join(MODEL_DIR, "url_model.joblib"))
url_feature_names = joblib.load(os.path.join(MODEL_DIR, "url_feature_names.joblib"))

url_data = pd.read_csv(os.path.join(DATA_DIR, "phishing_url_combined.csv"))
url_data["label"] = url_data["type"].map({"benign": 0, "phishing": 1})

feature_rows = url_data["url"].apply(extract_url_features).apply(pd.Series)
feature_rows = feature_rows.reindex(columns=url_feature_names, fill_value=0)

X = feature_rows.to_numpy(dtype="float64")
y = url_data["label"].to_numpy()

# Must match the split used in url_model.py, or these aren't truly "unseen" rows.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

url_preds = url_model.predict(X_test)
url_metrics = report_metrics(y_test, url_preds, "URL Model")



# 2. Evaluate the email/NLP model
print_section("EMAIL MODEL")

nlp_pipeline = joblib.load(os.path.join(MODEL_DIR, "nlp_pipeline.joblib"))

email_data = pd.read_csv(os.path.join(DATA_DIR, "spam_email.csv"))
email_data["input_text"] = email_data["subject"].fillna("") + " " + email_data["body"]

X_email = email_data["input_text"]
y_email = email_data["label"]

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_email, y_email, test_size=0.3, random_state=42, stratify=y_email
)

email_preds = nlp_pipeline.predict(X_test_e)
email_metrics = report_metrics(y_test_e, email_preds, "Email Model")
