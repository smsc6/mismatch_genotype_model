### This is BASE, first model - takes rudimentary training data for building model structure ###
### no model specifics in this - this translates to pipeline ##


import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#for saving training data
import json
from datetime import datetime


FEATURE_COLS = [
    "depth",
    "n_match",
    "n_mismatch",
    "match_rate",
    "mismatch_rate",
    "A_to_C", "A_to_G", "A_to_T",
    "C_to_A", "C_to_G", "C_to_T",
    "G_to_A", "G_to_C", "G_to_T",
    "T_to_A", "T_to_C", "T_to_G",
]


def load_training_data(in_csv):
    """
    Load training CSV into X, y
    """
    X, y = [], []

    with open(in_csv, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            X.append([float(row[col]) for col in FEATURE_COLS])
            y.append(int(row["y"]))

    return np.array(X), np.array(y)


def train_logistic_regression(X_train, y_train):
    """
    Train LR model
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train RF model
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model



def save_model_results(y_test, y_pred, model_name, exp_name, in_csv):
    """
    Save evaluation results for one model run

    :param y_test: true labels
    :param y_pred: predicted labels
    :param model_name: model version name
    :param in_csv: input training csv path
    :return: path to saved json         ****** better way ???
    """
   
    project_root = Path(__file__).resolve().parents[1] #make directory creation/ path robust
    out_dir = project_root / "outputs" / "model_training_results"   
    
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{model_name}_{exp_name}_{timestamp}.json"

    results = {
        "timestamp": timestamp,
        "model_name": model_name,
        "exp_name": exp_name,
        "input_csv": str(in_csv),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0
        ),
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"saved results to {out_path}")
    return out_path



def evaluate_model(model, X_test, y_test, model_name, exp_name, in_csv, save=True):
    """
    Print evaluation metrics - confusion matrix, classifications
    """
    y_pred = model.predict(X_test)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if save:
        save_model_results(y_test, y_pred, model_name, exp_name, in_csv)





def train_baseline_model(in_csv, exp_name, save=True):
    """
    Training and evaluation pipeline
    """
    
    X, y = load_training_data(in_csv)

    # debugging - check class dist
    print(np.bincount(y))

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Log reg
    lr_model = train_logistic_regression(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "lr", exp_name, in_csv, save=save)

    # Random forset
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "rf", exp_name, in_csv, save=save)




################## TESTING ###############
if __name__ == "__main__":
    IN_CSV = "outputs/chr10_training_data_v1.csv"

    EXP_NAME = "v0_stratified"
    train_baseline_model(IN_CSV, EXP_NAME, save=True)
    
    #for scaling: train_baseline_model(IN_CSV, save=False) - save selectively there
