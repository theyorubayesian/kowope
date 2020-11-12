from azureml.core.run import Run
from pandas import concat
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import numpy as np
import os

run = Run.get_context()


def main():
    # Add arguments to script 
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--input_data_name", type=str, help="Environment name for cleaned loan dataset")
    
    # SGD args
    parser.add_argument("--alpha", type=float, default="linear", help="Regularization strength")
    parser.add_argument("--l1_ratio", type=float, default=1.0, help="l1_ratio in elasticnet penalty")

    args = parser.parse_args()

    # Retrieve datasets by name | Create train/val
    train = run.input_datasets[args.input_data_name].to_pandas_dataframe()

    x_train, x_val = train_test_split(train, test_size=0.3, stratify=train.default_status, random_state=24)
    y_train, y_val = x_train.pop("default_status"), x_val.pop("default_status")

    # Log model parameters
    run.log("alpha", np.float(args.alpha))
    run.log("l1_ratio", np.float(args.l1_ratio))

    # Train SVM Classifier 
    scaler = StandardScaler()
    model = SGDClassifier(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        penalty="elasticnet",
        loss="modified_huber",
        class_weight="balanced",
        early_stopping=True
    )     

    clf = make_pipeline(scaler, model)
    clf.fit(x_train, y_train)
    
    # Make prediction on Val dataset & log AUC
    y_pred = clf.predict(x_val)
    auc_score = roc_auc_score(y_val, y_pred, average="weighted")
    run.log("auc", np.float(auc_score))

    print("Classification Report: \n", classification_report(y_val, y_pred))

    # Dump model artifact 
    os.makedirs('outputs/hyperdrive', exist_ok=True)
    joblib.dump(clf, "outputs/hyperdrive/model.joblib")

    
if __name__ == "__main__":
    main()
