from azureml.core.run import Run
from sklearn.impute import KNNImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import numpy as np
import os

from cleaning import clean_data 

run = Run.get_context()


def main():
    # Add arguments to script 
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--input_data_name", type=str, help="Environment name for y_test")

    # SVC args
    parser.add_argument("--kernel", type=str, default="linear", help="Kernel type to be used.")
    parser.add_argument("--penalty", type=float, default=1.0, help="Penalty parameter of the error term")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for KNNImputer")
    args = parser.parse_args()

    # Retrieve datasets by name | Clean & create train/val
    train = run.input_datasets[args.input_data_name]
    clean_train = clean_data(data)
    x_train, x_val = train(clean_train, test_size=0.3, stratify=train.default_status, random_state=42)
    y_train, y_val = x_train.pop("default_status"), x_val.pop("default_status")

    # Log model parameters
    run.log("Kernel type", np.str(args.kernel))
    run.log("Penalty", np.float(args.penalty))
    run.log("Neighbors", int(args.n_neighbors))

    # Train SVM Classifier 
    imputer = KNNImputer(n_neighbors=args.n_neighbors, weights="distance", add_indicator=True)
    model = SGDClassifier(
        alpha=args.alpha,
        class_weight="balanced",
        eta0=args.eta_zero,
        l1_ratio=args.l1_ratio,
    )

    clf = make_pipeline(imputer, model)
    cl.fit(x_train, y_train)
    
    # Make prediction on Val dataset & log AUC
    y_pred = clf.predict(x_val)
    auc_score = roc_auc_score(y_val, y_pred)
    run.log("auc", np.float(auc_score))

    # Dump model artifact 
    os.makedirs('outputs/hyperdrive', exist_ok=True)
    joblib.dump(clf, "outputs/hyperdrive/model.joblib")

    
if __name__ == "__main__":
    main()
