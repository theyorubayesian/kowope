from azureml.core.run import Run
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
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
    
    # ExtraTreesClassifier args
    parser.add_argument("--n_estimators", type=int, help="Number of trees in the forest")
    parser.add_argument("--min_samples_split", type=float, help="Min number to split a node")
    parser.add_argument("--min_samples_leaf", type=float, help="Min number of samples at leaf node")
    parser.add_argument("--max_features", type=float, help="Number of features to consider for split")
    parser.add_argument("--ccp_alpha", type=float, help="Complexity parameter for pruning")
    
    args = parser.parse_args()

    # Retrieve datasets by name | Create train/val/test
    train = run.input_datasets[args.input_data_name].to_pandas_dataframe()
    x_train, x_val = train_test_split(train, test_size=0.3, stratify=train.default_status, random_state=40)
    y_train, y_val = x_train.pop("default_status"), x_val.pop("default_status")

    # Log model parameters
    run.log("n_estimators", int(args.n_estimators))
    run.log("min_samples_split", int(args.min_samples_split))
    run.log("min_samples_leaf", np.float(args.min_samples_leaf))
    run.log("max_features", np.float(args.max_features))
    run.log("ccp_alpha", np.float(args.ccp_alpha))
    
    # train to find best number of boost rounds
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        ccp_alpha=args.ccp_alpha,
        class_weight="balanced"
    )
    model.fit(x_train, y_train)
    
    # Make prediction on Val dataset & log AUC
    y_pred = model.predict(x_val)
    auc_score = roc_auc_score(y_val, y_pred, average="weighted")
    run.log("auc", np.float(auc_score))

    print("Classification Report: \n", classification_report(y_val, y_pred))

    # Dump model artifact 
    os.makedirs('outputs/hyperdrive', exist_ok=True)
    joblib.dump(model, "outputs/hyperdrive/model.joblib")

    
if __name__ == "__main__":
    main()
