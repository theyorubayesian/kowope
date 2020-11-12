from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import argparse
import joblib
import os
import pandas as pd

from azureml.core import Dataset
from azureml.core.run import Run
from xgboost import XGBClassifier

run = Run.get_context()
workspace = run.experiment.workspace


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    # XGBClassifier args
    parser.add_argument("--eta", type=float, help="Learning rate for model")
    parser.add_argument("--max_depth", type=int, help="Depth for trees")
    parser.add_argument("--min_child_weight", type=int, help="Min child weight for tree")
    parser.add_argument("--subsample", type=float, help="Subsample of training set used for each iteration")
    parser.add_argument("--colsample_bytree", type=float, help="Subsample of columns to use for each iteration")
    parser.add_argument("--early_stopping_rounds", type=int, help="Model will stop iterating if no improvement after set number of rounds")
    parser.add_argument("--eval_metric", type=str, default="auc", help="Metric for evaluation")
    parser.add_argument("--scale_pos_weight", type=float, help="Control balance of positive and negative weights")
    parser.add_argument("--max_delta_step", type=int, help="Conservativeness of update step")
    parser.add_argument("--num_boost_rounds", type=int, help="Number of estimators ")
    # SGD args
    parser.add_argument("--alpha", type=float, default="linear", help="Regularization strength")
    parser.add_argument("--l1_ratio", type=float, default=1.0, help="l1_ratio in elasticnet penalty")

    # ExtraTreesClassifier args
    parser.add_argument("--n_estimators", type=int, help="Number of trees in the forest")
    parser.add_argument("--min_samples_split", type=float, help="Min number to split a node")
    parser.add_argument("--min_samples_leaf", type=float, help="Min number of samples at leaf node")
    parser.add_argument("--max_features", type=float, help="Number of features to consider for split")
    parser.add_argument("--ccp_alpha", type=float, help="Complexity parameter for pruning")

    args = parser.parse_args()

    # Retrieve datasets by name | Create train/val
    location = Dataset.get_by_name(workspace=workspace, name="cleaned_loan_dataset").download()
    print(location)
    
    train = pd.read_parquet(location[0])

    x_train, x_val = train_test_split(train, test_size=0.3, stratify=train.default_status, random_state=20)
    y_train, y_val = x_train.pop("default_status"), x_val.pop("default_status")

    # SVM Classifier 
    scaler = StandardScaler()
    sgd = SGDClassifier(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        penalty="elasticnet",
        loss="modified_huber",
        class_weight="balanced",
        early_stopping=True
    ) 
    sgd_clf = make_pipeline(scaler, sgd)

    # ExtraTreesClassifier
    etc_clf = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        ccp_alpha=args.ccp_alpha,
        class_weight="balanced"
    )

    # XGBoost
    xgb_clf = XGBClassifier(
        objective="binary:logistic",
        n_estimators=args.num_boost_rounds,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        learning_rate=args.eta,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        eval_metric=args.eval_metric,
        scale_pos_weight=args.scale_pos_weight,
        max_delta_step=args.max_delta_step
    )

    # VotingClassifier
    model = VotingClassifier(
        estimators=[("sgd", sgd_clf), ("etc", etc_clf), ("xgb", xgb_clf)], 
        voting="soft"
    )

    model.fit(x_train, y_train)

    # Make prediction on Val dataset & log AUC
    y_pred = model.predict(x_val)
    auc_score = roc_auc_score(y_val, y_pred, average="weighted")
    run.log("auc", float(auc_score))

    print("Classification Report: \n", classification_report(y_val, y_pred))

    # Dump model artifact 
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, "outputs/model_voting.joblib")

    
if __name__ == "__main__":
    main()
