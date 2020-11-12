from azureml.core.run import Run
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import argparse
import joblib
import numpy as np
import os
import xgboost as xgb

run = Run.get_context()


def main():
    # Add arguments to script 
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--input_data_name", type=str, help="Environment name for cleaned loan dataset")
    
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
    
    args = parser.parse_args()

    # Retrieve datasets by name | Create train/val/test
    train = run.input_datasets[args.input_data_name].to_pandas_dataframe()
    x_train, x_val = train_test_split(train, test_size=0.3, stratify=train.default_status, random_state=40)
    x_val, x_test = train_test_split(x_val, test_size=0.2, stratify=x_val.default_status, random_state=40)
    y_train, y_val, y_test = x_train.pop("default_status"), x_val.pop("default_status"), x_test.pop("default_status")

    # create DMatrix objects 
    dtrain = xgb.DMatrix(train.drop("default_status", axis=1).values, label=train.default_status.values)
    dtrain_1 = xgb.DMatrix(x_train.values, label=y_train.values)
    dval = xgb.DMatrix(x_val.values, label=y_val.values)
    dtest = xgb.DMatrix(x_test.values, label=y_test.values)

    # Log model parameters
    run.log("max_depth", int(args.max_depth))
    run.log("min_child_weight", int(args.min_child_weight))
    run.log("subsample", np.float(args.subsample))
    run.log("colsample_bytree", np.float(args.colsample_bytree))
    run.log("scale_pos_weight", np.float(args.scale_pos_weight))
    run.log("max_delta_step", int(args.max_delta_step))
    run.log("eta", np.float(args.eta))

    params = {
        "objective": "binary:logistic",
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "eval_metric": args.eval_metric,
        "scale_pos_weight": args.scale_pos_weight,
        "max_delta_step": args.max_delta_step,
    }

    # set to an arbitrarily large number
    num_boost_rounds = 999

    # train to find best number of boost rounds
    model = xgb.train(
        params=params,
        dtrain=dtrain_1,
        num_boost_round=num_boost_rounds,
        evals=[(dval, "Val")],
        early_stopping_rounds=args.early_stopping_rounds
    )
    # get optimal number of boost rounds
    run.log("num_boost_rounds", model.best_iteration+1)

    # Make prediction on Val dataset & log AUC
    y_pred = [1 if x >= 0.5 else 0 for x in model.predict(dtest, ntree_limit=model.best_ntree_limit)]
    auc_score = roc_auc_score(y_test, y_pred, average="weighted")
    run.log("auc", np.float(auc_score))

    print("Classification Report: \n", classification_report(y_test, y_pred))

    # Dump model artifact 
    os.makedirs('outputs/hyperdrive', exist_ok=True)
    joblib.dump(model, "outputs/hyperdrive/model.joblib")

    
if __name__ == "__main__":
    main()
