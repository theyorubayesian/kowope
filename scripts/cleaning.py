from ast import literal_eval
from azureml.core import Dataset, Run
from pandas import DataFrame
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
import argparse
import joblib

from utils import write_output

# key into current workspace
run = Run.get_context()


def clean_data(data: DataFrame, threshold: float, dropped_columns: list):
    """
    
    """
    df = data.copy()

    # Drop columns with >= % missing data
    df.dropna(axis=1, thresh=threshold*len(df), inplace=True)

    # Drop identification columns 
    df.drop(dropped_columns, axis=1, inplace=True)

    # LabelEncode the default_status & form_field47 columns
    enc = OrdinalEncoder()
    df.default_status = df.default_status.map({"yes": 1, "no": 0})
    df.form_field47 = enc.fit_transform(df.form_field47.to_frame())

    # Log form_field47 classes
    run.log_list("form_field47_categories", enc.categories_[0].tolist())
    # Log columns that are used
    run.log_list("useful_columns", list(df.drop("default_status", axis=1).columns))

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_name", type=str, help="Dataset name as registered for use")
    parser.add_argument("--dropped_columns", type=str, help="Columns to be dropped. To be inputed one after the other.")
    parser.add_argument("--threshold", type=float, help="Percentage of missing values dropped for dropped columns")
    parser.add_argument("--output_data", type=str, help="Output cleansed data.")

    # KNNImputer args
    parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors for KNNImputer")
    args = parser.parse_args()

    # Get dataset by name
    data = run.input_datasets[args.input_data_name]
    data_df = data.to_pandas_dataframe()
    clean_df = clean_data(data_df, threshold=args.threshold, dropped_columns=literal_eval(args.dropped_columns))
    print("Shape of cleaned dataset:\n", clean_df.shape)
    
    # Pop default_status so that it is not included in KNNImputer model 
    y = clean_df.pop("default_status")
    imputer = KNNImputer(n_neighbors=args.n_neighbors, weights="distance", add_indicator=False)
    imputed_df = DataFrame(imputer.fit_transform(clean_df), columns=clean_df.columns)
    imputed_df["default_status"] = y
    print("Fitted KNNImputer. Filled missing values.")

    # Dump model artifact 
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(imputer, "outputs/knnimputer.joblib")
    print("Saved KNNImputer artifact.")

    if not (args.output_data is None):
        write_output(imputed_df, path=args.output_data, filename="/cleaned.parquet")


if __name__ == "__main__":
    main()
