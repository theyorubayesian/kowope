from pandas import DataFrame
import os

def write_output(df: DataFrame, path: str, filename: str, index: bool = False):
    """
    Custom function to write output to datastore

    :param df: pandas.DataFrame object to be written
    :param path: Path on Datastore to write DataFrame to
    :param filename: Filename to save DataFrame with

    :return: None
    """
    # Create directory if not exist
    os.makedirs(path, exist_ok=True)

    print(f"Writing file to {path}")
    df.to_parquet(path + filename, index=False)
