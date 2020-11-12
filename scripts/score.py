"""
NOTE: Variables in bracket are loaded from notebook
"""

from numpy import array
from pandas import DataFrame
from sklearn.impute import KNNImputer
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib
import json
import os
import time

from azureml.core.model import Model
from azureml.core.run import Run
from xgboost import XGBClassifier



def init():
    global model
    global imputer

    # Voting Ensemble Model
    model_name = "kowope-ensemble"
    model_version = "5"
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), model_name, model_version, "votingensemble.joblib")

    # KNNImputer 
    imputer_name = "kowope-imputer"
    imputer_version = "3"
    imputer_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), imputer_name, imputer_version, "knnimputer.joblib")

    # Load artefacts 
    imputer = joblib.load(imputer_path)
    model = joblib.load(model_path)
    print("Initialized model" + time.strftime("%H:%M:%S"))


def clean_data(data: DataFrame):
    columns = ['form_field1', 'form_field2', 'form_field3', 'form_field4', 'form_field5', 'form_field6', 'form_field7', 'form_field8', 'form_field9', 'form_field10', 'form_field12', 'form_field13', 'form_field14', 'form_field16', 'form_field17', 'form_field18', 'form_field19', 'form_field20', 'form_field21', 'form_field22', 'form_field24', 'form_field25', 'form_field26', 'form_field27', 'form_field28', 'form_field29', 'form_field32', 'form_field33', 'form_field34', 'form_field36', 'form_field37', 'form_field38', 'form_field39', 'form_field42', 'form_field43', 'form_field44', 'form_field46', 'form_field47', 'form_field48', 'form_field49', 'form_field50']
    categories = [array(['charge', 'lending'], dtype=object)]

    df = data[columns]
    enc = OrdinalEncoder()
    enc.categories_ = categories
    df.form_field47 = enc.transform(df.form_field47.to_frame())
    return df


def run(data):
    try:
        input_data = json.loads(data)["data"]
        input_df = DataFrame(input_data)
        print("Input data shape: ", input_df.shape)

        cleaned_data = clean_data(input_df)
        print("Successfully cleaned data | " + time.strftime("%H:%M:%S"))

        imputed_data = imputer.transform(cleaned_data)
        imputed_data = DataFrame(imputed_data, columns=cleaned_data.columns)     # model requires columns names. 
        print("Successfully imputed missing values in data | " + time.strftime("%H:%M:%S"))

        predictions = model.predict(imputed_data)
        print(predictions)
        print("Successfully made predictions | " + time.strftime("%H:%M:%S"))
        return predictions.tolist()
    except Exception as e:
        # Development only. Change how error is returned for production
        return json.dumps(dict.fromkeys(["error"], str(e) + time.strftime("%H:%M:%S")))
