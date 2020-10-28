from sklearn.preprocessing import LabelEncoder

def clean_data(data):
    """
    
    """
    df = data.copy()
    # df = data.to_pandas_dataframe().dropna()

    # Drop columns with >= 60% missing data
    df.dropna(axis=1, thresh=0.6*len(df), inplace=True)
    # Drop identification columns 
    df.drop("Applicant_ID", axis=1, inplace=True)
    # LabelEncode the label column 
    enc = LabelEncoder()
    df.default_status = enc.fit_transform(df.default_status)
    df.form_field47 = enc.fit_transform(df.form_field47)

    return df
