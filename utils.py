import requests
from pprint import pprint


def get_data(path="data/Train.csv", subset="train"):
    """

    """
    api = "https://api.zindi.africa/v1/competitions/dsn-ai-bootcamp-qualification-hackathon/files/{}.csv"
    token = {"auth_token": "hKhCphfxxZk6yjG6kJVbbj92"}
    
    url_set = {
        "train": {"url": api.format("Train")},
        "test": {"url": api.format("Test")}
    }

    response = requests.post(url_set[subset]["url"], data=token)
    data = response.content

    with open(path, "wb") as f:
        f.write(data)

    return path


def print_model(model, prefix=""):
    # Customize featurization
    # https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features#customize-featurization
    for step in model.steps:
        print(prefix + step[0])
        if hasattr(step[1], 'estimators') and hasattr(step[1], 'weights'):
            pprint({'estimators': list(
                e[0] for e in step[1].estimators), 'weights': step[1].weights})
            print()
            for estimator in step[1].estimators:
                print_model(estimator[1], estimator[0] + ' - ')
        else:
            pprint(step[1].get_params())
            print()
