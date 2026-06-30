import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import inference, train_model

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]


def load_clean_data():
    data = pd.read_csv("data/census.csv")
# Clean column names
    data.columns = data.columns.str.strip()
# Clean string values with leading/trailing spaces
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].str.strip()

    return data

# TODO: implement the first test. Change the function name and input as needed


def test_train_split_size():
    """
    Tests training and test dataset sizes
    """
    data = load_clean_data()

    train, test = train_test_split(
        data,
        test_size=.20,
        random_state=42,
    )

    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) + len(test) == len(data)
    assert abs((len(test) / len(data)) - .20) < .01


# TODO: implement the second test. Change the function name and input as needed
def test_model_expected_algorithm():
    """
    #Test that train_model uses a RandomForestClassifier
    """
    data = load_clean_data()

    train, _ = train_test_split(
        data,
        test_size=.20,
        random_state=42,
    )

    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)

# TODO: implement the third test. Change the function name and input as needed


def test_inference():
    """
    Test that inference returns an array with predictions
    """
    data = load_clean_data()

    train, test = train_test_split(
        data,
        test_size=.20,
        random_state=42,
    )

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y_test)
