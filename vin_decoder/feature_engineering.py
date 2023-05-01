from typing import Tuple, Union

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)


def get_vin(vin: str) -> str:
    return vin[0:3]


def get_vehicle_attrs(vin: str) -> str:
    return vin[3:10]


def vectorize_text(
    vectorizer: Union[CountVectorizer, HashingVectorizer, TfidfVectorizer],
    X_train: Series,
    X_valid: Series,
) -> Tuple[ndarray]:
    X_train_featurized = vectorizer.fit_transform(X_train)
    X_valid_featurized = vectorizer.transform(X_valid)
    return (X_train_featurized, X_valid_featurized)


def extract_features_labels(df: DataFrame, label: str) -> Tuple[Union[Series, ndarray]]:
    df["wmi"] = df["vin"].apply(get_vin)
    df["vehicle_attrs"] = df["vin"].apply(get_vehicle_attrs)
    corpus = df["vehicle_attrs"]
    targets = df[label]
    class_names = targets.unique()
    return corpus, targets, class_names
