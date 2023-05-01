from typing import List

from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)

from .feature_engineering import get_vehicle_attrs


def predict(
    vectorizer: CountVectorizer | HashingVectorizer | TfidfVectorizer,
    model: CatBoostClassifier,
    vins: List[str],
) -> List[str]:
    """Predict model for a given list of VINs.

    Args:
        vectorizer (CountVectorizer | HashingVectorizer | TfidfVectorizer): fitted vectorizer object to convert text to features.
        model (CatBoostClassifier): a trained catboost classifier.
        vins (List[str]): VINs to classify.

    Returns:
        str: Predicted models of each VIN.
    """
    vehicle_attrs = [get_vehicle_attrs(vin=vin) for vin in vins]
    features = vectorizer.transform(vehicle_attrs)
    predicted_classes = model.predict(data=features).flatten().tolist()
    return predicted_classes
