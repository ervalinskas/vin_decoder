import logging
import sys
from pathlib import Path

import mlflow
from rich.logging import RichHandler
from sklearn.feature_extraction.text import (  # HashingVectorizer,; TfidfVectorizer,
    CountVectorizer,
)

# Assets
DATA_URL = "https://raw.githubusercontent.com/carVertical/ml-engineering-homework/master/data/ml-engineer-challenge-redacted-data.csv"

# Define base dirs
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")

# Define data dirs
DATA_DIR = Path(BASE_DIR, "data")
RAW_DATA_DIR = Path(DATA_DIR, "raw")
VALIDATED_DATA_DIR = Path(DATA_DIR, "data_validation")
PREPROCESSED_LABELS = Path(DATA_DIR, "preprocessed_labels")

# Define misc dirs
LOGS_DIR = Path(BASE_DIR, "logs")

# Define stores dirs
STORES_DIR = Path(BASE_DIR, "stores")
MODEL_REGISTRY = Path(STORES_DIR, "model")
BLOB_STORE = Path(STORES_DIR, "blob")

# Create data dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
VALIDATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSED_LABELS.mkdir(parents=True, exist_ok=True)

# Create misc dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Create stores dirs
MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)
BLOB_STORE.mkdir(parents=True, exist_ok=True)

# MLFlow model registry
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

# Logger
logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.ERROR,
        },
    },
    "root": {
        "handlers": ["console", "info", "error"],
        "level": logging.INFO,
        "propagate": True,
    },
}
logging.config.dictConfig(logging_config)
logger = logging.getLogger()
logger.handlers[0] = RichHandler(markup=True)

# Defining nested cross-validation parameters
# Commenting some parts to speed up the pipeline
vectorizers = {
    "Bigrams": CountVectorizer(ngram_range=(2, 2), analyzer="char"),
    # "Hash vectorizer": HashingVectorizer(ngram_range=(2, 2), analyzer="char"),
    # "TF-IDF": TfidfVectorizer(ngram_range=(2, 2), analyzer="char"),
}
# grid = {"learning_rate": [0.1, 0.15], "depth": [8, 10], "iterations": [700, 900]}
grid = {"learning_rate": [0.15], "depth": [10], "iterations": [10]}
n_splits = 2

# Defining relevant labels for different data pipeline steps
labels_to_validate = ["make", "model", "year", "body"]
labels_to_preprocess = ["model"]
labels_to_train = ["model"]

# Dicts to map labels
models_to_group_1 = {
    "335": "3 Series",
    "4": "4 Series",
    "428I (USA)": "4 Series",
    "530D": "5 Series",
    "530D (EUR)": "5 Series",
    "M5": "5 Series",
    "630dx (630dx)": "6 Series",
    "640dx (640dx)": "6 Series",
    "S5": "A5",
    "rs 7": "A7",
    "SQ5": "Q5",
}
models_to_group_2 = {
    "X1": "X Series",
    "X2": "X Series",
    "X3": "X Series",
    "X4": "X Series",
    "X5": "X Series",
    "X6": "X Series",
    "i3": "i",
    "i8": "i",
    "Z3": "Z Series",
    "Z4": "Z Series",
}
