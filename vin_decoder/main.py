import sys
import tempfile
import warnings
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import pandas as pd
import typer

# TODO: Fix this!
sys.path.append(Path(__file__).parent.parent.absolute().as_posix())

from config import config
from config.config import logger
from vin_decoder.data_validation import validate_labels
from vin_decoder.feature_engineering import extract_features_labels
from vin_decoder.label_preprocessing import impute_missing_values, map_labels
from vin_decoder.model_training import nested_cv, pick_best_params, train_best_model
from vin_decoder.predict import predict
from vin_decoder.utils import save_txt

warnings.filterwarnings("ignore")

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def extract_data():
    """_summary_"""
    vins = pd.read_csv(config.DATA_URL)
    path_to_save = Path(config.RAW_DATA_DIR, "ml-engineer-challenge-redacted-data.csv")
    vins.to_csv(path_to_save, index=False)
    logger.info(f"✅ VINs saved to - {path_to_save}")


@app.command()
def validate_data():
    """_summary_"""
    vins = pd.read_csv(
        Path(config.RAW_DATA_DIR, "ml-engineer-challenge-redacted-data.csv")
    )
    vins.drop_duplicates()

    for col in config.labels_to_validate:
        validate_labels(df=vins, col=col)

    logger.info("✅ Data validation is finished!")


@app.command()
def preprocess_labels():
    for label in config.labels_to_preprocess:
        df = pd.read_csv(Path(config.VALIDATED_DATA_DIR, f"vin_{label}_pairs_good.csv"))
        df = df[["vin", label]].drop_duplicates()
        for d in [config.models_to_group_1, config.models_to_group_2]:
            df = map_labels(df=df, label=label, d=d)
        df = impute_missing_values(df=df, label=label)
        df.to_csv(
            Path(config.PREPROCESSED_LABELS, f"vin_{label}_pairs_w_labels.csv"),
            index=False,
        )
    logger.info("✅ Label preprocessing is finished!")


@app.command()
def optimize_train_model(experiment_name: str = "baseline_vin_decoder"):
    for label in config.labels_to_train:
        df = pd.read_csv(
            Path(config.PREPROCESSED_LABELS, f"vin_{label}_pairs_w_labels.csv")
        )
        corpus, targets, class_names = extract_features_labels(df, label=label)
        mlflow.set_experiment(experiment_name=experiment_name)
        with mlflow.start_run(run_name=f"{label}-clf"):
            run_id = mlflow.active_run().info.run_id
            logger.info(f"Run ID: {run_id}")
            save_txt(filepath="./config/run_id.txt", file_body=run_id)

            outer_folds_df = nested_cv(
                X=corpus,
                y=targets,
                class_names=class_names,
                vectorizers=config.vectorizers,
                grid=config.grid,
                n_splits=config.n_splits,
            )

            # Log outer_fold result to MLflow
            mlflow.log_dict(outer_folds_df.to_dict("list"), "outer-folds")
            mlflow.log_metrics(
                {
                    "Mean accuracy": outer_folds_df["accuracy"].values.mean(),
                    "Std": outer_folds_df["accuracy"].values.std(),
                }
            )

            logger.info(
                f"Generalized accuracy for classifying '{label}' classes - {outer_folds_df['accuracy'].values.mean()} +/- {outer_folds_df['accuracy'].values.std()} \n "
            )

            best_params = pick_best_params(df=outer_folds_df)

            # Log best params to MLflow
            mlflow.log_params(best_params)

            logger.info(
                f"Best params for classifying '{label}' classes - vectorizer='{best_params['vectorizer']}', learning_rate='{best_params['learning_rate']}', depth='{best_params['depth']}', iterations='{best_params['iterations']}' \n "
            )

            best_vectorizer = config.vectorizers[best_params.pop("vectorizer")]
            best_vectorizer_pkl, best_model_pkl = train_best_model(
                X=corpus,
                y=targets,
                best_params=best_params,
                best_vectorizer=best_vectorizer,
            )

            # Log the best model as artifact
            with tempfile.TemporaryDirectory() as dp:
                joblib.dump(best_vectorizer_pkl, Path(dp, "vectorizer.pkl"))
                joblib.dump(best_model_pkl, Path(dp, "model.pkl"))
                mlflow.log_artifacts(dp)


def load_artifacts(run_id: str = None) -> Tuple:
    """Load vectorizer.pkl and model.pkl artifacts for a given run_id.
    Args:
        run_id (str): id of run to load artifacts from.
    Returns:
        Tuple: tuple with artifacts.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read().strip()

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")
    vectorizer = joblib.load(Path(artifacts_dir, "vectorizer.pkl"))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))

    return vectorizer, model


@app.command()
def predict_vehicle_model(vin: str = "", run_id: str = None) -> str:
    """Predict model of a vehicle from a VIN.

    Args:
        vin (str): vin to predict the model for.
        run_id (str, optional): id of run to load artifacts from. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read().strip()
    vectorizer, model = load_artifacts(run_id=run_id)
    predicted_class = predict(vectorizer=vectorizer, model=model, vins=[vin])
    logger.info(predicted_class)
    return predicted_class


if __name__ == "__main__":
    app()
