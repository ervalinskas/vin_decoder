from typing import Dict, List, Optional, Tuple, Union

from catboost import CatBoostClassifier, Pool
from numpy import array, ndarray
from pandas import DataFrame, Series, concat
from sklearn.feature_extraction.text import (
    CountVectorizer,
    HashingVectorizer,
    TfidfVectorizer,
)
from sklearn.model_selection import StratifiedKFold

from config.config import logger

from .feature_engineering import vectorize_text


def create_pools(
    X_train: ndarray, y_train: array, X_valid: ndarray, y_valid: array
) -> Tuple[Pool]:
    train = Pool(data=X_train, label=y_train)
    valid = Pool(data=X_valid, label=y_valid)
    return (train, valid)


def train_clf(
    train: Pool,
    learning_rate: float,
    depth: int,
    iterations: int,
    class_names: Optional[array] = None,
    valid: Optional[Pool] = None,
) -> CatBoostClassifier:
    inner_model = CatBoostClassifier(
        objective="MultiClass",
        learning_rate=learning_rate,
        depth=depth,
        iterations=iterations,
        random_seed=1,
        eval_metric="Accuracy",
        use_best_model=False,
        logging_level="Silent",
        class_names=class_names,
        devices="0:1",
    )

    inner_model.fit(train, eval_set=valid)
    return inner_model


def nested_cv(
    X: Series,
    y: Series,
    class_names: array,
    vectorizers: List[Union[CountVectorizer, HashingVectorizer, TfidfVectorizer]],
    grid: Dict,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 1,
) -> DataFrame:
    # TODO: Use hyperopt for param tuning
    cv_outer = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )
    cv_inner = StratifiedKFold(
        n_splits=n_splits, shuffle=shuffle, random_state=random_state
    )

    outer_result = []
    outer_fold_idx = 1

    # Outer loop
    for outer_train_idx, outer_valid_idx in cv_outer.split(X, y):
        X_outer_train, X_outer_valid = X.iloc[outer_train_idx], X.iloc[outer_valid_idx]
        y_outer_train, y_outer_valid = y.iloc[outer_train_idx], y.iloc[outer_valid_idx]

        inner_fold_idx = 1
        all_inner_fold_results = []

        logger.info(f"Running {outer_fold_idx} outer fold...")

        # Inner loop
        for inner_train_idx, inner_valid_idx in cv_inner.split(
            X_outer_train, y_outer_train
        ):
            X_inner_train, X_inner_valid = (
                X.iloc[inner_train_idx],
                X.iloc[inner_valid_idx],
            )
            y_inner_train, y_inner_valid = (
                y.iloc[inner_train_idx],
                y.iloc[inner_valid_idx],
            )

            inner_result = []

            logger.info(f"Running {inner_fold_idx} inner fold...")

            # Text-to-features loop
            for inner_vectorizer in vectorizers.keys():
                X_inner_train_featurized, X_inner_valid_featurized = vectorize_text(
                    vectorizer=vectorizers[inner_vectorizer],
                    X_train=X_inner_train,
                    X_valid=X_inner_valid,
                )

                # Hyperparam tuning loop
                for learning_rate in grid["learning_rate"]:
                    for depth in grid["depth"]:
                        for iterations in grid["iterations"]:
                            train_inner_pool, valid_inner_pool = create_pools(
                                X_train=X_inner_train_featurized,
                                y_train=y_inner_train,
                                X_valid=X_inner_valid_featurized,
                                y_valid=y_inner_valid,
                            )

                            inner_model = train_clf(
                                train=train_inner_pool,
                                valid=valid_inner_pool,
                                learning_rate=learning_rate,
                                depth=depth,
                                iterations=iterations,
                                class_names=class_names,
                            )

                            inner_metrics = inner_model.eval_metrics(
                                data=valid_inner_pool,
                                metrics=["Accuracy"],
                                ntree_start=iterations - 1,
                            )

                            logger.info(
                                f"""Text to features method - {inner_vectorizer} | learning rate - {learning_rate} | depth - {depth} | iterations - {iterations} | Accuracy - {inner_metrics["Accuracy"][0]}"""
                            )

                            inner_result.append(
                                {
                                    "outer-fold": outer_fold_idx,
                                    "inner-fold": inner_fold_idx,
                                    "vectorizer": inner_vectorizer,
                                    "learning_rate": learning_rate,
                                    "depth": depth,
                                    "iterations": iterations,
                                    "accuracy": inner_metrics["Accuracy"][0],
                                }
                            )
            inner_fold_idx += 1
            inner_result_df = DataFrame(inner_result)
            all_inner_fold_results.append(inner_result_df)

        all_inner_fold_results_df = concat(all_inner_fold_results)
        all_inner_fold_results_agg_df = (
            all_inner_fold_results_df.groupby(
                ["vectorizer", "learning_rate", "depth", "iterations"]
            )["accuracy"]
            .agg("mean")
            .reset_index()
            .rename(columns={"accuracy": "mean_accuracy"})
        )

        best_inner_params = all_inner_fold_results_agg_df[
            all_inner_fold_results_agg_df["mean_accuracy"]
            == all_inner_fold_results_agg_df["mean_accuracy"].max()
        ].to_dict("records")[0]

        X_outer_train_featurized, X_outer_valid_featurized = vectorize_text(
            vectorizer=vectorizers[best_inner_params["vectorizer"]],
            X_train=X_outer_train,
            X_valid=X_outer_valid,
        )

        train_outer_pool, valid_outer_pool = create_pools(
            X_train=X_outer_train_featurized,
            y_train=y_outer_train,
            X_valid=X_outer_valid_featurized,
            y_valid=y_outer_valid,
        )

        outer_model = train_clf(
            train=train_outer_pool,
            valid=valid_outer_pool,
            learning_rate=best_inner_params["learning_rate"],
            depth=best_inner_params["depth"],
            iterations=best_inner_params["iterations"],
            class_names=class_names,
        )

        # Outer model evaluation
        outer_metrics = outer_model.eval_metrics(
            data=valid_outer_pool,
            metrics=["Accuracy"],
            ntree_start=best_inner_params["iterations"] - 1,
        )

        outer_result.append(
            {
                "outer-fold": outer_fold_idx,
                "vectorizer": best_inner_params["vectorizer"],
                "learning_rate": best_inner_params["learning_rate"],
                "depth": best_inner_params["depth"],
                "iterations": best_inner_params["iterations"],
                "accuracy": outer_metrics["Accuracy"][0],
            }
        )

        outer_fold_idx += 1

    # Generalization performance estimation
    outer_folds_df = DataFrame(outer_result)
    return outer_folds_df


def pick_best_params(df: DataFrame) -> Dict:
    hyperparam_voting_df = (
        df.groupby(["vectorizer", "learning_rate", "depth", "iterations"])["outer-fold"]
        .agg("count")
        .reset_index()
        .rename(columns={"outer-fold": "count"})
    )

    best_params = (
        hyperparam_voting_df[
            hyperparam_voting_df["count"] == hyperparam_voting_df["count"].max()
        ]
        .drop(columns=["count"])
        .to_dict("records")[0]
    )
    return best_params


def train_best_model(
    X: Series,
    y: Series,
    best_params: dict,
    best_vectorizer: Union[CountVectorizer, HashingVectorizer, TfidfVectorizer],
) -> CatBoostClassifier:
    corpus_featurized = best_vectorizer.fit_transform(X)
    train = Pool(data=corpus_featurized, label=y)
    best_model = train_clf(train=train, **best_params)
    return best_vectorizer, best_model
