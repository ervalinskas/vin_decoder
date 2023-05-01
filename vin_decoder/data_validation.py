from pathlib import Path
from typing import Callable, List

from pandas import DataFrame

from config import config
from config.config import logger


def check_for_conflicts(df: DataFrame, col: str, gr_col: str = "vin") -> DataFrame:
    """Check if there are VINs with conflicting labels.

    Args:
        df (DataFrame): DataFrame object to run the checks against.
        col (str): column to check for conflicts.
        gr_col (str, optional): column for grouping the DataFrame. Defaults to "vin".

    Returns:
        DataFrame: filtered initial DataFrame containing only records with conflicting labels.
    """
    vin_col_df = df[[gr_col, col]].copy().drop_duplicates()
    non_null_vin_col_df = vin_col_df[vin_col_df[col].notnull()]
    non_null_vin_col_grouped = non_null_vin_col_df.groupby(gr_col)
    non_null_vin_col_conflicts = non_null_vin_col_grouped.filter(
        lambda x: x[col].nunique() > 1
    )
    return non_null_vin_col_conflicts


def get_conflicting_vins(
    df: DataFrame,
    col: str,
    gr_col: str = "vin",
    func_list: List[Callable | str] = [set, "count"],
) -> DataFrame:
    """Fetch exact conflicting values as a set.

    Args:
        df (DataFrame): DataFrame containing VINs with conflicting values.
        col (str): column to check for conflicts.
        func_list (List[Callable  |  str], optional): helper functions. Defaults to [set, "count"].

    Returns:
        DataFrame: DataFrame with unique VINs and unique conflicting values.
    """
    result = (
        df.groupby(gr_col).agg(func_list).rename(columns={"shortened_vin": "count"})
    )
    result.columns = result.columns.to_flat_index().str.join("_")
    result = result[result[col + "_count"] > 1].drop(columns=[col + "_count"])
    return result.reset_index()


def validate_labels(df: DataFrame, col: str, gr_col: str = "vin") -> None:
    """Validate vin-label pairs and save results to files.

    Args:
        df (DataFrame): DataFrame with vin-label pairs
        col (str): label column.
        gr_col (str, optional): column to conflicts against. Defaults to "vin".
    """
    check_df = check_for_conflicts(df=df, col=col)
    if not check_df.empty:
        logger.info(f"There are conflicts in the '{col}' column!")
        conflicts_df = get_conflicting_vins(df=check_df, col=col)

        good_vin_label_pairs = df[~df[gr_col].isin(conflicts_df[gr_col])]
        good_vin_label_pairs.to_csv(
            Path(config.VALIDATED_DATA_DIR, f"{gr_col}_{col}_pairs_good.csv"),
            index=False,
            header=True,
        )

        bad_vin_label_pairs = df[~df[gr_col].isin(conflicts_df[gr_col])]
        bad_vin_label_pairs.to_csv(
            Path(config.VALIDATED_DATA_DIR, f"{gr_col}_{col}_pairs_bad.csv"),
            index=False,
            header=True,
        )
    else:
        logger.info(f"There are no conflicts in the '{col}' column!")
