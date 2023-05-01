from pandas import DataFrame


def map_labels(df: DataFrame, label: str, d: dict) -> DataFrame:
    """Map labels to alternative values.

    Args:
        df (DataFrame): DataFrame containing VIN and label pairs.
        label (str): label to map
        d (dict): dictionary containing old/new label values as key/values.

    Returns:
        DataFrame: DataFrame with mapped labels.
    """
    df[label] = df[label].map(d).fillna(df[label])
    return df


def impute_missing_values(df: DataFrame, label: str) -> DataFrame:
    """Impute missing values.

    Args:
        df (DataFrame): DataFrame containing VIN and label pairs.
        label (str): label column to impute missing values.
        gr_col (str, optional): It's basically VIN. Defaults to "vin".

    Returns:
        DataFrame: DataFrame with imputed missing values.
    """
    df[f"{label}_new"] = df.groupby("vin")[label].fillna(method="ffill")
    df[f"{label}_new"] = df.groupby("vin")[f"{label}_new"].fillna(method="bfill")

    # Some VINs consist only of NaN values
    df = df[df[f"{label}_new"].notnull()]
    df = df[["vin", f"{label}_new"]].rename(columns={f"{label}_new": label})
    return df
