"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.14
"""

import logging
import pandas as pd


def dropna_row(df: pd.DataFrame, sub: list) -> pd.DataFrame:
    """ Drop the row if there any missing value in either column in sub

    Args:
        df (pd.DataFrame): The dataframe
        sub (list): List of columns where to look for missings

    Returns:
        pd.DataFrame: _description_
    """
    data = df.dropna(axis=0, subset=sub)
    return data


def encode_cell(cell: pd.Series, labels: list) -> pd.Series:
    """Encode the multilabels cell such that the cell content is replaced by \n
    a list of same length as labels and containing 0/1.

    Args:
        cell (pd.Series): cell containing the multilabel target
        labels (list): actual list of labels to classify.

    Returns:
        pd.Series: Expand of the cell with number of cols\n
        equal to number of element in labels.
    """
    cell_anomalies = [item.strip() for item in cell.split(';')]
    splited_cell_anomalies = {label: any(item.startswith(labels)
                                         for item in cell_anomalies)
                              for label in labels}
    return pd.Series(splited_cell_anomalies)


def target_encoder(df: pd.DataFrame, target: str, labels: list) -> pd.DataFrame :
    """Encode the multilabels cells such that each cell is replaced by \n
    a list of same length as labels and containing 0/1.

    Args:
        df (pd.DataFrame): Task dataframe containing the multilabel target
        target (str): The multilabel target in df
        labels (list): actual list of labels to classify.

    Returns:
        pd.DataFrame: The dataframe with the encoded target
    """
    data = df
    encoding_series = data[target].apply(lambda cell: encode_cell(cell, labels))
    data[target] = encoding_series.values.tolist()
    return data
