"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.14
"""

import logging
import os
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
    splited_cell_anomalies = {label: any(item.startswith(label)
                                         for item in cell_anomalies)
                              for label in labels}
    splited_cell_anomalies['Other'] = not any(splited_cell_anomalies.values())
    return pd.Series(splited_cell_anomalies)


def target_encoder(**kwargs):
    """_summary_

    Args:
        kwargs:
            Datasets: kwargs should key-value of the datasets
            target(str): All the passed datasets should contain this column.
            That's the column to encode.
            labels(list): list of the labels to encode.
    """
    # logger = logging.getLogger(__name__)
    # filesuffix = ['raw', 'int', 'prm', 'fea', 'moi', 'mod', 'moo', 'rep', 'tra', 'ses']
    data_list = []
    for name, item in kwargs.items():
        if "target" in name.lower():
            target = item
        if "labels" in name.lower():
            labels = tuple(item) if isinstance(item, list) else item
    for name, item in kwargs.items() :
        if isinstance(item, pd.DataFrame):
            # nameroot = name[:-4] if name[-3:] in filesuffix else name
            # filename = str(nameroot + "_fea.parquet")
            data = item
            encoding_series = data[target].apply(
                lambda cell: encode_cell(cell, labels))
            data[target] = encoding_series.values.tolist()
            data_list.append(data)
            # logger.info("Storing the new dataset from %s in parquet format.", name)
            # data.to_parquet(os.path.join("data/04_feature", filename))
    return data_list
