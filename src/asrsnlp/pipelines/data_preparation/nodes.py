"""
This is a boilerplate pipeline 'data_preparation'
generated using Kedro 0.18.14
"""

import os
import logging
import pandas
import numpy


def drop_useless(**kwargs) -> None:
    """Keep only useful columns in the dataframe.

    Args:
        data (str): Name of the dataset as defined in the catalog
    """
    filesuffix = ['raw', 'int', 'prm', 'fea', 'moi', 'mod', 'moo', 'rep', 'tra', 'ses']
    logger = logging.getLogger(__name__)
    for name, item in kwargs.items():
        nameroot = name[:-4] if name[-3:] in filesuffix else name
        filename = str(nameroot + "_prm.parquet")
        data = item[['Narrative', 'Anomaly', 'Synopsis']]
        try:
            logger.info("Storing the new dataset from %s in parquet format.", filename)
            data.to_parquet(os.path.join("data/03_primary", filename))
        except AttributeError as e:
            logger.error("Error occured : %s", e)
