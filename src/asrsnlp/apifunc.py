"""
The codes below using Kaggle API will not work if credentials are not well set.\n
Visit https://github.com/Kaggle/kaggle-api#api-credentials for credentials \n
configuration.
"""


try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except OSError:
    pass

import zipfile
import os
import glob
from pathlib import Path
from typing import Union


def unziper(source: Union[str, Path],
            destination: Union[str, Path],
            keepzip: bool = True):
    """Method for unziping files.

    Args:
        source (str): _Source path of the file to unzip_
        destination (str): _Destination path to send the file to_
        keepzip (bool): _Whether keep the zip files after extraction_
    """
    if isinstance(source, Path):
        src = str(source)
    else:
        src = source

    zip_files = glob.glob(src)
    if keepzip:
        for file in zip_files :
            with zipfile.ZipFile(file, 'r') as zipsource:
                zipsource.extractall(destination)
    else:
        for file in zip_files :
            with zipfile.ZipFile(file, 'r') as zipsource:
                zipsource.extractall(destination)
            os.remove(file)


def dsdownloader(owner: str, dataset: str, output: str, uzip: bool = True):
    """Download dataset from Kaggle using Kaggle Python API. This function assumes\n
    that Kaggle credentials are well set.\n
    Visit https://github.com/Kaggle/kaggle-api#api-credentials for credentials \n
    configuration.

    Args:
        owner (str): _Owner of the dataset on Kaggle.com_
        dataset (str): _Name of the dataset_
        output (str): _Path to download to_
        uzip (bool, optional): _Whether to extract the zip files_. Defaults to True.
    """
    try:
        kaggle_client = KaggleApi()
        kaggle_client.authenticate()
        dsname = owner+"/"+dataset
        kaggle_client.dataset_download_files(dataset=dsname, path=output, unzip=uzip)
    except OSError:
        pass


def compfiles(comp: str, output: str):
    """Download competition files from Kaggle using Kaggle Python API. This function \n 
    assumes that Kaggle credentials are well set.\n
    Visit https://github.com/Kaggle/kaggle-api#api-credentials for credentials \n
    configuration.

    Args:
        comp (str): _Name of the competttion on Kaggle.com_
        output (str): _Path to download to_
        uzip (bool, optional): _Whether to extract the zip files_. Defaults to True.
    """
    try:
        kaggle_client = KaggleApi()
        kaggle_client.authenticate()
        kaggle_client.competition_download_files(competition=comp,
                                                 path=os.path.abspath(output))
        unziper(os.path.join(output, "*.zip"), output, keepzip=False)
    except OSError:
        pass
