"""
This is a boilerplate pipeline 'train_bert'
generated using Kedro 0.18.14
"""
from sklearn import metrics
from torch import nn, cuda
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, AutoModel
from kedro_mlflow.io.metrics import MlflowMetricHistoryDataSet
from kedro_mlflow.io.artifacts import MlflowArtifactDataSet
from kedro_mlflow.io.models import MlflowModelSaverDataSet, MlflowModelLoggerDataSet
from typing import Union
import mlflow
import random
import os
import logging
import torch
import pandas as pd
import transformers as hf
import numpy as np
import torch_xla.core.xla_model as xm


class CustomDataset(Dataset):
    """PyTorch custom Dataset class. The PyTorch DataLoader will wrap an iterable\n
    around this CustomDataset to enable easy access to the samples.
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 max_len: int) -> None:
        """ This function is run once when instantiating\n
        the Dataset object.

        Args:
            dataframe (pd.DataFrame): Dataset object
            tokenizer (BertTokenizer): Tokenizer
            max_len (int): Model max lengh
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.narrative = dataframe.Narrative
        self.targets = self.data.Anomaly
        self.max_len = max_len

    def __len__(self) -> int:
        """Returns the number of samples in the dataframe.

        Returns:
            int: number of samples in the dataframe
        """
        return len(self.narrative)

    def __getitem__(self, index: int) -> dict:
        """Loads and returns a sample from the dataframe\n
        at the given index.

        Args:
            index (int): index

        Returns:
            dict: Training inputs
        """
        narrative = str(self.narrative.iloc[index])
        narrative = " ".join(narrative.split())

        inputs = self.tokenizer(
            narrative,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }


class ModelClass(torch.nn.Module):
    """PyTorch neural network model.
    """
    def __init__(self, modelname: str,
                 lastlayer: tuple,
                 dropoutratio: float):
        """ This function is run once when instantiating\n
        the Dataset object.
        """
        super(ModelClass, self).__init__()
        self.name = modelname
        self.ratio = dropoutratio
        self.ll = lastlayer
        self.l1 = AutoModel.from_pretrained(self.name)
        self.l2 = nn.Dropout(self.ratio)  # 0.3
        self.l3 = nn.Linear(self.ll[0], self.ll[1])  # 768, 14

    def forward(self, ids, mask, token_type_ids):
        """_summary_

        Args:
            ids (_type_): _description_
            mask (_type_): _description_
            token_type_ids (_type_): _description_

        Returns:
            _type_: _description_
        """
        _, output_1 = self.l1(ids, attention_mask=mask,
                              token_type_ids=token_type_ids,
                              return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


# Think about generic later that suport both maskedLM and base
def import_model(model_name: str,
                 lastlayer: Union[tuple, list],
                 doratio: float):
    """Returns the models to train form hugging face

    Args:
        model_name (str): Model name
        lastlayer (tuple): Config parameter to set de last layer of the pretrained model
        doratio (float): Config parameter for the dropout layer before the last layer

    Returns:
        the model and its tokenizer
    """
    llayer = tuple(lastlayer) if isinstance(lastlayer, list) else lastlayer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ModelClass(modelname=model_name,
                       lastlayer=llayer,
                       dropoutratio=doratio)
    return tokenizer, model


def loader(data: pd.DataFrame, tokenizer: hf.AutoTokenizer,
           max_len: int, trainparams: dict, sample: float = 1.0):
    """Tokenizes and Loads the data in a way suitable for torch and training/evaluation.

    Args:
        data (pd.DataFrame): Data to load for training
        tokenizer (hf.AutoTokenizer): Model tokenizer
        max_len (int): Number of tokens to pass to the model. Sentences of higher token 
        number are cut.
        trainparams (dict): Should contain batch_size, shuffle, num_workers
        sample (float, optional): Size of the subset to load. Defaults to 1.0

    Returns:
        Tokenized data for training/eval
    """
    dataset = CustomDataset(data, tokenizer, max_len)
    if sample > 0 and sample < 1:
        datasize = data.shape[0]
        sample_size = int(sample * datasize)
        subset_idx = np.random.random_integers(low=0,
                                               high=datasize,
                                               size=sample_size).tolist()
        randomsampler = SubsetRandomSampler(subset_idx)
        trainparams['shuffle'] = False
        loadeddata = DataLoader(dataset, **trainparams, sampler=randomsampler)
    else :
        loadeddata = DataLoader(dataset, **trainparams)

    return loadeddata


def get_cuda_device():
    """return the device available
    """
    device = 'cuda' if cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.empty_cache()
    return device


def get_xla_device():
    """return available XLA device
    """
    return xm.xla_device()


def get_loss():
    """Retuns the loss function"""
    return torch.nn.BCEWithLogitsLoss()


def get_optimizer(mymodel, learningrate: float):
    """Returns the Adam optimizer for the model
    """
    return torch.optim.Adam(params=mymodel.parameters(), lr=learningrate)


def train_model(**kwargs):
    """Train de model.
    kwargs:
        Args:
            mymodel (_type_): _description_
            loss_func (_type_): _description_
            optimizer (_type_): _description_
            epochs (int): _description_
            dataloader (_type_): _description_
            device (_type_): _description_        

    Returns:
        _type_: _description_
    """
    for name, item in kwargs.items():
        if "loss" in name.lower():
            loss_func = item
        if "optim" in name.lower():
            optimizer = item
        if "epoch" in name.lower():
            epochs = item
        if "device" in name.lower():
            device = item
        if isinstance(item, DataLoader):
            dataloader = item
        if isinstance(item, torch.nn.Module) and "loss" not in name.lower():
            modelname = name + "model"
            model = item
    modelsaver = MlflowModelSaverDataSet(filepath=os.path.join("data/06_models",
                                                               modelname),
                                         flavor='mlflow.pytorch')
    epochhistory = MlflowMetricHistoryDataSet(key="epochs_loss",
                                              save_args={"mode": "list"})
    model.to(device)
    epoch = 1
    losshistory = {}
    while epoch <= epochs:
        with mlflow.start_run(nested=True,
                              run_name=f"epoch_{epoch}"):
            model.train()
            for _, data in enumerate(dataloader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)
                optimizer.zero_grad()
                loss = loss_func(outputs, targets)
                losshistory[f"epoch_{epoch}"] = loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            modelsaver.save(model)
#           Ajouter ici le mlflow model logger
        epoch += 1
    epochhistory.save(losshistory.values())
    return model, {modelname: {"loss": losshistory}}


def eval_pt_model(**kwargs):
    """Evaluate the model.
    kwargs:
        Args:
            mymodel (_type_): _description_
            testingloader (_type_): _description_
            device (_type_): _description_
    Returns:
        _type_: _description_
    """
    for name, item in kwargs.items():
        if isinstance(item, torch.nn.Module):
            model = item
            modelname = name
        if isinstance(item, DataLoader):
            dataloader = item
        if "device" in name.lower():
            device = item
    model.to(device)
    model.eval()
    fin_targets = []
    fin_outputs = []
    with mlflow.start_run(nested=True,
                          run_name=f"eval_{modelname}"):
        with torch.no_grad():
            for _, data in enumerate(dataloader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu()
                                   .detach().numpy().tolist())
        fin_outputs = np.array(fin_outputs) >= 0.5
        acc = metrics.accuracy_score(fin_targets, outputs)
        f1_mic = metrics.f1_score(targets, outputs, average='micro')
        f1_mac = metrics.f1_score(targets, outputs, average='macro')
        evalres = dict(accuracy=acc,
                       f1_score_micro=f1_mic,
                       f1_score_macro=f1_mac)
    return evalres
