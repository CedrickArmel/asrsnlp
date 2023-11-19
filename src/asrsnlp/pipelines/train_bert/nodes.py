"""
This is a boilerplate pipeline 'train_bert'
generated using Kedro 0.18.14
"""
import os
import logging
import torch
import pandas as pd
import transformers as hf
import numpy as np
from sklearn import metrics
from torch import nn, cuda
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


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


# Think about generique later that suport both maskedLM and base
def import_model(model_name: str, lastlayer: tuple, doratio: float):
    """Returns the models to train form hugging face

    Args:
        model_name (str): Model name
        lastlayer (tuple): Config parameter to set de last layer of the pretrained model
        doratio (float): Config parameter for the dropout layer before the last layer

    Returns:
        the model and its tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = ModelClass(modelname=model_name,
                       lastlayer=lastlayer,
                       dropoutratio=doratio)
    return tokenizer, model


def loader(data: pd.DataFrame, tokenizer: hf.AutoTokenizer,
           max_len: int, trainparams: dict):
    """Tokenizes and Loads the data in a way suitable for torch and training/evaluation.

    Args:
        data (pd.DataFrame): data to load for training
        tokenizer (hf.AutoTokenizer): Model tokenizer
        modelparams (dict): models params. Should contain 'max_len' key
        trainparams (dict): Should contain batch_size, shuffle, num_workers

    Returns:
        Tokenized data for training/eval
    """
    dataset = CustomDataset(data, tokenizer, max_len)
    loadeddata = DataLoader(dataset, **trainparams)
    return loadeddata


def get_device():
    """return the device available
    """
    device = 'cuda' if cuda.is_available() else 'cpu'
    return device


def get_loss():
    """Retuns the loss function"""
    return torch.nn.BCEWithLogitsLoss()


def get_optimizer(mymodel, learningrate: float):
    """Returns the Adam optimizer for the model
    """
    return torch.optim.Adam(params=mymodel.parameters(), lr=learningrate)


def train_model(mymodel,
                loss_func,
                optimizer,
                epochs: int,
                dataloader,
                device) :
    """Train de model

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
    model = mymodel
    model.to(device)
    epoch = 1
    while epoch <= epochs :
        model.train()  # tell PyTorch i'm training the model
        size = len(dataloader.dataset)
        for batch, data in enumerate(dataloader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)
            optimizer.zero_grad()
            loss = loss_func(outputs, targets)
            if batch % 1000 == 0:
                current = (batch + 1) * len(targets)
                print(f"Epoch: {epoch}, loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch += 1
    return model


def eval_pt_model(mymodel,
                  epochs,
                  testingloader,
                  device):
    """Evaluate the model

    Args:
        mymodel (_type_): _description_
        epochs (_type_): _description_
        testingloader (_type_): _description_
        device (_type_): _description_
    """
    model = mymodel
    model.to(device)
    epoch = 1
    while epoch <= epochs :
        model.eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(testingloader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = model(ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu()
                                   .detach().numpy().tolist())
        fin_outputs = np.array(fin_outputs) >= 0.5
        accuracy = metrics.accuracy_score(fin_targets, outputs)
        f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
        f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
        print(f"Accuracy Score = {accuracy}")
        print(f"F1 Score (Micro) = {f1_score_micro}")
        print(f"F1 Score (Macro) = {f1_score_macro}")
        epoch += 1

