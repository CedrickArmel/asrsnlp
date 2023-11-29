"""
This is a boilerplate pipeline 'train_safeaero_gpu'
generated using Kedro 0.18.14
"""
from typing import Union
from sklearn import metrics
from torch import nn, cuda
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer, AutoModel
from kedro_mlflow.io.metrics import MlflowMetricHistoryDataSet
from kedro_mlflow.io.models import MlflowModelSaverDataSet, MlflowModelLoggerDataSet
import mlflow
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
        for param in self.l1.parameters():
            param.requires_grad = False
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


def train_func(model: ModelClass,
               loss_func: nn.Module,
               optimizer: nn.Module,
               dataloader: DataLoader,
               device):
    """Trains the model.

    Args:
        model (ModelClass): The model instance to train.
        loss_func (nn.Module): The loss function to optimize.
        optimizer (nn.Module): The optimization algorithm.
        dataloader (DataLoader): PyTorch DataLoader instance.
        device (_type_): The device to use for computations. See PyTorch methods to\n
        detect your device.

    Returns:
        Loss value of the trainning step.
    """
    model.train()
    for _, data in enumerate(dataloader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def eval_func(model: ModelClass,
              dataloader: DataLoader,
              device):
    """Evaluate the model.

    Args:
        model (ModelClass): The model instance to train.
        dataloader (DataLoader): PyTorch DataLoader instance.
        device (_type_): The device to use for computations. See PyTorch methods to\n
        detect your device.

    Returns:
        Evaluation metrics
    """
    model.eval()
    fin_targets = []
    fin_outputs = []
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
    fin_outputs = (np.array(fin_outputs) >= 0.5).tolist()
    acc = metrics.accuracy_score(fin_targets, fin_outputs)
    f1_mic = metrics.f1_score(fin_targets, fin_outputs, average='micro')
    f1_mac = metrics.f1_score(fin_targets, fin_outputs, average='macro')
    return acc, f1_mic, f1_mac


def train_model(**kwargs):
    """Train the model (Trainning + Evaluation).
    kwargs:
        Args:
            model (Callable): The model instance to train.
            loss_func (Callable): The loss function to optimize.
            optimizer (Callable): The optimization algorithm.
            epochs (int): # epochs to train the model through.
            traindataloader (DataLoader): PyTorch DataLoader instance of your trainset.
            testdataloader (DataLoader): PyTorch DataLoader instance of your testset.
            device(_type_): The device to train your model on. See PyTorch methods to\n
            detect your device.
    """
    logger = logging.getLogger(__name__)
    for name, item in kwargs.items():
        if "loss" in str(type(item).__mro__).lower():
            loss_fn = item
        if "optim" in str(type(item).__mro__).lower():
            optimizer = item
        if "epoch" in name.lower():
            epochs = item
        if "device" in name.lower():
            device = item
        if isinstance(item, DataLoader) and "eval" in name.lower():
            evaldataloader = item
        if isinstance(item, DataLoader) and "train" in name.lower():
            traindataloader = item
        if isinstance(item, ModelClass):
            modelname = name + "model"
            model = item
    modelsaver = MlflowModelSaverDataSet(filepath=os.path.join("data/06_models",
                                                               modelname),
                                         flavor='mlflow.pytorch')
    modellogger = MlflowModelLoggerDataSet(flavor='mlflow.pytorch')
    losshistory = MlflowMetricHistoryDataSet(key="loss",
                                             save_args={"mode": "list"})
    acchistory = MlflowMetricHistoryDataSet(key="accuracy",
                                            save_args={"mode": "list"})
    f1michistory = MlflowMetricHistoryDataSet(key="f1_micro",
                                              save_args={"mode": "list"})
    f1machistory = MlflowMetricHistoryDataSet(key="f1_macro",
                                              save_args={"mode": "list"})
    model.to(device)
    lossvalues = []
    accvalues = []
    f1microvalues = []
    f1macrovalues = []
    epoch = 1
    while epoch <= epochs:
        logger.info("Starting epoch %s...", epoch)
        with mlflow.start_run(nested=True,
                              run_name=f"epoch_{epoch}"):
            loss = train_func(model=model,
                              loss_func=loss_fn,
                              optimizer=optimizer,
                              dataloader=traindataloader,
                              device=device)
            acc, f1mic, f1mac = eval_func(model, evaldataloader, device)
            lossvalues.append(loss)
            accvalues.append(acc)
            f1microvalues.append(f1mic)
            f1macrovalues.append(f1mac)
            modelsaver.save(model)
            modellogger.save(model)
        epoch += 1
    losshistory.save(lossvalues)
    acchistory.save(accvalues)
    f1michistory.save(f1microvalues)
    f1machistory.save(f1macrovalues)
    return model, {'loss': loss, 'accuracy': acc, 'f1_micro': f1mic, 'f1_macro': f1mac}
