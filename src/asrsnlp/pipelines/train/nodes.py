"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.14
"""
import logging
import os
from typing import Union, Dict, List
import numpy as np
import mlflow
import pandas as pd
import transformers as hf
import torch_xla.core.xla_model as xm
import torch
from kornia.losses import BinaryFocalLossWithLogits
from kedro_mlflow.io.metrics import MlflowMetricHistoryDataSet
from kedro_mlflow.io.models import MlflowModelSaverDataSet, MlflowModelLoggerDataSet
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }


class SequenceClassifierModel(nn.Module):
    """SequenceClassifier class.\n
    This class imports the adequate model from HuggingFace with a classification head.
    """

    def __init__(self,
                 checkpoint: str,
                 nb_labels: int):
        """This function is run once when instantiating\n
        the SequenceClassifierModel Class.

        Args:
            checkpoint (str): Pretrained model name or path in the Huggingface librairy.
            nb_labels (int): Number of labels for the classification head.
        """
        super(SequenceClassifierModel, self).__init__()
        self.checkpoint = checkpoint
        self.nb_labels = nb_labels
        self.l1 = AutoModelForSequenceClassification.\
            from_pretrained(self.checkpoint,
                            num_labels=self.nb_labels)

    def forward(self,
                data):
        """Perform forward pass through the model.

        Args:
            data : Input data dictionary containing IDs, attention masks, \n
            and token type IDs.

        Returns:
            torch.Tensor : Logits produced by the model.
        """
        ids = data['ids']
        mask = data['attention_mask']
        token_type_ids = data['token_type_ids']
        output = self.l1(ids,
                         attention_mask=mask,
                         token_type_ids=token_type_ids,
                         return_dict=False)
        logits = output.logits
        return logits

    def tokenizer(self):
        """Return the tokenizer associated to the model

        Returns:
            _type_: _description_
        """
        return AutoTokenizer.from_pretrained(self.checkpoint)

    def _set_requires_grad(self,
                           layer: nn.Module,
                           requires_grad: bool = True):
        """Activate/deactivate gradient computation for the layer.

        Args:
            layer (nn.Module): Layer to activate/deactivate
            requires_grad (bool, optional): Boolean. Defaults to True.
        """
        for param in layer.parameters():
            param.requires_grad = requires_grad

    def _find_and_unfreeze_layers(self,
                                  module: torch.nn.Module,
                                  layers: Dict[str, Dict]) -> bool:
        """Go through the layers of module to find and unfreeze the specified \n
        layers of interest.

        Args:
            module (torch.nn.Module): The module those layers should be unfroze.
            layers (Dict[str, Dict]): Layers to unfreeze in the module

        Returns:
            bool : Wether at least one of the specified layer were fond.
        """
        found = False
        for group, attr in layers.items():
            if attr['attr'] is None:
                if hasattr(module, group):
                    if isinstance(attr['layers'], str) and (attr['layers'] == 'All'):
                        layer = getattr(module, group)
                        self._set_requires_grad(layer)
                        found = True
                    elif isinstance(attr['layers'], List):
                        for lay in attr['layers']:
                            try:
                                layer = getattr(getattr(module, group), str(lay))
                                self._set_requires_grad(layer)
                                found = True
                            except AttributeError:
                                logger.warning(
                                    "Layer %s not found in %s, so it was ignored.",
                                    lay,
                                    group)
                                continue
                else:
                    for child in module.children():
                        if self._find_and_unfreeze_layers(child, layers):
                            found = True
                            break
            else:
                if hasattr(module, group) and hasattr(getattr(module, group),
                                                      attr['attr']):
                    for lay in attr['layers']:
                        try:
                            layer = getattr(
                                getattr(getattr(module, group), attr['attr']), str(lay))
                            self._set_requires_grad(layer)
                            found = True
                        except AttributeError:
                            logger.warning(
                                "Layer %s not found in %s, so it was ignored.",
                                lay, group)
                            continue
                else:
                    for child in module.children():
                        if self._find_and_unfreeze_layers(child, layers):
                            found = True
                            break
        return found

    def set_trainable_layers(self,
                             layers: Dict[str, Dict]):
        """Set the layers to unfreeze for fine-tuning the model.

        This method allows specifying which layers of the model to unfreeze for \n
        fine-tuning, facilitating transfer learning scenarios.

        Args:
            layers (Dict[str, Dict]): A dictionary specifying the layers to unfreeze.\n
                Each key represents a layer type (e.g., 'encoder') with a \n
                corresponding dictionary value containing the attribute name holding \n
                the layers and the list of layers to unfreeze. \n
                Example: {'encoder': {'attr': 'layer', 'layers': [8, 9, 'dense']}}
                - 'attr': The attribute containing the layers. 'attr' can be None
                - 'layers': A list of layer indices or names to unfreeze. \n
                if not a list must be 'All'.
        """
        self._set_requires_grad(self, False)
        found = self._find_and_unfreeze_layers(self.l1, layers=layers)

        if not found:
            logger.warning("""None of the provided layers were found.\n
                           As this method initially freezes all layers before\n
                           attempting to unfreeze the specified one, all layers\n
                           are now frozen.""")

    def __getitem__(self, key):
        return getattr(self, key)


def import_model(checkpoint: str,
                 nb_labels: int):
    """Returns the models to train form hugging face.

    Args:
        checkpoint (str): Pretrained model name or path in the Huggingface librairy.
        nb_labels (int): Number of labels for the classification head.
    """
    model = SequenceClassifierModel(checkpoint=checkpoint,
                                    nb_labels=nb_labels)
    tokenizer = model.tokenizer()
    return tokenizer, model


def loader(data: pd.DataFrame,
           tokenizer: hf.AutoTokenizer,
           max_len: int,
           trainparams: dict,
           sample: float = 1.0):
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
    else:
        loadeddata = DataLoader(dataset, **trainparams)

    return loadeddata


def get_device():
    """return available device
    """
    device = xm.xla_device()
    if 'TPU' not in str(xm.xla_real_devices(devices=[device])):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            os.environ['PJRT_DEVICE'] = 'GPU'
            device = torch.device('cuda')
            try:
                _ = torch.tensor([1, 2]).to(device) + torch.tensor([1, 2]).to(device)
            except RuntimeError:
                device = torch.device('cuda:0')
                try:
                    _ = torch.tensor([1, 2]).to(device) + \
                        torch.tensor([1, 2]).to(device)
                except RuntimeError:
                    device = xm.xla_device()
    return device


def get_focal_loss(pos_weight: list[float] | None = None,
                   weight: list[float] | None = None,
                   alpha: float = 1,
                   gamma: float = 0.5,
                   reduction: str = 'mean'):
    """Retuns the loss function"""
    if pos_weight is None:
        if weight is None:
            loss = BinaryFocalLossWithLogits(alpha=alpha,
                                             gamma=gamma,
                                             reduction=reduction)
        else:
            loss = BinaryFocalLossWithLogits(alpha=alpha,
                                             gamma=gamma,
                                             reduction=reduction,
                                             weight=torch.tensor(weight))
    else:
        if weight is None:
            loss = BinaryFocalLossWithLogits(alpha=alpha,
                                             gamma=gamma,
                                             reduction=reduction,
                                             pos_weight=torch.tensor(pos_weight))
        else:
            loss = BinaryFocalLossWithLogits(pos_weight=torch.tensor(pos_weight),
                                             weight=torch.tensor(weight),
                                             gamma=gamma,
                                             alpha=alpha,
                                             reduction=reduction)
    return loss


def get_optimizer(model: SequenceClassifierModel,
                  learningrate: float):
    """Returns the Adam optimizer for the model

    Args:
        model (SequenceClassifierModel): Model
        learningrate (float): Learning rate
    """
    return torch.optim.Adam(params=model.parameters(), lr=learningrate)


def train_func(model: SequenceClassifierModel,
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
        optimizer.zero_grad()
        outputs = model(data)
        targets = data['targets'].to(device, dtype=torch.float)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        xm.mark_step()
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
    f1_mic = metrics.f1_score(fin_targets, fin_outputs,
                              average='micro', zero_division=1)
    f1_mac = metrics.f1_score(fin_targets, fin_outputs,
                              average='macro', zero_division=1)
    f1_by_class = metrics.f1_score(fin_targets, fin_outputs,
                                   average=None, zero_division=1)

    return acc, f1_mic, f1_mac, f1_by_class


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
            modelname = name
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
    f1byclasshistory = MlflowMetricHistoryDataSet(key="f1_by_class",
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
            acc, f1mic, f1mac, f1byclass = eval_func(model, evaldataloader, device)
            f1byclasshistory.save(f1byclass.tolist())
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
    f1byclasshistory.save(f1byclass.tolist())
    return model, {'loss': loss, 'accuracy': acc, 'f1_micro': f1mic, 'f1_macro': f1mac}
