
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm



"""
    A PyTorch Lightning module representing a bi-encoder used for contrastive Encoder
    
    Args:
        hl_features (int): Dimension of the input features (default: 1408)
        embed_features (int): Dimension of encoder output embedding features (default: 384)
    """
class HiddenLayerBiEncoder(pl.LightningModule):
  NUM_HL_FEATURES = 1408
  NUM_EMBED_FEATURES = 384
  label_embeds = None
  
  def __init__(self, hl_features=1408, embed_features=384):

    
    super(HiddenLayerBiEncoder, self).__init__()

    self.NUM_EMBED_FEATURES = embed_features
    self.NUM_HL_FEATURES = hl_features

    self.embedder = nn.Sequential(
      nn.Linear(self.NUM_HL_FEATURES, 720),
      nn.BatchNorm1d(720),
      
      nn.Linear(720, 640),
      nn.BatchNorm1d(640),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(640, 560),
      nn.BatchNorm1d(560),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(560, 560),
      nn.BatchNorm1d(560),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(560, 560),
      nn.BatchNorm1d(560),
      nn.ReLU(),
      nn.Dropout(p=0.2),

      nn.Linear(560, self.NUM_EMBED_FEATURES) 

    )
    
        
  def forward(self, x):
    x = self.embedder(x)
    return x
  


"""
  A PyTorch Lightning module for training the  contrastive Encoder.
  
  Args:
      model (nn.Module): The model to be used for the training
"""
class SimilarityTask(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        loss = self._shared_eval_step(batch, batch_idx)
        metrics = {'test_loss': loss}
        self.log_dict(metrics)
        return metrics

    def _weighted_loss(self, a):
      negative_mask = a <= 0.0
      a[negative_mask] = 0.0
      return torch.clamp(a, min=0.0, max=1.0)


    def _shared_eval_step(self, batch, batch_idx):
        xa, xb, y = batch
        y_a = self.model(xa)
        y_b = self.model(xb)
        sim = self._weighted_loss(F.cosine_similarity(y_a,y_b))
        ny = self._weighted_loss(y)
        loss = F.binary_cross_entropy(sim, ny)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=7e-4)


"""
    A PyTorch Dataset representing BEData.

    Args:
        bi_df (pd.DataFrame): DataFrame containing the data
        labelList (list): List of possible labels
        featureCol (str): Name of the feature column in the DataFrame
        labelCol (str): Name of the label column in the DataFrame
        sample_count (int): Number of samples to consider (default: 600000)
"""
class BEData(Dataset):
    def __init__(self, bi_df, labelList,
                featureCol, labelCol,
                sample_count=600000):
      self.data = {}    
      for lbl in labelList:
        self.data[lbl] = bi_df[bi_df[labelCol] == lbl][featureCol].tolist()
      self.len=sample_count
      self.labelList = labelList


    def __getitem__(self,index):
        smp = np.random.choice(self.labelList, 2, replace=True)        
        xa = torch.from_numpy(random.choice(self.data[smp[0]])).type(torch.FloatTensor)
        xb = torch.from_numpy(random.choice(self.data[smp[1]])).type(torch.FloatTensor)
        y=torch.tensor(int(smp[0]==smp[1])).type(torch.FloatTensor)
        return xa,xb,y

    def __len__(self):
        return self.len



"""
    Trains a contrastive encoder

    Args:
        bi_dataset : Trainingdataset
        labelList (list): List of possible labels
        featureCol (str): Name of the feature column in the dataset
        labelCol (str): Name of the label column in the dataset
        max_epochs (int): Maximum number of training epochs (default: 100)
        pretrained_model (nn.Module): Pretrained model to use as starting point (default: None)
        hl_features (int): Dimension of the input features (default: 1408)
        embed_features (int): Dimension of encoder output embedding features (default: 384)
        accelerator (str): Accelerator device to use (default: 'cpu')

    Returns:
        nn.Module: Trained contrastive encoder model
    """
def train_hidden_layer_biencoder(bi_dataset, labelList, featureCol,labelCol,
                                 max_epochs=100,
                                 pretrained_model = None,
                                 hl_features = 1408,
                                 embed_features = 384,
                                 accelerator = 'cpu'
                                 ):
  torch.autograd.set_detect_anomaly(True)

  logger = TensorBoardLogger("lightning_logs", name="HiddenLayerBiEncoder")
  if accelerator == "mps":
    accelerator = "mps"
  elif accelerator != "cpu":
    accelerator = "gpu"
  
  sz = len(bi_dataset)
  sz_train, sz_test = int(sz*0.7), int(sz*0.9)
  bi_train_df, bi_test_df,bi_val_df = (bi_dataset[0:sz_train],
                                       bi_dataset[sz_train:sz_test],
                                       bi_dataset[sz_test:])


  num_workers = 0 if accelerator == "mps" else 4

  train_loader=DataLoader(dataset= BEData(pd.DataFrame(bi_train_df),
                                          labelList, featureCol, labelCol,
                                          1200000),
                          num_workers=num_workers, batch_size=512)
  val_loader=DataLoader(dataset= BEData(pd.DataFrame(bi_val_df),
                                          labelList, featureCol,labelCol,
                                         100000),
                        num_workers=num_workers, batch_size=512)
  test_loader=DataLoader(dataset= BEData(pd.DataFrame(bi_test_df),
                                          labelList, featureCol, labelCol,
                                         100000),
                        num_workers=num_workers, batch_size=512)

  trainer = Trainer(accelerator=accelerator, devices=1,
                      logger=logger,
                      max_epochs=max_epochs)
  if pretrained_model is None:
    model = HiddenLayerBiEncoder(hl_features=hl_features,
                                 embed_features=embed_features)
  else:
    model = pretrained_model
    
  task = SimilarityTask(model)
  trainer.fit(task,            
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            )
    
  trainer.test(task, dataloaders=test_loader)
  return model
