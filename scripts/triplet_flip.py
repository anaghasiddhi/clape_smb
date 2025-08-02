import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import datetime

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.losses import TripletCenterLoss, FocalLoss
from utils.data import ProteinLigandData
from utils.model import SimpleModel

# ------------------ Parameters ------------------
model_params = {
    'lr': 1e-3,
    'margin': 1.0,
    'batch_size': 32,
    'backbone': 'simple'
}

ligand = 'FLIP'
training_time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
save_dir = f'triplet_classification/{ligand}{model_params["backbone"]}/{training_time}'
os.makedirs(save_dir, exist_ok=True)

# ------------------ Model Definition ------------------
class TripletNet(pl.LightningModule):
    def __init__(self, model_params):
        super(TripletNet, self).__init__()
        self.save_hyperparameters()
        self.full_model = SimpleModel()
        self.triplet_criterion = TripletCenterLoss(margin=model_params['margin'])
        self.clf_criterion = FocalLoss()
        self.lr = model_params['lr']

    def training_step(self, batch, batch_idx):
        feature, label = batch
        score, embedding = self.full_model(feature)
        triplet_loss = self.triplet_criterion(embedding, label)
        clf_loss = self.clf_criterion(score, label)
        loss = triplet_loss + clf_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        score, embedding = self.full_model(feature)  # ✅ correct unpacking
        triplet_loss = self.triplet_criterion(embedding, label)
        clf_loss = self.clf_criterion(score, label)
        loss = triplet_loss + clf_loss
        self.log('val_loss', loss)
        return {'preds': score.argmax(dim=1), 'targets': label}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass

# ------------------ Data Preparation ------------------
data_params = {
    'batch_size': model_params['batch_size'],
    'train_data_root': './Dataset/FLIP/esm_train_FLIP.pkl',
    'val_data_root': './Dataset/FLIP/esm_valid_FLIP.pkl',
}
data = ProteinLigandData(**data_params)

# ------------------ Training ------------------
model = TripletNet(model_params)

trainer = pl.Trainer(
    max_epochs=5,
    precision="16-mixed",
    devices=3,
    accelerator="gpu",
    strategy="ddp",  # Enables Distributed Data Parallel across 3 GPUs
    default_root_dir=save_dir,
    log_every_n_steps=1,
    enable_checkpointing=True
)

trainer.fit(
    model,
    train_dataloaders=data.train_dataloader(),
    val_dataloaders=data.val_dataloader()
)
