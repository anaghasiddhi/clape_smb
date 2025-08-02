import os
import pandas as pd
import sys
import time
import datetime
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics.classification import BinaryAUROC
from torchmetrics.classification import BinaryConfusionMatrix

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from utils.data import ProteinLigandData
from utils.model import SimpleModel
from utils.model import TransformerModel

# ------------------ Model Definition ------------------
class TripletNet(pl.LightningModule):
    def __init__(self, model_params):
        super().__init__()
        self.save_hyperparameters()
        self.full_model = TransformerModel()
        self.clf_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 8.0], device=self.device),ignore_index=-1)
        self.auroc = BinaryAUROC()
        self.confmat = BinaryConfusionMatrix()

        self.lr = model_params['lr']
        self.val_preds = []
        self.val_targets = []

    def training_step(self, batch, batch_idx):
        feature, label = batch
        score, _ = self.full_model(feature)

        B, L, C = score.shape
        score_flat = score.view(-1, C)
        label_flat = label.view(-1)

        loss = self.clf_criterion(score_flat, label_flat)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feature, label = batch
        score, _ = self.full_model(feature)
        preds = score.argmax(dim=-1)

        mask = label != -1
        preds_flat = preds[mask].detach().cpu().numpy()
        targets_flat = label[mask].detach().cpu().numpy()

        self.val_preds.extend(preds_flat.tolist())
        self.val_targets.extend(targets_flat.tolist())

        B, L, C = score.shape
        score_flat = score.view(-1, C)
        label_flat = label.view(-1)
        probs = torch.softmax(score_flat[mask.view(-1)], dim=1)[:, 1]
        labels = label_flat[mask.view(-1)]

        
        loss = self.clf_criterion(score_flat, label_flat)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.auroc.update(probs, labels)
        self.confmat.update((probs > 0.5).int(), labels)

        return loss.detach()

    def on_validation_epoch_end(self):
        y_true = self.val_targets
        y_pred = self.val_preds

        acc = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)
        self.log("val_f1", f1, sync_dist=True)

        self.val_preds.clear()
        self.val_targets.clear()
        auroc = self.auroc.compute()
        confmat = self.confmat.compute()

        self.log("val_auroc", auroc, prog_bar=True, sync_dist=True)

        tn, fp, fn, tp = confmat.flatten().tolist()
        self.log("val_TP", tp, prog_bar=False, sync_dist=True)
        self.log("val_FP", fp, prog_bar=False, sync_dist=True)
        self.log("val_FN", fn, prog_bar=False, sync_dist=True)
        self.log("val_TN", tn, prog_bar=False, sync_dist=True)

        self.auroc.reset()
        self.confmat.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ------------------ Main Training Entry ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train TripletNet")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--param_name', type=str, required=True, help='Parameter being tuned')
    parser.add_argument('--param_value', type=str, required=True, help='Value of that parameter'),
    parser.add_argument('--backbone', type=str, default='simple', choices=['simple', 'transformer'], help='Model backbone to use')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_params = {
        'lr': args.lr,
        'margin': 1.0,
        'batch_size': args.batch_size,
        'backbone': args.backbone  # dynamic backbone
    }

    data_params = {
        'batch_size': args.batch_size,
        'train_data_root': args.train_data,
        'val_data_root': args.val_data
    }

    training_time = datetime.datetime.now().strftime('%m-%d-%H-%M-%S')
    log_root = f"Results/logs/{args.dataset}/{args.backbone}/{args.param_name}/{args.param_value}"
    save_dir = os.path.join(log_root, "lightning_logs")
    os.makedirs(save_dir, exist_ok=True)

    data = ProteinLigandData(**data_params)
    model = TripletNet(model_params)
    log_dir = save_dir
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        dirpath=ckpt_dir,
        filename="epoch={epoch}-step={step}-val_loss={val_loss:.4f}"
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,
    mode='min'
    )

    logger = TensorBoardLogger(
    save_dir=save_dir,       # base directory
    name=args.dataset,             # subdirectory per dataset
    version=training_time          # use timestamp as run version
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        precision="16-mixed",
        devices=1,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        default_root_dir=save_dir,
        limit_val_batches=0.2,
        val_check_interval=1.0,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, lr_monitor, early_stop_callback]
    )

    start_time = time.time()
    trainer.fit(model, data.train_dataloader(), data.val_dataloader())
    elapsed = time.time() - start_time

    model_path = os.path.join(save_dir, "final_model.pt")
    torch.save(model.full_model.state_dict(), model_path)
    metrics = {
    "elapsed_minutes": elapsed / 60,
    "final_model_path": model_path,
    "param_name": args.param_name,
    "param_value": args.param_value,
    "backbone": args.backbone
    }
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)

    print(f"Saving results to: {save_dir}")
    print(f"\nFinal model saved to: {model_path}")
    print(f"Training completed in {elapsed/60:.2f} minutes")
