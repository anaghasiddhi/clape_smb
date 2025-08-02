import torch
import argparse
import pickle
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from utils.model import SimpleModel
from utils.data import LigandData, BatchCollate

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feature, label in dataloader:
            feature = feature.to(device)
            label = label.to(device)
            score, _ = model(feature)
            preds = score.argmax(dim=-1)

            mask = label != -1
            all_preds.extend(preds[mask].view(-1).cpu().tolist())
            all_targets.extend(label[mask].view(-1).cpu().tolist())

    flat_preds = all_preds
    flat_targets = all_targets

    acc = accuracy_score(flat_targets, flat_preds)
    precision = precision_score(flat_targets, flat_preds)
    recall = recall_score(flat_targets, flat_preds)
    f1 = f1_score(flat_targets, flat_preds)

    return acc, precision, recall, f1, flat_preds, flat_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name of the dataset (e.g. FLIP, ProteinNet)")
    parser.add_argument("checkpoint", help="Path to .ckpt checkpoint file")
    args = parser.parse_args()

    dataset = args.dataset
    ckpt_path = args.checkpoint

    # Load model
    model = SimpleModel()

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    # If it's already a plain state_dict
    if isinstance(checkpoint, dict) and "state_dict" not in checkpoint:
        state_dict = checkpoint
    else:
    # It's a Lightning-style checkpoint
        state_dict = {k.replace("full_model.", ""): v for k, v in checkpoint["state_dict"].items()}

    model.load_state_dict(state_dict, strict=False)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Load test data
    test_data_path = f"./Dataset/{dataset}/esm_test_{dataset}.pkl"
    if not os.path.exists(test_data_path):
        print(f" Test file not found: {test_data_path}")
        return

    test_dataset = LigandData(test_data_path)
    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=BatchCollate())
    
    device = next(model.parameters()).device
    acc, precision, recall, f1, all_preds, all_targets = evaluate(model, dataloader, device)

    print("\nTest Results:")
    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    

    # Save main evaluation metrics
    metrics_path = f"Results/metrics/{args.dataset}_test_metrics.csv"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write("Accuracy,Precision,Recall,F1\n")
        f.write(f"{acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f}\n")
    print(f"\nMetrics saved to {metrics_path}")

    # Save per-residue predictions vs. targets
    flat_preds = all_preds
    flat_targets = all_targets

    df = pd.DataFrame({
        "Prediction": flat_preds,
        "Target": flat_targets
    })

    predictions_path = f"Results/metrics/{args.dataset}_predictions.csv"
    df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    main()
