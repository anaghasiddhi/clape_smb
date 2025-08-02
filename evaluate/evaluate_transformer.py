import torch
import argparse
import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from torch.utils.data import DataLoader
from utils.data import LigandData, BatchCollate
from utils.model import TransformerModel  # Make sure this matches your path

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for feature, label in dataloader:
            feature = feature.to(device)
            label = label.to(device)
            logits, _ = model(feature)

            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            mask = label != -1
            masked_preds = preds[mask].view(-1).cpu().tolist()
            masked_probs = probs[mask][:, 1].cpu().tolist()  # prob of class 1
            masked_targets = label[mask].view(-1).cpu().tolist()

            all_preds.extend(masked_preds)
            all_probs.extend(masked_probs)
            all_targets.extend(masked_targets)

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    try:
        auroc = roc_auc_score(all_targets, all_probs)
    except:
        auroc = float('nan')

    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "fp": fp,
        "fn": fn,
        "preds": all_preds,
        "targets": all_targets,
        "masked_probs": masked_probs
    }, all_preds, all_targets, all_probs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="BioLiP")
    args = parser.parse_args()

    # Load model
    model = TransformerModel()
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    test_dataset = LigandData(args.test_data)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=BatchCollate())

    metrics, all_preds, all_targets, all_probs = evaluate(model, dataloader, device)

    print("\n🔍 Test Results:")
    for k in ["accuracy", "precision", "recall", "f1", "auroc", "fp", "fn"]:
        print(f" {k.upper():<9}: {metrics[k]:.4f}")

    # Save metrics
    os.makedirs("Results/metrics", exist_ok=True)
    metrics_path = f"Results/metrics/{args.dataset}_test_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"📁 Metrics saved to {metrics_path}")

    # Save predictions
    pred_path = f"Results/metrics/{args.dataset}_predictions.csv"
    pd.DataFrame({
        "Prediction": metrics["preds"],
        "Target": metrics["targets"],
        "Probability": all_probs
    }).to_csv(pred_path, index=False)
    print(f"📁 Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
