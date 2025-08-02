import torch
import argparse
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from utils.model import TransformerModel  # ✅ Update this import if needed
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

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)

    return acc, precision, recall, f1, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="BioLiP")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to final_model.pt")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test .pkl folder")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    # Load model
    model = TransformerModel({
        "lr": 1e-4,
        "batch_size": args.batch_size,
        "backbone": "transformer"
    })

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = {k.replace("full_model.", ""): v for k, v in checkpoint["state_dict"].items()}
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load test set
    test_dataset = LigandData(args.test_data)
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=BatchCollate())

    acc, precision, recall, f1, preds, targets = evaluate(model, dataloader, device)

    print("\n🔍 Test Results:")
    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1 Score:  {f1:.4f}")

    # Save metrics and predictions
    os.makedirs("Results/metrics", exist_ok=True)

    metrics_path = f"Results/metrics/{args.dataset}_test_metrics.csv"
    pd.DataFrame([{
        "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1
    }]).to_csv(metrics_path, index=False)
    print(f"📁 Metrics saved to {metrics_path}")

    pred_path = f"Results/metrics/{args.dataset}_predictions.csv"
    pd.DataFrame({"Prediction": preds, "Target": targets}).to_csv(pred_path, index=False)
    print(f"📁 Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()
