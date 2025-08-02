import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def parse_ground_truth(tsv_path):
    true_labels = {}
    with open(tsv_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                pid, seq, label = parts
                true_labels[pid] = list(map(int, list(label)))
    return true_labels

def parse_predictions(pred_path):
    pred_labels = {}
    with open(pred_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            pid = lines[i].strip()
            seq = lines[i+1].strip()
            preds = list(map(int, list(lines[i+2].strip())))
            pred_labels[pid] = preds
    return pred_labels

def evaluate(true_labels, pred_labels):
    all_true = []
    all_pred = []

    for pid in true_labels:
        if pid not in pred_labels:
            continue
        true_seq = true_labels[pid]
        pred_seq = pred_labels[pid]
        min_len = min(len(true_seq), len(pred_seq))  # safeguard
        all_true.extend(true_seq[:min_len])
        all_pred.extend(pred_seq[:min_len])

    acc = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)

    return acc, precision, recall, f1, all_true, all_pred

def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Confusion matrix saved to: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', required=True, help='Path to test_biolip.txt (ground truth)')
    parser.add_argument('--preds', required=True, help='Path to CLAPE-SMB predictions txt file')
    parser.add_argument('--outdir', default='Results/metrics/', help='Where to save metrics')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    gt = parse_ground_truth(args.labels)
    preds = parse_predictions(args.preds)

    acc, precision, recall, f1, y_true, y_pred = evaluate(gt, preds)

    print("\nEvaluation Results:")
    print(f" Accuracy:  {acc:.4f}")
    print(f" Precision: {precision:.4f}")
    print(f" Recall:    {recall:.4f}")
    print(f" F1 Score:  {f1:.4f}")

    metrics_path = os.path.join(args.outdir, "clape_smb_metrics.csv")
    pd.DataFrame([{
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }]).to_csv(metrics_path, index=False)

    print(f"\nMetrics saved to: {metrics_path}")

    cm_path = os.path.join(args.outdir, "clape_smb_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)

if __name__ == "__main__":
    main()
