import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def read_ground_truth(path):
    ground_truth = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                pid, seq, label = parts
                ground_truth[pid] = label
    return ground_truth

def read_predictions(path):
    predictions = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            pid = lines[i].strip()
            seq = lines[i+1].strip()
            pred = lines[i+2].strip()
            predictions[pid] = pred
    return predictions

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, prec, rec, f1

def plot_confusion(y_true, y_pred, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non-binding", "Binding"],
                yticklabels=["Non-binding", "Binding"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, help="Ground truth TSV file (e.g. test_biolip.txt)")
    parser.add_argument("--pred", required=True, help="Predictions .txt file (3-line format)")
    parser.add_argument("--outdir", default="Results/metrics", help="Where to save metrics/plots")
    args = parser.parse_args()

    y_true_all = []
    y_pred_all = []

    truth_dict = read_ground_truth(args.truth)
    pred_dict = read_predictions(args.pred)

    for pid in pred_dict:
        if pid not in truth_dict:
            continue
        true = truth_dict[pid]
        pred = pred_dict[pid]
        if len(true) != len(pred):
            print(f"[!] Skipping {pid} due to length mismatch.")
            continue
        y_true_all.extend([int(c) for c in true])
        y_pred_all.extend([int(c) for c in pred])

    acc, prec, rec, f1 = compute_metrics(y_true_all, y_pred_all)
    print(f"\nEvaluation on BioLiP test set:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "biolip_eval_metrics.csv"), "w") as f:
        f.write("Accuracy,Precision,Recall,F1\n")
        f.write(f"{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f}\n")

    plot_confusion(y_true_all, y_pred_all, args.outdir)
    print(f"✅ Metrics and confusion matrix saved to: {args.outdir}")

if __name__ == "__main__":
    main()

