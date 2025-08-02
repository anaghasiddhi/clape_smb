import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import pandas as pd

# Path to your TensorBoard log file
log_dir = "/home/scratch1/asiddhi/benchmarking_model1/Results/logs/BioLiP/05-18-16-27-40/lightning_logs/lightning_logs/version_0"
output_dir = "/home/scratch1/asiddhi/benchmarking_model1/Results/metrics/"
os.makedirs(output_dir, exist_ok=True)

# Load the events file
event_file = [f for f in os.listdir(log_dir) if f.startswith("events.out")][0]
ea = EventAccumulator(os.path.join(log_dir, event_file))
ea.Reload()

# Metrics you want to extract
tags = ["val_accuracy", "val_f1", "val_recall", "val_auroc"]
metric_data = {}

for tag in tags:
    if tag not in ea.Tags()["scalars"]:
        print(f"[!] Metric not found in logs: {tag}")
        continue
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    metric_data[tag] = values

    # Save plot
    plt.figure()
    plt.plot(steps, values, marker="o", label=tag)
    plt.xlabel("Epoch")
    plt.ylabel(tag)
    plt.title(f"{tag} over epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{tag}_per_epoch.png"))
    print(f"Saved: {tag}_per_epoch.png")

# Save all metrics to a CSV
df = pd.DataFrame(metric_data)
df.index.name = "Epoch"
df.to_csv(os.path.join(output_dir, "per_epoch_metrics.csv"))
print("Saved: per_epoch_metrics.csv")
