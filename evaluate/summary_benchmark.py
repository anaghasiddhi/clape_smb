import pandas as pd
from sklearn.metrics import confusion_matrix

# === Helper to compute specificity and sensitivity ===
def compute_specificity_sensitivity(pred_csv_path):
    df = pd.read_csv(pred_csv_path)
    y_true = df["Target"]
    y_pred = df["Prediction"]
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    return round(specificity, 4), round(sensitivity, 4)

# === Load test metrics ===
df_uni = pd.read_csv("Results/metrics/UniProtSMB_test_metrics.csv")
df_biolip = pd.read_csv("Results/metrics/BioLiP_test_metrics.csv")
df_crypto = pd.read_csv("Results/metrics/Cryptobench_test_metrics.csv")

# Add dataset labels
df_uni["Dataset"] = "UniProtSMB"
df_biolip["Dataset"] = "BioLiP"
df_crypto["Dataset"] = "Cryptobench"

# Compute specificity and sensitivity from predictions
spec_uni, sens_uni = compute_specificity_sensitivity("Results/metrics/UniProtSMB_predictions.csv")
spec_biolip, sens_biolip = compute_specificity_sensitivity("Results/metrics/BioLiP_predictions.csv")
spec_crypto, sens_crypto = compute_specificity_sensitivity("Results/metrics/Cryptobench_predictions.csv")

# Add to DataFrames
df_uni["Specificity"] = spec_uni
df_uni["Sensitivity"] = sens_uni

df_biolip["Specificity"] = spec_biolip
df_biolip["Sensitivity"] = sens_biolip

df_crypto["Specificity"] = spec_crypto
df_crypto["Sensitivity"] = sens_crypto

# Combine all
summary_df = pd.concat([df_uni, df_biolip, df_crypto], ignore_index=True)

# Reorder columns
cols = ["Dataset", "Accuracy", "Precision", "Sensitivity", "Specificity", "Recall", "F1"]
summary_df = summary_df[cols]

# Save and print
summary_df.to_csv("Results/metrics/Model_Performance_Comparison.csv", index=False)
print(summary_df)
