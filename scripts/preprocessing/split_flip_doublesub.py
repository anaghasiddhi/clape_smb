import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Paths
input_file = "../Dataset/FLIP/parsed_data/flip_doublesub_clean.txt"
output_dir = "../Dataset/FLIP/splits"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(input_file, sep="\t", names=["id", "sequence", "label"])

# Split data
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save splits
train_df.to_csv(os.path.join(output_dir, "train_flip.txt"), sep="\t", index=False, header=False)
valid_df.to_csv(os.path.join(output_dir, "valid_flip.txt"), sep="\t", index=False, header=False)
test_df.to_csv(os.path.join(output_dir, "test_flip.txt"), sep="\t", index=False, header=False)

print("✅ Split complete: train/valid/test saved in Dataset/FLIP/splits/")
