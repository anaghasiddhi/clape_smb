import os
import re
import random

# Input and output paths
input_file = "/home/scratch1/asiddhi/benchmarking_model1/Raw_data/BioLip/BioLiP.txt"
output_dir = "/home/scratch1/asiddhi/benchmarking_model1/Dataset/BioLiP/splits"
os.makedirs(output_dir, exist_ok=True)

all_data = []

# Parse and convert each line
with open(input_file, "r") as fin:
    for line in fin:
        parts = line.strip().split("\t")
        if len(parts) < 21:
            continue

        pid = parts[0]
        sequence = parts[20].strip()
        binding_res = parts[8].strip().split()

        label = [0] * len(sequence)
        for res in binding_res:
            match = re.match(r"[A-Z](\d+)", res)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(label):
                    label[idx] = 1

        label_str = ''.join(map(str, label))
        all_data.append(f"{pid}\t{sequence}\t{label_str}\n")

# Shuffle and split into train/valid/test
random.seed(42)
random.shuffle(all_data)

n = len(all_data)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

splits = {
    "train": all_data[:n_train],
    "valid": all_data[n_train:n_train + n_val],
    "test":  all_data[n_train + n_val:]
}

# Write out the splits
for name, lines in splits.items():
    filename = os.path.join(output_dir, f"{name}_biolip.txt")
    with open(filename, "w") as fout:
        fout.writelines(lines)
    print(f"{name}: {len(lines)} entries written to {filename}")
