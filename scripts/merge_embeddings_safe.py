import os
import pickle
import glob
import argparse

def merge_pickles_list(input_dir, output_file):
    all_entries = []
    files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    print(f"Merging {len(files)} files from: {input_dir}")

    for f in files:
        print(f"Loading: {f}")
        with open(f, "rb") as pf:
            data = pickle.load(pf)
            for pid, value in data.items():
                all_entries.append((pid, value))

    print(f"Total merged entries (with duplicates allowed): {len(all_entries)}")

    with open(output_file, "wb") as out:
        pickle.dump(all_entries, out)

    print(f"Saved merged file to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="e.g., BioLiP")
    args = parser.parse_args()

    dataset = args.dataset
    base_dir = f"./Dataset/{dataset}"

    for split in ["train", "valid", "test"]:
        input_dir = os.path.join(base_dir, f"intermediate_{split}")
        output_file = os.path.join(base_dir, f"esm_{split}_{dataset}.pkl")
        merge_pickles_list(input_dir, output_file)
