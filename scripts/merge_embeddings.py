
import os
import pickle
import glob
import argparse

def merge_pickles(input_dir, output_file):
    merged = {}
    files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    print(f"Merging {len(files)} files from: {input_dir}")

    for f in files:
        with open(f, "rb") as pf:
            data = pickle.load(pf)
            overlap = set(merged).intersection(data)
            if overlap:
                print(f"Warning: overlapping keys found in {f}, skipping duplicates.")
            merged.update(data)

    with open(output_file, "wb") as out:
        pickle.dump(merged, out)

    print(f"Saved merged file to: {output_file}")
    print(f"Total entries: {len(merged)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Name of the dataset, e.g. BioLiP")
    args = parser.parse_args()

    dataset = args.dataset
    base_dir = f"./Dataset/{dataset}"

    for split in ["train", "valid", "test"]:
        input_dir = os.path.join(base_dir, f"intermediate_{split}")
        output_file = os.path.join(base_dir, f"esm_{split}_{dataset}.pkl")
        merge_pickles(input_dir, output_file)
