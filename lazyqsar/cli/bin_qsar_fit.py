import os
import csv
import argparse

from ..qsar import LazyBinaryQSAR


def main():

    parser = argparse.ArgumentParser(description="Fit a LazyBinaryQSAR model.")
    parser.add_argument("--mode", type=str, default="default", help="Mode for the LazyBinaryQSAR (fast, default or slow).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the training data.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save the fitted model.")
    args = parser.parse_args()

    for fn in os.listdir(args.data_dir):
        if not fn.endswith(".csv"):
            continue

        smiles_list = []
        y = []
        with open(os.path.join(args.data_dir, fn), "r") as f:
            reader = csv.reader(f)
            next(reader)
            for r in reader:
                smiles_list += [r[0]]
                y += [int(r[1])]
        
        model = LazyBinaryQSAR(mode=args.mode)
        model.fit(smiles_list, y)

        model_subdir = os.path.join(args.model_dir, os.path.splitext(fn)[0])
        model.save(model_subdir)


if __name__ == "__main__":
    main()





