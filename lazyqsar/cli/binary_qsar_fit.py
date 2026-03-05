import argparse

from .api import fit


def main():

    parser = argparse.ArgumentParser(description="Fit a LazyBinaryQSAR model.")
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        help="Mode for the LazyBinaryQSAR (fast, default or slow).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing the training data.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the fitted model.",
    )
    args = parser.parse_args()

    fit(data_dir=args.data_dir, model_dir=args.model_dir, mode=args.mode)


if __name__ == "__main__":
    main()
