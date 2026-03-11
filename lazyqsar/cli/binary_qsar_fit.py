import argparse

from ..api.binary_qsar_fit import fit


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
    parser.add_argument(
        "--models_txt",
        type=str,
        default=None,
        help="Path to a text file containing a list of specific models to fit. If not provided, all models in the data directory will be fitted (alphabetically)",
    )
    args = parser.parse_args()

    fit(data_dir=args.data_dir, model_dir=args.model_dir, models_txt=args.models_txt, mode=args.mode)


if __name__ == "__main__":
    main()
