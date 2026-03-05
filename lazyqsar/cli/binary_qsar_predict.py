import argparse

from .api import predict


def main():
    parser = argparse.ArgumentParser(description="Predict with a LazyBinaryQSAR model.")

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the fitted model.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file containing the SMILES strings to predict on.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output file to save the predictions.",
    )
    parser.add_argument(
        "--models_txt",
        type=str,
        default=None,
        help="Path to a text file containing a list of specific models to use for prediction. If not provided, all models in the model directory will be used (alphabetically).",
    )
    args = parser.parse_args()

    models = None
    if args.models_txt is not None:
        with open(args.models_txt, "r") as f:
            models = [line.strip() for line in f]

    predict(model_dir=args.model_dir, input_csv=args.input_csv, output_csv=args.output_csv, models=models)


if __name__ == "__main__":
    main()
