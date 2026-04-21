"""
lazyqsar — unified CLI entry point.

Subcommands
-----------
lazyqsar setup [--descriptors] [--fit]
    Install optional dependencies and download model checkpoints.

lazyqsar fit --task classification --input DATA_DIR --output MODEL_DIR [--mode MODE] [--models_txt FILE]
    Fit a classifier on CSV data.

lazyqsar predict --input INPUT_CSV --model MODEL_DIR --output OUTPUT_CSV [--models_txt FILE]
    Run predictions with a saved model.
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


def _cmd_setup(args):
    if not args.descriptors and not args.fit:
        print("Nothing to do. Use --descriptors, --fit, or both.", file=sys.stderr)
        sys.exit(1)

    if args.fit:
        _setup_fit()

    if args.descriptors:
        _setup_descriptors()


def _setup_fit():
    import subprocess

    print(
        "Installing fit dependencies (sklearn, xgboost, scipy, skl2onnx, onnxmltools, joblib)..."
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "scikit-learn==1.6.1",
            "xgboost",
            "scipy",
            "onnxmltools",
            "onnxconverter-common==1.16.0",
            "skl2onnx==1.19.1",
            "joblib==1.5.1",
        ]
    )
    print("Fit dependencies installed.")


def _setup_descriptors():
    from ..utils.setup import (
        install_torch,
        install_chemprop,
        install_rdkit,
        install_fpsim2,
    )
    from ..utils.setup import download_chemeleon, download_cddd

    install_torch()
    install_chemprop()
    install_rdkit()
    install_fpsim2()
    download_chemeleon()
    download_cddd()


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def _cmd_fit(args):
    task = args.task.lower()
    if task == "classification":
        from ..api.classifier_fit import fit

        fit(
            data_dir=args.input,
            model_dir=args.output,
            models_txt=args.models_txt,
            mode=args.mode,
        )
    elif task == "regression":
        print("Error: regression task is not yet implemented.", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Unknown task {args.task!r}.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _cmd_predict(args):
    from ..api.classifier_predict import predict

    predict(
        model_dir=args.model,
        input_csv=args.input,
        output_csv=args.output,
        models_txt=args.models_txt,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="lazyqsar",
        description="LazyQSAR — fast QSAR modelling CLI",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # --- setup ---
    p_setup = sub.add_parser(
        "setup",
        help="Install optional dependencies and download model checkpoints.",
    )
    p_setup.add_argument(
        "--descriptors",
        action="store_true",
        help="Install descriptor dependencies and download Chemeleon / CDDD checkpoints.",
    )
    p_setup.add_argument(
        "--fit",
        action="store_true",
        help="Install fit dependencies (sklearn, xgboost, scipy, skl2onnx, onnxmltools, joblib).",
    )

    # --- fit ---
    p_fit = sub.add_parser(
        "fit",
        help="Fit a model.",
    )
    p_fit.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Modelling task.",
    )
    p_fit.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="DATA_DIR",
        help=(
            "Directory containing one CSV per task. "
            "Each CSV must have SMILES in the first column and labels in the second."
        ),
    )
    p_fit.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="MODEL_DIR",
        help="Directory where the fitted model will be saved.",
    )
    p_fit.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["fast", "default", "slow"],
        help="Descriptor mode (default: default).",
    )
    p_fit.add_argument(
        "--models_txt",
        type=str,
        default=None,
        metavar="FILE",
        help="Text file listing task names (CSV stems) to fit, one per line. Fits all tasks if omitted.",
    )

    # --- predict ---
    p_predict = sub.add_parser(
        "predict",
        help="Run predictions with a saved model.",
    )
    p_predict.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="INPUT_CSV",
        help="Input CSV with SMILES in the first column.",
    )
    p_predict.add_argument(
        "--model",
        type=str,
        required=True,
        metavar="MODEL_DIR",
        help="Directory containing the fitted model.",
    )
    p_predict.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="OUTPUT_CSV",
        help="Output CSV where predictions will be written.",
    )
    p_predict.add_argument(
        "--models_txt",
        type=str,
        default=None,
        metavar="FILE",
        help="Text file listing task names to predict. Uses all tasks if omitted.",
    )

    args = parser.parse_args()

    if args.command == "setup":
        _cmd_setup(args)
    elif args.command == "fit":
        _cmd_fit(args)
    elif args.command == "predict":
        _cmd_predict(args)


if __name__ == "__main__":
    main()
