"""
lazyqsar — unified CLI entry point.

Subcommands
-----------
lazyqsar setup [--descriptors] [--fit] [--only LIST] [--target-dir DIR]
    Install optional dependencies and download model checkpoints.
    --only       Comma-separated subset of descriptors to download: chemeleon, cddd, clamp.
                 Default: all three. Only meaningful with --descriptors.
    --target-dir Directory to write checkpoint files into (default: ~/.lazyqsar/).
                 Only meaningful with --descriptors.

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


_ALL_DESCRIPTORS = {"chemeleon", "cddd", "clamp"}


def _cmd_setup(args):
    if not args.descriptors and not args.fit:
        print("Nothing to do. Use --descriptors, --fit, or both.", file=sys.stderr)
        sys.exit(1)

    if not args.descriptors:
        for flag, name in [
            (args.only, "--only"),
            (args.target_dir, "--target-dir"),
            (args.cpu_torch, "--cpu-torch"),
        ]:
            if flag:
                print(
                    f"Warning: {name} has no effect without --descriptors.",
                    file=sys.stderr,
                )

    if args.fit:
        _setup_fit()

    if args.descriptors:
        _setup_descriptors(args)


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


def _setup_descriptors(args):
    from ..utils.setup import (
        install_torch,
        install_cpu_torch_force,
        install_chemprop,
        install_rdkit,
        install_fpsim2,
        download_chemeleon,
        download_cddd,
        download_clamp,
    )

    only = {d.strip() for d in args.only.split(",")} if args.only else _ALL_DESCRIPTORS
    unknown = only - _ALL_DESCRIPTORS
    if unknown:
        print(
            f"Unknown descriptor(s): {', '.join(sorted(unknown))}. "
            f"Valid options: {', '.join(sorted(_ALL_DESCRIPTORS))}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.cpu_torch:
        install_cpu_torch_force()
    else:
        install_torch()
    install_chemprop()
    install_rdkit()
    install_fpsim2()
    if "chemeleon" in only:
        download_chemeleon(target_dir=args.target_dir)
    if "cddd" in only:
        download_cddd(target_dir=args.target_dir)
    if "clamp" in only:
        download_clamp(target_dir=args.target_dir)


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
        predict_type=args.predict_type,
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
    p_setup.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated subset of descriptors to download: chemeleon, cddd, clamp "
            "(default: all three). Only meaningful with --descriptors."
        ),
    )
    p_setup.add_argument(
        "--target-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to download checkpoints into (default: ~/.lazyqsar/). Only meaningful with --descriptors.",
    )
    p_setup.add_argument(
        "--cpu-torch",
        action="store_true",
        help="Force-reinstall torch from PyTorch's CPU index, replacing any CUDA wheel pip may have installed via PyPI. Only meaningful with --descriptors.",
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
        default="slow",
        choices=["fast", "slow"],
        help="Descriptor mode: fast (Morgan only) or slow (all descriptors). Default: slow.",
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
    p_predict.add_argument(
        "--predict_type",
        type=str,
        default="proba",
        metavar="TYPE",
        choices=["proba", "rank", "logit", "lift", "score", "binary"],
        help="Type of prediction output (default: proba).",
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
