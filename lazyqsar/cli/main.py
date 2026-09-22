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
import os
import sys


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


_ALL_DESCRIPTORS = {"chemeleon", "cddd", "clamp"}


def _cmd_setup(args):
    reference = getattr(args, "reference", False)
    if not args.descriptors and not args.fit and not reference:
        print(
            "Nothing to do. Use --descriptors, --fit, --reference, or a combination.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.descriptors and not reference:
        for flag, name in [
            (args.only, "--only"),
            (args.target_dir, "--target-dir"),
        ]:
            if flag:
                print(
                    f"Warning: {name} has no effect without --descriptors "
                    "or --reference.",
                    file=sys.stderr,
                )
    if args.cpu_torch and not args.descriptors:
        print(
            "Warning: --cpu-torch has no effect without --descriptors.", file=sys.stderr
        )

    if args.fit:
        _setup_fit()

    if args.descriptors:
        _setup_descriptors(args)

    # Deliberately not implied by --descriptors: a fast-mode model needs one 6.5 MB matrix,
    # and the whole bundle is 267 MB. Asking for it is the opt-in.
    if reference:
        _setup_reference(args)


def _reference_descriptors(only):
    """Which reference matrices to fetch: the named subset, or everything shipped."""
    from ..registry import DESCRIPTOR_TYPES

    known = set(DESCRIPTOR_TYPES)
    if not only:
        return sorted(known)
    names = [n.strip() for n in only.split(",") if n.strip()]
    unknown = sorted(set(names) - known)
    if unknown:
        print(
            f"Unknown descriptor(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}.",
            file=sys.stderr,
        )
        sys.exit(1)
    return names


def _setup_reference(args):
    from ..reference import identity
    from ..reference.download import ReferenceDownloadError, download

    if args.target_dir:
        os.environ["LAZYQSAR_HOME"] = args.target_dir

    names = _reference_descriptors(args.only)
    n = identity.default_n()
    files = [identity.smiles_filename(n)] + [
        identity.descriptor_filename(name, n) for name in names
    ]
    try:
        download(files, n=n)
    except ReferenceDownloadError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Reference library ready in {identity.reference_dir()}")


def _extra_requirements(extra: str) -> list:
    """The requirements of one optional-dependency group, read from package metadata.

    Read rather than restated. This list used to be a hand-maintained copy of the ``fit``
    extra, and it had already drifted: ``xgboost`` and ``onnxmltools`` were unpinned here
    while ``pyproject.toml`` pinned them exactly, so ``lazyqsar setup --fit`` could install
    a different set of versions from ``pip install "lazyqsar[fit]"``. Deriving both from
    the same metadata removes the class of bug rather than re-syncing the copy.
    """
    from importlib.metadata import requires

    marker = f'extra == "{extra}"'
    specs = []
    for req in requires("lazyqsar") or []:
        if marker not in req:
            continue
        spec = req.split(";", 1)[0]
        # Metadata spells these "scikit-learn (==1.6.1)"; pip wants "scikit-learn==1.6.1".
        spec = spec.replace("(", "").replace(")", "").replace(" ", "")
        # `all` is expressed as a self-reference; installing it here would recurse.
        if spec.lower().startswith("lazyqsar"):
            continue
        specs.append(spec)
    return specs


def _setup_fit():
    import subprocess

    specs = _extra_requirements("fit")
    if not specs:
        print(
            "Could not read the 'fit' extra from package metadata; "
            'install it directly with: pip install "lazyqsar[fit]"',
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Installing fit dependencies: {', '.join(specs)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *specs])
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


def _cmd_reference(args):
    from ..reference import identity, store
    from ..reference.download import ReferenceDownloadError, download

    n = identity.default_n()

    if args.action == "status":
        state = store.status(n)
        print(f"reference : {state['reference_id']}  (tier {state['n']:,})")
        print(f"directory : {state['dir']}")
        if not state["files"]:
            print("  nothing cached. Fetch with `lazyqsar setup --reference`.")
            return
        for name, size in sorted(state["files"].items()):
            print(f"  {name:<32} {size / 1e6:8.1f} MB")

    elif args.action == "fetch":
        names = _reference_descriptors(args.only)
        files = [identity.smiles_filename(n)] + [
            identity.descriptor_filename(name, n) for name in names
        ]
        try:
            download(files, n=n, force=args.force)
        except ReferenceDownloadError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)

    elif args.action == "verify":
        problems = store.verify(n)
        if not problems:
            print("Every cached reference file is present and well formed.")
            return
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        sys.exit(1)

    elif args.action == "smiles":
        try:
            molecules = store.reference_smiles(n)
        except Exception as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        text = "smiles\n" + "\n".join(molecules) + "\n"
        if args.output:
            with open(args.output, "w") as handle:
                handle.write(text)
            print(f"Wrote {len(molecules):,} molecules to {args.output}")
        else:
            sys.stdout.write(text)


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
        "--reference",
        action="store_true",
        help=(
            "Download the reference library that `rank` is reported against. Not implied "
            "by --descriptors: a fast-mode model needs one 6.5 MB matrix and the whole "
            "bundle is 267 MB."
        ),
    )
    p_setup.add_argument(
        "--cpu-torch",
        action="store_true",
        help="Force-reinstall torch from PyTorch's CPU index, replacing any CUDA wheel pip may have installed via PyPI. Only meaningful with --descriptors.",
    )

    # --- reference ---
    p_ref = sub.add_parser(
        "reference",
        help="Inspect or fetch the reference library `rank` is reported against.",
    )
    p_ref.add_argument(
        "action",
        choices=["status", "fetch", "verify", "smiles"],
        help=(
            "status: what is cached. fetch: download it. verify: check what is cached. "
            "smiles: write the molecule list, which is all a bring-your-own-descriptor "
            "caller needs."
        ),
    )
    p_ref.add_argument(
        "--only",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated descriptors to fetch (default: all).",
    )
    p_ref.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Where `smiles` writes to (default: stdout).",
    )
    p_ref.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a file is already cached.",
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
    elif args.command == "reference":
        _cmd_reference(args)


if __name__ == "__main__":
    main()
