#!/usr/bin/env python
"""Prove the *published* bundle is intact and usable, from a clean cache.

    python scripts/verify_reference_bundle.py              # everything, ~272 MB
    python scripts/verify_reference_bundle.py --only morgan

The acceptance gate, and the only step that tests what S3 actually serves. Every other
check in this repository runs against the bundle directory on the machine that built it,
where a file can be correct locally and absent, truncated or stale in the published prefix.

So this deliberately downloads into a throwaway cache rather than the user's: it must not be
able to pass by reading something the build left behind. It never touches ``data/``.

Five things, in the order that a failure is cheapest to diagnose:

1. the manifest is published and parses
2. every file matches its published sha256
3. the matrices agree structurally -- tier, dimensions, and the molecule list they were
   built from, which is what makes row *i* the same molecule in all of them
4. the installed descriptor code reproduces the published canary values
5. a model fits against it and ranks the way it should

Run it on a second machine before anyone depends on the bundle. Passing here only says the
prefix is good *from this environment*; a different RDKit or a different checkpoint is
exactly what step 4 exists to catch, and it cannot catch what it never runs on.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

OK, BAD = "  ok   ", "  FAIL "


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated descriptors to verify (default: every published one).",
    )
    ap.add_argument(
        "--keep",
        action="store_true",
        help="Leave the downloaded cache in place for inspection.",
    )
    args = ap.parse_args()

    # A clean cache, before importing anything that reads the environment.
    cache = Path(tempfile.mkdtemp(prefix="lazyqsar-verify-"))
    os.environ["LAZYQSAR_HOME"] = str(cache)
    os.environ.pop("LAZYQSAR_REFERENCE_DIR", None)
    os.environ.pop("LAZYQSAR_REFERENCE_OFFLINE", None)

    import numpy as np

    from lazyqsar.reference import identity, manifest as mf
    from lazyqsar.reference.download import ReferenceDownloadError, download
    from lazyqsar.registry import DESCRIPTOR_TYPES

    n = identity.default_n()
    failures: list[str] = []

    def check(passed: bool, label: str, detail: str = "") -> None:
        print(f"{OK if passed else BAD} {label}{('  ' + detail) if detail else ''}")
        if not passed:
            failures.append(label)

    print(f"verifying {identity.REFERENCE_ID} (tier {n:,}) into {cache}\n")

    # --- 1. the manifest -------------------------------------------------------
    try:
        book = mf.load()
    except Exception as exc:  # pragma: no cover - network shapes vary
        print(f"{BAD} manifest could not be fetched: {exc}")
        return 1
    if not book:
        print(f"{BAD} no manifest published: nothing else can be verified")
        return 1
    check(
        book.get("reference_id") == identity.REFERENCE_ID,
        "manifest identifies the bundle",
        book.get("reference_id", "?"),
    )
    check(book.get("n") == n, "manifest tier matches this install", str(book.get("n")))

    names = sorted(
        [x.strip() for x in args.only.split(",")] if args.only else book["descriptors"]
    )
    unknown = sorted(set(names) - set(DESCRIPTOR_TYPES))
    if unknown:
        print(f"{BAD} unknown descriptor(s): {', '.join(unknown)}")
        return 1

    # --- 2. download and hash --------------------------------------------------
    files = [identity.smiles_filename(n)] + [
        identity.descriptor_filename(name, n) for name in names
    ]
    try:
        download(files, n=n)
    except ReferenceDownloadError as exc:
        print(f"{BAD} download failed: {exc}")
        return 1

    root = identity.reference_dir()
    for name in sorted(book.get("files", {})):
        path = root / name
        if not path.is_file():
            continue  # only what was fetched is checkable
        expected = book["files"][name]["sha256"]
        check(_sha256(path) == expected, f"sha256 {name}")

    # --- 3. structure and row alignment ----------------------------------------
    import h5py

    smiles_path = root / identity.smiles_filename(n)
    molecules = [
        s.strip() for s in smiles_path.read_text().splitlines()[1:] if s.strip()
    ]
    check(len(molecules) == n, "molecule list length", f"{len(molecules):,}")

    shared = book["smiles"]["sha256"]
    for name in names:
        path = root / identity.descriptor_filename(name, n)
        with h5py.File(path, "r") as handle:
            dset = handle["X"]
            attrs = dict(dset.attrs)
            entry = book["descriptors"][name]
            check(dset.shape[0] == n, f"{name}: row count", str(dset.shape[0]))
            check(
                dset.shape[1] == entry["shape"][1],
                f"{name}: feature count",
                str(dset.shape[1]),
            )
            # The one that matters most: every matrix must have been built from the same
            # molecule list, or row i is a different molecule in different matrices and the
            # pooled probability is meaningless.
            check(
                str(attrs.get("smiles_sha256", "")) == shared,
                f"{name}: built from the published molecule list",
            )
            block = np.asarray(dset[: min(2048, n)], dtype=np.float64)
            check(bool(np.isfinite(block).all()), f"{name}: finite values")

        live = (
            DESCRIPTOR_TYPES
            and __import__(
                "lazyqsar.registry", fromlist=["get_descriptor_type"]
            ).get_descriptor_type(name)()
        )
        check(
            int(getattr(live, "n_dim", 0) or len(getattr(live, "features", [])))
            == entry["shape"][1],
            f"{name}: matches the installed descriptor's dimension",
        )

    # --- 4. would this install compute the same values? ------------------------
    try:
        drift = mf.check_environment(names, book)
    except mf.ReferenceDriftError as exc:
        print(f"{BAD} drift: {exc}")
        failures.append("drift check")
        drift = {}
    for name, result in sorted(drift.items()):
        status = result.get("status")
        detail = result.get("reason") or (
            f"cosine {result['cosine']:.7f}" if "cosine" in result else ""
        )
        check(
            status in {"pass", "skip"},
            f"{name}: reproduces published values",
            f"{status} {detail}",
        )
        if status == "warn":
            print("         (within backend tolerance, recorded rather than fatal)")

    # --- 5. does a model actually fit against it? ------------------------------
    if "morgan" in names:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from _helpers.smiles import load_reference_dataset

        from lazyqsar.qsar import LazyClassifierQSAR

        smiles, y = load_reference_dataset()
        y = np.asarray(y)
        model = LazyClassifierQSAR(mode="fast")
        model.fit(smiles_list=smiles, y=y)
        ranks = model.predict_rank(smiles_list=smiles)[:, 1]
        proba = model.predict_proba(smiles_list=smiles)[:, 1]
        check(len(model.pooled_rank_knots_) > 0, "a model fits against the bundle")
        check(
            np.array_equal(
                np.argsort(np.argsort(ranks)), np.argsort(np.argsort(proba))
            ),
            "rank orders molecules exactly as proba",
        )
        check(
            float(ranks[y == 1].mean()) > float(ranks[y == 0].mean()),
            "known actives rank above known inactives",
            f"{ranks[y == 1].mean():.3f} vs {ranks[y == 0].mean():.3f}",
        )
    else:
        print("  skip   fit check (needs morgan)")

    if args.keep:
        print(f"\ncache kept at {cache}")
    else:
        shutil.rmtree(cache, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) — {', '.join(failures[:5])}")
        return 1
    print("The published bundle is intact and this install reproduces it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
