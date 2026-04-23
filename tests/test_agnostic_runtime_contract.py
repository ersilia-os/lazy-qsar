import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

from lazyqsar.agnostic import LazyClassifier


def _synthetic_binary_data(n_samples: int = 80, n_features: int = 24, seed: int = 123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype("float32")
    margin = X[:, 0] + 0.25 * X[:, 1] + rng.normal(scale=0.15, size=n_samples)
    y = (margin > 0).astype(int)
    return X, y


def _assert_onnx_artifacts_exist(model_dir: Path):
    batch_dir = model_dir / "batch_0"
    expected = [
        model_dir / "metadata.json",
        batch_dir / "preprocessor.onnx",
        batch_dir / "preprocessor.json",
        batch_dir / "linear.onnx",
        batch_dir / "linear.json",
        batch_dir / "xgboost.onnx",
        batch_dir / "xgboost.json",
        batch_dir / "pooler.json",
    ]
    missing = [str(p) for p in expected if not p.is_file()]
    assert not missing, f"Missing expected artifacts: {missing}"


def test_train_from_xy_and_save_contains_onnx_artifacts(tmp_path):
    X, y = _synthetic_binary_data()
    model = LazyClassifier()
    model.fit(X=X, y=y)

    model_dir = tmp_path / "agnostic_model"
    save_path = model.save(str(model_dir))

    assert save_path == str(model_dir)
    _assert_onnx_artifacts_exist(model_dir)

    loaded = LazyClassifier.load(str(model_dir))
    proba = loaded.predict_proba(X[:12])

    assert proba.shape == (12, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_load_and_infer_with_fit_deps_blocked_in_subprocess(tmp_path):
    X, y = _synthetic_binary_data(seed=456)
    model = LazyClassifier()
    model.fit(X=X, y=y)

    model_dir = tmp_path / "agnostic_model_blocked"
    model.save(str(model_dir))
    _assert_onnx_artifacts_exist(model_dir)

    script = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys
        import numpy as np

        BLOCKED = ("sklearn", "xgboost", "scipy")

        class BlockImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                for name in BLOCKED:
                    if fullname == name or fullname.startswith(name + "."):
                        raise ModuleNotFoundError(f"Blocked import: {fullname}")
                return None

        sys.meta_path.insert(0, BlockImports())

        from lazyqsar.agnostic import LazyClassifier

        model = LazyClassifier.load(sys.argv[1])
        n_features = int(sys.argv[2])
        X = np.random.default_rng(7).normal(size=(10, n_features)).astype("float32")
        proba = model.predict_proba(X)
        print(json.dumps({
            "shape": list(proba.shape),
            "sum_ok": bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)),
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(model_dir), str(X.shape[1])],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"Subprocess failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["shape"] == [10, 2]
    assert payload["sum_ok"] is True
