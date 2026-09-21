"""Run the CLI in a subprocess with the descriptor stubs installed in the child.

The parity tests have to exercise the real console-script path -- argparse, ``_cmd_predict``,
the CSV writer -- not just the function it calls. But a subprocess does not inherit the
parent's monkeypatches, and the `fit` tier has no RDKit, so the child has to install the
stubs itself before importing anything that would reach a descriptor.

The launcher below does exactly that, using pytest's own ``MonkeyPatch`` so the stub
installer is shared with the in-process tests rather than reimplemented.
"""

import os
import subprocess
import sys

_LAUNCHER = """\
import sys

import pytest

from _helpers.stubs import install_stub_registry

_mp = pytest.MonkeyPatch()
register = install_stub_registry(_mp)
register(*{descriptors!r})

from lazyqsar.cli.main import main

sys.argv = ["lazyqsar"] + {argv!r}
main()
"""


def run_cli(argv, tmp_path, descriptors=("morgan",), check=True):
    """Invoke ``lazyqsar <argv>`` in a child process with stubbed descriptors.

    Returns the ``CompletedProcess``. ``check=False`` is for the tests that assert on exit
    codes and stderr.
    """
    tests_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(str(tmp_path), "_run_cli.py")
    with open(script, "w") as f:
        f.write(_LAUNCHER.format(argv=list(argv), descriptors=list(descriptors)))

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [tests_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, script], capture_output=True, text=True, env=env
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"lazyqsar {' '.join(argv)} exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc
