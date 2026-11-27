"""SPARQL endpoint on the store of an experiment (Oxigraph server)."""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Union

from up_lab.config import UPLabError


def serve_command(
    exp_dir: Union[str, Path], host: str = "localhost", port: int = 7878
) -> List[str]:
    """
    Builds the command line of a read-only Oxigraph server on the store.

    :param exp_dir: The experiment directory.
    :param host: The host to bind.
    :param port: The port to bind.
    :return: The command; queries see the union of all named graphs.
    """
    store = Path(exp_dir) / "store"
    if not store.exists():
        raise UPLabError(f"No store in {exp_dir}: run 'uplab collect' first.")
    bin_dir = str(Path(sys.executable).parent)
    binary = shutil.which("oxigraph") or shutil.which("oxigraph", path=bin_dir)
    if binary is None:
        raise UPLabError(
            "The 'oxigraph' server binary was not found; "
            "install it with: pip install 'up-lab[serve]'"
        )
    command = [binary, "serve-read-only", "--location", str(store)]
    return command + ["--bind", f"{host}:{port}", "--union-default-graph"]


def serve(exp_dir: Union[str, Path], host: str = "localhost", port: int = 7878) -> None:
    """
    Replaces the current process with a SPARQL endpoint at
    `http://<host>:<port>/query`.

    :param exp_dir: The experiment directory.
    :param host: The host to bind.
    :param port: The port to bind.
    """
    command = serve_command(exp_dir, host, port)
    print(f"SPARQL endpoint: http://{host}:{port}/query", flush=True)
    os.execv(command[0], command)
