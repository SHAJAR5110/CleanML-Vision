"""Generate a reproducible .ipynb from the cleaning history."""

from __future__ import annotations

import json


def _cell(cell_type: str, source: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
    }


def build_notebook(source_label: str, history: list[dict]) -> bytes:
    """Return UTF-8 bytes of an ipynb that replays the history."""
    cells = []

    cells.append(_cell("markdown",
        f"# CleanML — reproducible cleaning notebook\n\n"
        f"Source dataset: `{source_label}`\n\n"
        f"This notebook reproduces every transformation applied in CleanML, "
        f"so you can re-run the same pipeline on new data."
    ))

    cells.append(_cell("code",
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "# Load your CSV (replace the path with your file):\n"
        f"df = pd.read_csv({source_label!r})\n"
        "print(df.shape)\n"
        "df.head()"
    ))

    for i, entry in enumerate(history, 1):
        op = entry["op"]
        cells.append(_cell("markdown",
            f"## Step {i}: {op.get('family')} · {op.get('strategy')}"
            + (f" — `{op.get('column')}`" if op.get("column") else "")
            + f"\n\n{entry.get('message','')}"
        ))
        code = entry.get("code") or "# (no-op)"
        cells.append(_cell("code", code))

    cells.append(_cell("markdown", "## Final result"))
    cells.append(_cell("code",
        "print('shape:', df.shape)\n"
        "print('missing per col:'); print(df.isna().sum())\n"
        "df.head()"
    ))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1).encode("utf-8")
