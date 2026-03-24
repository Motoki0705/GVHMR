#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any
import torch

import numpy as np


def load_pkl(path: Path) -> None:
    print(f"[PKL] {path}")
    data = torch.load(path, map_location="cpu")
    return data

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect SMPL / SMPL-X body model files (.pkl / .npz)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Target file or directory. Example: checkpoints/body_models",
    )
    args = parser.parse_args()
    data = load_pkl(path=args.path)

    print(data.keys())

if __name__ == "__main__":
    main()