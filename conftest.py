"""Make the package importable when running `pytest` from the repo root."""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
