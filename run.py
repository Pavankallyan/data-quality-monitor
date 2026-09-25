"""End-to-end pipeline: generate -> profile -> drift -> report + plots.

Usage:
    python run.py
Run from the project root (where run.py lives).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data import main as generate  # noqa: E402
from report import build_report  # noqa: E402


def main() -> None:
    print("== 1/3 generate synthetic batches ==")
    generate()
    print("== 2/3 build quality report (profiles + drift + plots) ==")
    build_report()
    print("== done ==")


if __name__ == "__main__":
    main()
