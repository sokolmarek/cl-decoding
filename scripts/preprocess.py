"""Resample, filter and segment WESAD or CLAS into 5, 3 and 1 s windows.

    python scripts/preprocess.py --dataset WESAD --root /path/to/WESAD
    python scripts/preprocess.py --dataset CLAS --root /path/to/CLAS --windows 5 1
"""

import argparse
import logging

import numpy as np

from cldecoding.datasets import LOADERS
from cldecoding.preprocessing import WINDOWS
from cldecoding.storage import save, segments_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS))
    parser.add_argument("--root", required=True, help="dataset folder as downloaded")
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS), choices=WINDOWS)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for window in args.windows:
        seg = LOADERS[args.dataset](args.root, window)
        path = segments_path(args.data_dir, args.dataset, window)
        save(path, **seg)
        counts = np.bincount(seg["label"], minlength=2)
        logging.info("%s %ds: %d segments (neutral %d, load %d) -> %s",
                     args.dataset, window, len(seg["label"]), counts[0], counts[1], path)


if __name__ == "__main__":
    main()
