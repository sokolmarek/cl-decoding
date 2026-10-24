"""Compute GADF, MDF and GCN images of preprocessed segments.

    python scripts/extract_features.py --dataset WESAD --windows 5 3 1 --jobs 8
"""

import argparse
import logging

from cldecoding.features import extract
from cldecoding.preprocessing import WINDOWS
from cldecoding.storage import features_path, load, save, segments_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["WESAD", "CLAS"])
    parser.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS), choices=WINDOWS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel workers for the causality networks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for window in args.windows:
        seg = load(segments_path(args.data_dir, args.dataset, window))
        logging.info("%s %ds: %d segments", args.dataset, window, len(seg["label"]))
        feats = extract(seg["ecg"], seg["eda"], n_jobs=args.jobs)
        path = features_path(args.data_dir, args.dataset, window)
        save(path, **feats, label=seg["label"], subject=seg["subject"])
        logging.info("saved %s", path)


if __name__ == "__main__":
    main()
