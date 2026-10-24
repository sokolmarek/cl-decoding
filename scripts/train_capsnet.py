"""Stratified group 10-fold evaluation of the capsule network.

    python scripts/train_capsnet.py --dataset WESAD --window 5
    python scripts/train_capsnet.py --dataset CLAS --window 1 --modalities gadf gcn
"""

import argparse
from pathlib import Path

from cldecoding.fusion import MODALITIES, fuse
from cldecoding.preprocessing import WINDOWS
from cldecoding.storage import features_path, load
from cldecoding.training import N_SPLITS, cross_validate_capsnet, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["WESAD", "CLAS"])
    parser.add_argument("--window", type=int, required=True, choices=WINDOWS)
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES)
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    feats = load(features_path(args.data_dir, args.dataset, args.window))
    x = fuse(feats, args.modalities)
    y, groups = feats["label"], feats["subject"]
    print(f"{args.dataset} {args.window}s, {'+'.join(args.modalities)}: input {x.shape}")

    df = cross_validate_capsnet(x, y, groups, n_splits=args.folds, epochs=args.epochs,
                                batch_size=args.batch_size, lr=args.lr, verbose=args.verbose)
    out = Path(args.out_dir) / f"capsnet_{args.dataset}_{args.window}s_{'-'.join(args.modalities)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(summarize(df).round(4))
    print(f"fold scores saved to {out}")


if __name__ == "__main__":
    main()
