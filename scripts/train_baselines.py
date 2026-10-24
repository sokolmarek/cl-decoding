"""Classical classifiers on the flattened fused features, with the same stratified group 10-fold split.

    python scripts/train_baselines.py --dataset WESAD --window 5
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from cldecoding.fusion import MODALITIES, fuse
from cldecoding.preprocessing import WINDOWS
from cldecoding.storage import features_path, load
from cldecoding.training import N_SPLITS, folds, scores

CLASSIFIERS = {
    "LOGIT": lambda: LogisticRegression(max_iter=1000, n_jobs=-1),
    "ET": lambda: ExtraTreesClassifier(n_jobs=-1, random_state=42),
    "RF": lambda: RandomForestClassifier(n_jobs=-1, random_state=42),
    "XGBoost": lambda: XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=42),
    "NB": lambda: GaussianNB(),
    "MLP": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=["WESAD", "CLAS"])
    parser.add_argument("--window", type=int, required=True, choices=WINDOWS)
    parser.add_argument("--models", nargs="+", default=list(CLASSIFIERS), choices=list(CLASSIFIERS))
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    feats = load(features_path(args.data_dir, args.dataset, args.window))
    x = fuse(feats, MODALITIES).reshape(len(feats["label"]), -1)
    y, groups = feats["label"], feats["subject"]

    rows = []
    for name in args.models:
        for i, (train_idx, test_idx) in enumerate(folds(y, groups, args.folds)):
            clf = CLASSIFIERS[name]().fit(x[train_idx], y[train_idx])
            rows.append({"model": name, "fold": i, **scores(y[test_idx], clf.predict(x[test_idx]))})
        acc = pd.DataFrame(rows).query("model == @name")["accuracy"]
        print(f"{name}: accuracy {acc.mean():.4f} +- {acc.std():.4f}")

    df = pd.DataFrame(rows)
    out = Path(args.out_dir) / f"baselines_{args.dataset}_{args.window}s.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.drop(columns="fold").groupby("model").agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
