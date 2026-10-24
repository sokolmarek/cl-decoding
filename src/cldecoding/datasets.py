"""Readers for the WESAD and CLAS datasets.

Both readers return segmented, preprocessed ECG and EDA together with binary labels
(0 = neutral, 1 = cognitive load) and subject identifiers.

WESAD (Schmidt et al., 2018), RespiBAN chest device at 700 Hz:
    neutral         baseline phase (label 1)
    cognitive load  mental arithmetic part of the Trier Social Stress Test, i.e. the stress phase (label 2)
                    after its first 5.5 minutes of speech preparation and public speaking

CLAS (Markova et al., 2019), Shimmer3 at 256 Hz:
    neutral         "Baseline" block
    cognitive load  "Stroop Test" block
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import preprocess_recording

log = logging.getLogger(__name__)

WESAD_FS = 700
WESAD_SPEECH_DURATION = 5.5 * 60

CLAS_FS = 256
CLAS_LABELS = {"Baseline": 0, "Stroop Test": 1}


def _empty():
    return {"ecg": [], "eda": [], "label": [], "subject": []}


def _append(out, subject, ecg, eda, labels):
    out["ecg"].append(ecg)
    out["eda"].append(eda)
    out["label"].append(labels)
    out["subject"].append(np.full(len(labels), subject))


def _stack(out):
    if not out["label"]:
        raise RuntimeError("No segments were extracted, check the dataset path.")
    return {key: np.concatenate(val) for key, val in out.items()}


def wesad_labels(raw):
    """Map WESAD protocol labels to the binary task (-1 marks samples that are not used)."""
    raw = np.asarray(raw).ravel()
    labels = np.full(raw.shape, -1, dtype=int)
    labels[raw == 1] = 0

    stress = np.flatnonzero(raw == 2)
    if len(stress):
        arithmetic_start = stress[0] + int(WESAD_SPEECH_DURATION * WESAD_FS)
        labels[stress[stress >= arithmetic_start]] = 1
    return labels


def load_wesad(root, window):
    """Read ``root/S*/S*.pkl`` and return segments of ``window`` seconds."""
    out = _empty()
    files = sorted(Path(root).glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    for file in files:
        with open(file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        chest = data["signal"]["chest"]
        ecg, eda = chest["ECG"].ravel(), chest["EDA"].ravel()
        labels = wesad_labels(data["label"])

        ecg_seg, eda_seg, seg_labels = preprocess_recording(ecg, eda, labels, WESAD_FS, window)
        log.info("WESAD %s: %d segments", file.stem, len(seg_labels))
        _append(out, file.stem, ecg_seg, eda_seg, seg_labels)
    return _stack(out)


def _clas_block_file(folder, block, kind):
    matches = sorted(folder.glob(f"{block}_{kind}_*.csv"))
    return matches[0] if matches else None


def load_clas(root, window):
    """Read the CLAS ``Block_details`` and ``Participants/*/by_block`` folders.

    Every block is preprocessed on its own because blocks are stored as separate recordings.
    """
    root = Path(root)
    out = _empty()
    details = sorted((root / "Block_details").glob("*.csv"), key=lambda p: int(p.stem.split("_")[0][4:]))
    for file in details:
        part = file.stem.split("_")[0]
        folder = root / "Participants" / part / "by_block"
        blocks = pd.read_csv(file)

        n_before = sum(len(x) for x in out["label"])
        for _, row in blocks.iterrows():
            label = CLAS_LABELS.get(row["Block Type"])
            if label is None:
                continue
            ecg_file = _clas_block_file(folder, row["Block"], "ecg")
            gsr_file = _clas_block_file(folder, row["Block"], "gsr_ppg")
            if ecg_file is None or gsr_file is None:
                log.warning("CLAS %s: missing files for block %s, skipped", part, row["Block"])
                continue

            ecg = pd.read_csv(ecg_file)["ecg2"].to_numpy(dtype=float)
            eda = pd.read_csv(gsr_file)["gsr"].to_numpy(dtype=float)
            n = min(len(ecg), len(eda))
            if n < 2 * window * CLAS_FS:
                continue
            labels = np.full(n, label)

            ecg_seg, eda_seg, seg_labels = preprocess_recording(ecg[:n], eda[:n], labels, CLAS_FS, window)
            _append(out, part, ecg_seg, eda_seg, seg_labels)

        log.info("CLAS %s: %d segments", part, sum(len(x) for x in out["label"]) - n_before)
    return _stack(out)


LOADERS = {"WESAD": load_wesad, "CLAS": load_clas}
