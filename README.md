# Cognitive load decoding from causal spatiotemporal patterns

Code for the paper **Efficient cognitive load decoding using causal spatiotemporal patterns from multimodal
physiological signals** (Sokol, Hejda, Volf, Leová and Kutílek, *Computers in Biology and Medicine* 197, 111063,
2025). [doi:10.1016/j.compbiomed.2025.111063](https://doi.org/10.1016/j.compbiomed.2025.111063)

Short ECG and EDA segments (1, 3 or 5 s) are turned into image-like features and classified by a lightweight
capsule network into neutral and cognitive load states:

- **GADF**, Gramian Angular Difference Fields, encode the global temporal dynamics of a segment,
- **MDF**, Motif Difference Fields, encode local repeating triadic motifs over all delays,
- **GCN**, forward–backward copula Lasso-Granger causality networks, encode nonlinear causal dependencies
  between sub-series of a segment.

The six feature maps (three per signal) are stacked into a 64 × 64 × 6 tensor and classified by a modified
Efficient-CapsNet with self-attention routing that has 323,104 parameters.

```
ECG, EDA ─► resample 256 Hz ─► filter ─► 5 / 3 / 1 s windows ─┬─► GADF  64×64×2 ─────────────────┐
                                                              ├─► MDF   32×64×2 ─► z-score, pad ─┼─► 64×64×6 ─► Efficient-CapsNet ─► neutral / load
                                                              └─► GCN   32×32×2 ─► resize ───────┘
```

## Repository

```
src/cldecoding/
    preprocessing.py   resampling, Butterworth filtering, fixed and R-peak centred windows
    datasets.py        WESAD and CLAS readers and label definitions
    features/          gadf.py, mdf.py, gcn.py
    fusion.py          normalisation, padding and stacking of the feature maps
    model/             Efficient-CapsNet layers and network
    training.py        stratified group K-fold training and evaluation
scripts/               command line entry points for every step
tests/                 unit tests
```

## Installation

Python 3.9 to 3.11 and TensorFlow 2.10 to 2.15.

```bash
git clone https://github.com/sokolmarek/cl-decoding.git
cd cl-decoding
pip install -e .
```

## Data

Both datasets are public and are not redistributed here.

- **WESAD**, [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).
  Chest (RespiBAN) ECG and EDA at 700 Hz. Neutral class: baseline. Cognitive load class: mental arithmetic part
  of the Trier Social Stress Test (stress phase after its first 5.5 minutes).
- **CLAS**, [IEEE DataPort](https://ieee-dataport.org/open-access/database-cognitive-load-affect-and-stress-recognition).
  Shimmer3 ECG and EDA at 256 Hz. Neutral class: Baseline block. Cognitive load class: Stroop Test block.

The readers expect the folders as downloaded: `WESAD/S*/S*.pkl` and `CLAS/{Block_details,Participants}`.

## Usage

```bash
# 1. Resample to 256 Hz, filter (ECG high-pass 0.5 Hz, EDA low-pass 3 Hz) and segment
python scripts/preprocess.py --dataset WESAD --root /path/to/WESAD
python scripts/preprocess.py --dataset CLAS  --root /path/to/CLAS

# 2. GADF, MDF and GCN features (the causality networks dominate the run time, use --jobs)
python scripts/extract_features.py --dataset WESAD --jobs 16
python scripts/extract_features.py --dataset CLAS  --jobs 16

# 3. Capsule network, stratified group 10-fold (subject-independent)
python scripts/train_capsnet.py --dataset WESAD --window 5
python scripts/train_capsnet.py --dataset CLAS  --window 1 --modalities gadf gcn   # feature ablation

# 4. Classical baselines on the same features and folds (LOGIT, ET, RF, XGBoost, NB, MLP)
python scripts/train_baselines.py --dataset WESAD --window 5

# 5. Feature generation time per segment
python scripts/benchmark_features.py
```

Intermediate files go to `data/segments` and `data/features`, fold scores to `results/`.

The features can also be used directly:

```python
from cldecoding.features import gadf, motif_difference_field, fb_copula_granger

img = gadf(segment)                       # (1, 64, 64)
field = motif_difference_field(segment)   # (32, 64)
graph = fb_copula_granger(segment)        # (32, 32), entry [j, i] = influence of sub-series j on i
```

## Settings

| Component | Setting |
| --- | --- |
| GADF | PAA to 64 points, values rescaled to [-1, 1] |
| MDF | motif length 3, all delays, bicubic resize to 32 × 64, z-score per sample and channel |
| GCN | 32 sub-series per segment, lag 4, λ ∈ {10⁻³, …, 10²} (6 values) chosen per target by AIC |
| Network | Adam, learning rate 10⁻⁴ with exponential annealing, 20 epochs, batch 32, margin loss + 0.392 × reconstruction |
| Evaluation | stratified group 10-fold, subjects never shared between training and test folds |

## Results

Accuracy (mean ± SD over 10 folds) of the full model, from Table 1 of the paper.

| Dataset | 5 s | 3 s | 1 s |
| --- | --- | --- | --- |
| WESAD | 0.92 ± 0.07 | 0.90 ± 0.06 | 0.84 ± 0.09 |
| CLAS | 0.86 ± 0.01 | 0.81 ± 0.02 | 0.80 ± 0.03 |

## Citation

```bibtex
@article{sokol2025efficient,
  author  = {Sokol, Marek and Hejda, Jan and Volf, Petr and Leov{\'a}, L{\'y}die and Kut{\'i}lek, Patrik},
  title   = {Efficient cognitive load decoding using causal spatiotemporal patterns from multimodal physiological signals},
  journal = {Computers in Biology and Medicine},
  volume  = {197},
  pages   = {111063},
  year    = {2025},
  doi     = {10.1016/j.compbiomed.2025.111063}
}
```

## Acknowledgements

The capsule network is adapted from [Efficient-CapsNet](https://github.com/EscVM/Efficient-CapsNet) by Mazzia and
Salvetti (Apache 2.0). This work was supported by the Ministry of the Interior of the Czech Republic under project
OPSEC VK01020078.

## License

MIT, see [LICENSE](LICENSE). Files in `src/cldecoding/model/` are derived from Efficient-CapsNet and keep their
Apache 2.0 notice.
