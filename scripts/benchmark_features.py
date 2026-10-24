"""Time needed to compute each feature type for one signal segment (Table 4).

    python scripts/benchmark_features.py --repeats 100
"""

import argparse
import timeit

import neurokit2 as nk

from cldecoding.features import fb_copula_granger, gadf, motif_difference_field
from cldecoding.preprocessing import FS, WINDOWS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    # Compile the numba kernels before timing
    warmup = nk.ecg_simulate(duration=1, sampling_rate=FS, random_state=0)
    gadf(warmup), motif_difference_field(warmup), fb_copula_granger(warmup)

    print(f"{'feature':<8}{'window':>8}{'mean (ms)':>12}")
    for window in WINDOWS:
        x = nk.ecg_simulate(duration=window, sampling_rate=FS, random_state=0)[: window * FS]
        for name, fn in (("GADF", lambda: gadf(x)), ("MDF", lambda: motif_difference_field(x)),
                         ("GCN", lambda: fb_copula_granger(x))):
            t = timeit.timeit(fn, number=args.repeats) / args.repeats
            print(f"{name:<8}{window:>7}s{t * 1e3:>12.2f}")


if __name__ == "__main__":
    main()
