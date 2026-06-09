# ASTM E1049 rainflow cycle counting for bridge member stress histories.
#
# Thin wrapper around the `rainflow` package (pip install rainflow).
# Falls back to a simple peak-valley extraction if the package is unavailable.
#
# No Omniverse imports -- pure Python, unit-testable from the CLI.

from __future__ import annotations

from typing import List, Tuple

import numpy as np

try:
    import rainflow as _rf
    _RF_AVAILABLE = True
except ImportError:
    _rf = None              # type: ignore[assignment]
    _RF_AVAILABLE = False
    print("[rainflow_counter] 'rainflow' package not found; "
          "falling back to simple peak-valley cycle extraction.")


# Public type alias
Cycle = Tuple[float, float, float]   # (amplitude_pa, mean_pa, n_cycles)


def count_cycles(stress_history_pa: np.ndarray) -> List[Cycle]:
    h = np.asarray(stress_history_pa, dtype=float)
    if h.size < 3:
        # Too short for meaningful counting -- return one half-cycle at peak
        peak = float(np.max(np.abs(h))) if h.size > 0 else 0.0
        return [(peak, 0.0, 0.5)]

    if _RF_AVAILABLE:
        return _rainflow_astm(h)
    else:
        return _peak_valley_fallback(h)


def count_cycles_per_member(
    stress_histories: dict,   # member_index -> np.ndarray (Pa)
) -> dict:                    # member_index -> List[Cycle]
    return {m_idx: count_cycles(hist)
            for m_idx, hist in stress_histories.items()}



def _rainflow_astm(h: np.ndarray) -> List[Cycle]:
    cycles: List[Cycle] = []
    try:
        for rng, mean, count, *_ in _rf.extract_cycles(h):
            amplitude = float(rng) / 2.0
            cycles.append((amplitude, float(mean), float(count)))
    except Exception as exc:
        print(f"[rainflow_counter] rainflow.extract_cycles failed ({exc}); "
              "using peak-valley fallback.")
        return _peak_valley_fallback(h)
    return cycles if cycles else [(float(np.max(np.abs(h))), 0.0, 0.5)]


def _peak_valley_fallback(h: np.ndarray) -> List[Cycle]:
    # Extract turning points (local maxima/minima)
    turning: List[float] = [h[0]]
    for i in range(1, len(h) - 1):
        prev, curr, nxt = h[i - 1], h[i], h[i + 1]
        if (curr > prev and curr > nxt) or (curr < prev and curr < nxt):
            turning.append(curr)
    turning.append(h[-1])

    if len(turning) < 2:
        return [(float(abs(h[0])), float(h[0]) / 2.0, 0.5)]

    cycles: List[Cycle] = []
    for i in range(0, len(turning) - 1, 2):
        a, b = turning[i], turning[i + 1]
        amplitude = abs(b - a) / 2.0
        mean      = (a + b) / 2.0
        cycles.append((amplitude, mean, 0.5))      # half-cycles

    if not cycles:
        peak = float(np.max(np.abs(h)))
        cycles = [(peak, 0.0, 0.5)]
    return cycles


# Self-test
def run_self_test(verbose: bool = True) -> None:
    A  = 100e6    # 100 MPa amplitude
    f  = 1.0      # 1 Hz
    N  = 5        # 5 complete periods
    dt = 1e-3     # 1 ms sampling

    t = np.arange(0, N / f, dt)
    h = A * np.sin(2 * np.pi * f * t)

    cycles = count_cycles(h)

    total_counts = sum(c for _, _, c in cycles)
    total_damage_proxy = sum(c * amp for amp, _, c in cycles)

    if verbose:
        print(f"Sine wave: A={A/1e6:.0f} MPa, {N} periods, {len(t)} samples")
        print(f"  Cycle count groups : {len(cycles)}")
        print(f"  Total cycles counted: {total_counts:.1f}  (expect ~{N})")
        print(f"  Mean amplitude: {sum(amp*c for amp,_,c in cycles)/max(total_counts,1)/1e6:.1f} MPa"
              f"  (expect ~{A/1e6:.0f} MPa)")

    tol = 0.15   # 15% tolerance -- edge half-cycles cause slight deviation
    err_count = abs(total_counts - N) / N
    if verbose:
        print(f"  Count error: {err_count*100:.1f}%  (threshold {tol*100:.0f}%)")

    if err_count >= tol:
        raise AssertionError("SELF-TEST FAILED: cycle count deviates too much")

    if verbose:
        print("SELF-TEST PASSED")
    if False:
        pass
        print("SELF-TEST FAILED -- cycle count deviates too much")
        sys.exit(1)

    # Quick test of per-member wrapper
    histories = {0: h, 1: h * 0.5, 2: np.zeros(10)}
    per_member = count_cycles_per_member(histories)
    assert 0 in per_member and 1 in per_member and 2 in per_member
    if verbose:
        print(f"  per-member wrapper: OK ({len(per_member)} members)")
        print("ALL TESTS PASSED")


if __name__ == "__main__":
    run_self_test()


