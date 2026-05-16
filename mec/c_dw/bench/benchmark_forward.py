"""Benchmark the three forward (equilibrium) solvers on random TU markets:

    1. pure Gurobi  : TUMatching.gurobi_solve   (full LP in Python+Gurobi)
    2. Python RROA  : TUMatching.rroa_solve     (column generation in Python)
    3. C RROA       : TUMatching.rroa_solve_C   (column generation in C)

Sweeps over market size (I=J) at fixed X=Y, runs each method, records the
total wall time (build + solve), confirms all three return the same
objective, and saves a log-scale plot to forward_bench.png.

Usage:
    cd mec/c_dw && make lib    # build libdw.dylib first
    python benchmark_forward.py
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np

# Allow running directly from this file: add the parent of `mec/` to sys.path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from mec.dw import TUMatching  # noqa: E402


def make_problem(I, J, X, Y, seed):
    rng = np.random.default_rng(seed)
    Phi    = rng.normal(size=(X, Y))
    eps_iy = rng.normal(size=(I, Y))
    eta_xj = rng.normal(size=(X, J))
    eps_i0 = rng.normal(size=I)
    eta_0j = rng.normal(size=J)
    x_i = rng.integers(0, X, size=I)
    y_j = rng.integers(0, Y, size=J)
    return Phi, eps_iy, eta_xj, eps_i0, eta_0j, np.eye(X)[x_i], np.eye(Y)[y_j]


def time_method(method_name, T):
    t0 = time.perf_counter()
    if method_name == "gurobi":
        _, _, _, _, m = T.gurobi_solve()
        obj = m.ObjVal
    elif method_name == "py_rroa":
        history, *_ = T.rroa_solve()
        obj = history[-1]
    elif method_name == "c_rroa":
        history, *_ = T.rroa_solve_C()
        obj = history[-1]
    else:
        raise ValueError(method_name)
    return time.perf_counter() - t0, obj


def main():
    X, Y = 10, 10
    sizes = [100, 200, 500, 1000, 2000, 3000, 5000]
    methods = ["gurobi", "py_rroa", "c_rroa"]
    timings = {m: [] for m in methods}
    objs    = {m: [] for m in methods}

    # Warm-up so dlopen / Gurobi-env init don't pollute the first timed point.
    P = make_problem(50, 50, X, Y, seed=0)
    for m in methods:
        time_method(m, TUMatching(*P))

    print(f"{'size (I=J)':>10}  " + "  ".join(f"{m:>10}" for m in methods) + "  obj match")
    for n in sizes:
        Phi, eiy, exj, ei0, e0j, dix, djy = make_problem(n, n, X, Y, seed=42)

        # Each call needs its own TUMatching to isolate state.
        method_objs = []
        method_times = []
        for m in methods:
            T = TUMatching(Phi, eiy, exj, ei0, e0j, dix, djy)
            t, o = time_method(m, T)
            method_objs.append(o); method_times.append(t)
            timings[m].append(t); objs[m].append(o)

        ok = max(abs(o - method_objs[0]) for o in method_objs) < 1e-5
        print(f"{n:>10}  " + "  ".join(f"{t:>10.3f}" for t in method_times)
              + ("  YES" if ok else "  NO"))

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    pretty = {"gurobi": "Pure Gurobi (Python)",
              "py_rroa": "RROA / DW (Python)",
              "c_rroa": "RROA / DW (C, OpenMP)"}
    style = {"gurobi": "o-", "py_rroa": "s-", "c_rroa": "^-"}
    for m in methods:
        ax.plot(sizes, timings[m], style[m], label=pretty[m], linewidth=2, markersize=7)
    ax.set_xlabel("market size  I = J")
    ax.set_ylabel("total wall time (s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Forward TU matching, X=Y={X}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "forward_bench.png")
    fig.savefig(out, dpi=140)
    print(f"\nplot saved to: {out}")


if __name__ == "__main__":
    main()
