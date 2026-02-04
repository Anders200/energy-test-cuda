from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


# ============================================================
# Data model
# ============================================================

@dataclass
class Row:
    n: int
    dim: int
    method: str  # "cpu", "cross", "cuda"
    stat: float
    p: float
    stat_t: float
    p_t: float


# ============================================================
# Pretty printing
# ============================================================

def print_table(rows: List[Row]) -> None:
    if not rows:
        print("(no results)")
        return

    dims = sorted({r.dim for r in rows})
    for dim in dims:
        print(f"\n=== Dimension {dim} ===")
        rdim = [r for r in rows if r.dim == dim]
        for n in sorted({r.n for r in rdim}):
            for r in [x for x in rdim if x.n == n]:
                if r.method == "cpu":
                    label = "CPU"
                elif r.method == "cross":
                    label = "CROSS"
                elif r.method == "cuda":
                    label = "CUDA"
                else:
                    label = "UNK"

                print(
                    f"{n:7d} | dim={dim:<3d} | {label:5s} "
                    f"stat={r.stat:12.6g} | p={r.p:8.4g} | "
                    f"stat_t={r.stat_t:8.4f}s | p_t={r.p_t:8.4f}s"
                )


# ============================================================
# Utilities
# ============================================================

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_energy_py_on_path() -> None:
    build_lib = repo_root() / "build" / "lib"
    if build_lib.exists() and str(build_lib) not in sys.path:
        sys.path.insert(0, str(build_lib))


# ============================================================
# Subprocess mode (legacy, optional)
# ============================================================

LINE_RE = re.compile(
    r"^\s*(?P<n>\d+)\s*\|\s*dim=(?P<dim>\d+)\s*\|\s*"
    r"(?P<label>CPU|CROSS|CUDA)\s+stat=\s*"
    r"(?P<stat>[-+0-9.eE]+)\s*\|\s*p=(?P<p>[-+0-9.eE]+)\s*\|\s*"
    r"stat_t=(?P<stat_t>[-+0-9.eE]+)s\s*\|\s*p_t=(?P<p_t>[-+0-9.eE]+)s\s*$"
)


def run_app(exe: Path) -> str:
    proc = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
    return proc.stdout


def parse_output(text: str) -> List[Row]:
    rows: List[Row] = []
    for line in text.splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        label = m.group("label").lower()
        rows.append(
            Row(
                n=int(m.group("n")),
                dim=int(m.group("dim")),
                method=label,
                stat=float(m.group("stat")),
                p=float(m.group("p")),
                stat_t=float(m.group("stat_t")),
                p_t=float(m.group("p_t")),
            )
        )
    return rows


# ============================================================
# Pybind mode (MAIN PATH)
# ============================================================

def rows_from_pybind(
    sample_sizes: List[int],
    dims: List[int],
    permutations: int,
    delta: float,
    seed: int,
    include_baseline: bool,
) -> List[Row]:
    _ensure_energy_py_on_path()

    try:
        import energy_py
    except ModuleNotFoundError as e:
        raise SystemExit(
            "energy_py module not found. Build it with CMake (pybind11)."
        ) from e

    cfg = energy_py.BenchmarkConfig()
    cfg.permutations = int(permutations)
    cfg.delta = float(delta)
    cfg.seed = int(seed)
    cfg.warmup = True

    # ---------------------------
    # Select methods
    # ---------------------------
    methods = [energy_py.Method.CPU_CROSS_DIST]

    if hasattr(energy_py.Method, "CUDA_CROSS_DIST"):
        methods.append(energy_py.Method.CUDA_CROSS_DIST)

    if include_baseline:
        methods.insert(0, energy_py.Method.CPU_BASELINE)

    print(
        "Running benchmarks via pybind: "
        f"dims={list(dims)} n={list(sample_sizes)} perms={permutations} "
        f"methods={[m.name for m in methods]}"
    )

    results = energy_py.run_benchmarks(sample_sizes, dims, cfg, methods)

    out: List[Row] = []
    for r in results:
        if r.method == energy_py.Method.CPU_BASELINE:
            method = "cpu"
        elif r.method == energy_py.Method.CPU_CROSS_DIST:
            method = "cross"
        elif hasattr(energy_py.Method, "CUDA_CROSS_DIST") and r.method == energy_py.Method.CUDA_CROSS_DIST:
            method = "cuda"
        else:
            method = "unknown"

        out.append(
            Row(
                n=int(r.n),
                dim=int(r.dim),
                method=method,
                stat=float(r.statistic),
                p=float(r.p_value),
                stat_t=float(r.stat_seconds),
                p_t=float(r.p_value_seconds),
            )
        )
    return out


# ============================================================
# Plotting
# ============================================================

def plot(rows: List[Row], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit("matplotlib is required for plotting") from e

    if not rows:
        raise SystemExit("No benchmark rows parsed")

    out_dir.mkdir(parents=True, exist_ok=True)

    dims = sorted({r.dim for r in rows})
    for dim in dims:
        rdim = [r for r in rows if r.dim == dim]

        def series(method: str, field: str):
            xs = [r.n for r in rdim if r.method == method]
            ys = [getattr(r, field) for r in rdim if r.method == method]
            return xs, ys

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Energy test benchmarks (dim={dim})")

        for ax, field, title, ylabel in [
            (axs[0], "stat", "Energy statistic", "stat"),
            (axs[1], "stat_t", "Statistic time", "seconds"),
            (axs[2], "p_t", "P-value time", "seconds"),
        ]:
            for method, label in [
                ("cpu", "CPU baseline"),
                ("cross", "CPU cross-dist"),
                ("cuda", "CUDA cross-dist"),
            ]:
                xs, ys = series(method, field)
                if xs:
                    ax.plot(xs, ys, marker="o", label=label)

            ax.set_xscale("log")
            ax.set_xlabel("n (log scale)")
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", linestyle=":")
            ax.legend()

        fig.tight_layout()
        out_path = out_dir / f"bench_dim_{dim}.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"Saved plot: {out_path}")


# ============================================================
# CLI
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Run energy test benchmarks and plot results")
    ap.add_argument("--mode", choices=["pybind", "subprocess"], default="pybind")
    ap.add_argument(
        "--exe",
        type=Path,
        default=repo_root() / "build" / "bin" / "energy_test_app",
    )
    ap.add_argument("--out-dir", type=Path, default=repo_root() / "run" / "out")
    ap.add_argument("--permutations", type=int, default=199)
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", nargs="*", type=int, default=[100, 500, 2000])
    ap.add_argument("--dim", nargs="*", type=int, default=[2])
    ap.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also run the slow CPU baseline",
    )
    ap.add_argument("--no-print", action="store_true")
    args = ap.parse_args()

    if args.mode == "subprocess":
        text = run_app(args.exe)
        rows = parse_output(text)
    else:
        rows = rows_from_pybind(
            args.n,
            args.dim,
            args.permutations,
            args.delta,
            args.seed,
            args.include_baseline,
        )

    plot(rows, args.out_dir)
    if not args.no_print:
        print_table(rows)

    print(f"\nParsed {len(rows)} rows. Plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
