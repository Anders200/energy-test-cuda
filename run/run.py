
from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


def sample_normal_py(n: int, dim: int, mean_shift: float, seed: int) -> List[List[float]]:
	"""Generate N(0,1) points in Python.

	We use Python's built-in RNG so generation is guaranteed to happen on the Python side.
	"""
	import random
	rng = random.Random(int(seed))
	out: List[List[float]] = []
	for _ in range(int(n)):
		out.append([rng.gauss(0.0, 1.0) + float(mean_shift) for _ in range(int(dim))])
	return out


def to_points(energy_py, xs: List[List[float]]):
	pts = []
	for row in xs:
		p = energy_py.Point()
		p.coords = [float(v) for v in row]
		pts.append(p)
	return pts


def print_table(rows: List[Row]) -> None:
	if not rows:
		print("(no results)")
		return

	# Group by dim then by n for nice console output
	dims = sorted({r.dim for r in rows})
	for dim in dims:
		print(f"\n=== Dimension {dim} ===")
		rdim = [r for r in rows if r.dim == dim]
		for n in sorted({r.n for r in rdim}):
			for r in [x for x in rdim if x.n == n]:
				label = r.method.upper() if r.method in {"cpu", "cross", "cuda"} else r.method
				print(
					f"{n:7d} | dim={dim:<3d} | {label:5s} | "
					f"stat={r.stat:12.6g} | p={r.p:8.4g} | "
					f"stat_t={r.stat_t:8.4f}s | p_t={r.p_t:8.4f}s"
				)


@dataclass
class Row:
	n: int
	dim: int
	method: str  # "cpu" or "cross"
	stat: float
	p: float
	stat_t: float
	p_t: float


LINE_RE = re.compile(
	r"^\s*(?P<n>\d+)\s*\|\s*dim=(?P<dim>\d+)\s*\|\s*"
	r"(?:(?P<m_cpu>CPU)\s+stat=\s*|(?P<m_cross>CROSS)\s+stat=)\s*"
	r"(?P<stat>[-+0-9.eE]+)\s*\|\s*p=(?P<p>[-+0-9.eE]+)\s*\|\s*"
	r"stat_t=(?P<stat_t>[-+0-9.eE]+)s\s*\|\s*p_t=(?P<p_t>[-+0-9.eE]+)s\s*$"
)


def repo_root() -> Path:
	return Path(__file__).resolve().parents[1]


def _ensure_energy_py_on_path() -> None:
	"""Make a best-effort attempt to import the pybind module built by CMake.

	By default, `pybind11_add_module` places the extension into `build/lib/`.
	When running `python run/run.py`, that directory isn't on sys.path.
	"""
	build_lib = repo_root() / "build" / "lib"
	if build_lib.exists() and str(build_lib) not in sys.path:
		sys.path.insert(0, str(build_lib))


def run_app(exe: Path) -> str:
	proc = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
	return proc.stdout


def parse_output(text: str) -> List[Row]:
	rows: List[Row] = []
	for line in text.splitlines():
		m = LINE_RE.match(line)
		if not m:
			continue
		method = "cpu" if m.group("m_cpu") else "cross"
		rows.append(
			Row(
				n=int(m.group("n")),
				dim=int(m.group("dim")),
				method=method,
				stat=float(m.group("stat")),
				p=float(m.group("p")),
				stat_t=float(m.group("stat_t")),
				p_t=float(m.group("p_t")),
			)
		)
	return rows


def plot(rows: List[Row], out_dir: Path) -> None:
	try:
		import matplotlib.pyplot as plt
	except ModuleNotFoundError as e:
		raise SystemExit(
			"matplotlib is required for plotting. Install it and re-run."
		) from e

	if not rows:
		raise SystemExit("No benchmark rows parsed from app output")

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


def rows_from_pybind(
	sample_sizes: List[int],
	dims: List[int],
	permutations: int,
	delta: float,
	seed: int,
	include_baseline: bool,
	include_cuda: bool,
) -> List[Row]:
	_ensure_energy_py_on_path()
	try:
		import energy_py
	except ModuleNotFoundError as e:
		raise SystemExit(
			"energy_py module not found. Build it with CMake (pybind11). "
			"Or run with --mode subprocess to parse the app output."
		) from e

	cfg = energy_py.BenchmarkConfig()
	cfg.permutations = int(permutations)
	cfg.delta = float(delta)
	cfg.seed = int(seed)
	cfg.warmup = True

	# CUDA note: the GPU backend now attempts all sizes, but it materializes the full pooled
	# distance matrix D of size N*N (N=2n). Large N may be slow or run out of GPU memory.
	if include_cuda:
		max_n = max(sample_sizes) if sample_sizes else 0
		max_N = 2 * int(max_n)
		approx_bytes = 4 * (max_N * max_N)  # float32
		approx_gib = approx_bytes / (1024**3) if max_N > 0 else 0.0
		if max_N >= 4096:
			print(
				f"NOTE: CUDA enabled for large N (max N={max_N}); distance matrix is ~{approx_gib:.2f} GiB. "
				"If you hit OOM, reduce n or run CPU-only."
			)

	# Progress prints so long runs feel alive.
	methods_label = ["cross-dist"]
	if include_baseline:
		methods_label.append("baseline")
	if include_cuda:
		methods_label.append("cuda")
	print(
		"Running benchmarks via pybind: "
		f"dims={list(dims)} n={list(sample_sizes)} perms={permutations} "
		f"methods={methods_label}"
	)

	methods = [energy_py.Method.CPU_CROSS_DIST]
	if include_baseline:
		methods.insert(0, energy_py.Method.CPU_BASELINE)
	if include_cuda:
		if hasattr(energy_py.Method, "CUDA_CROSS_DIST"):
			methods.append(energy_py.Method.CUDA_CROSS_DIST)
		else:
			print("NOTE: energy_py.Method.CUDA_CROSS_DIST not available; rebuild with CUDA enabled")

	# Generate distributions in Python and pass X/Y into C++.
	results = []
	for dim in dims:
		for n in sample_sizes:
			X_py = sample_normal_py(int(n), int(dim), 0.0, seed=int(seed))
			# Use a different seed stream for Y but keep it deterministic.
			Y_py = sample_normal_py(int(n), int(dim), delta, seed=int(seed) ^ 0x9E3779B9)
			X = to_points(energy_py, X_py)
			Y = to_points(energy_py, Y_py)
			for m in methods:
				results.append(energy_py.run_benchmark_xy(cfg, m, X, Y))

	out: List[Row] = []
	for r in results:
		if r.method == energy_py.Method.CPU_BASELINE:
			method = "cpu"
		elif hasattr(energy_py.Method, "CUDA_CROSS_DIST") and r.method == energy_py.Method.CUDA_CROSS_DIST:
			method = "cuda"
		else:
			method = "cross"
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


def main() -> None:
	ap = argparse.ArgumentParser(description="Run energy_test_app and plot benchmark metrics")
	ap.add_argument(
		"--mode",
		choices=["pybind", "subprocess"],
		default="pybind",
		help="pybind: call C++ directly via energy_py; subprocess: parse energy_test_app output",
	)
	ap.add_argument(
		"--exe",
		type=Path,
		default=repo_root() / "build" / "bin" / "energy_test_app",
		help="Path to energy_test_app executable",
	)
	ap.add_argument(
		"--out-dir",
		type=Path,
		default=repo_root() / "run" / "out",
		help="Output directory for plots",
	)
	ap.add_argument("--permutations", type=int, default=500)
	ap.add_argument("--delta", type=float, default=0.2)
	ap.add_argument("--seed", type=int, default=42)
	ap.add_argument("--n", nargs="*", type=int, default=[100, 500, 1000], help="Sample sizes")
	ap.add_argument("--dim", nargs="*", type=int, default=[2], help="Dimensions")
	ap.add_argument(
		"--include-baseline",
		action="store_true",
		help="Also run the slow CPU baseline method (off by default)",
	)
	ap.add_argument(
		"--include-cuda",
		action="store_true",
		help="Also run CUDA cross-dist method (requires CUDA-enabled build)",
	)
	ap.add_argument(
		"--no-print",
		action="store_true",
		help="Don't print per-run results to stdout",
	)
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
			args.include_cuda,
		)

	plot(rows, args.out_dir)
	if not args.no_print:
		print_table(rows)
	print(f"Parsed {len(rows)} rows. Plots saved to: {args.out_dir}")


if __name__ == "__main__":
	main()

