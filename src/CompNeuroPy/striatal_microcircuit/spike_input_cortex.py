"""Spike-count simulation with controllable shared fractions (group-free).

This module simulates spike counts per receiver using a Gaussian-copula-based
``ReceiverSimulator``. Shared input structure is provided directly as either
(a) a scalar shared fraction applied to all receiver pairs or (b) an explicit
shared-fraction/correlation matrix. Distance-dependent sharing is supported by
constructing such a matrix from receiver positions and a target shared-fraction
function ``f(d)``—no explicit input groups are needed anymore.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union
import math
import tempfile
import os

import numpy as np
from scipy.stats import beta, binom, norm
from tqdm import tqdm
import matplotlib.pyplot as plt


class ReceiverSimulator:
    def __init__(
        self, n_receivers: int, n_steps: int, n_trials: int, rng: np.random.Generator
    ):
        self.n_receivers = n_receivers
        self.n_steps = n_steps
        self.n_trials = n_trials
        self.rng = rng

    def _get_correlated_uniforms(self, correlation_matrix: np.ndarray) -> np.ndarray:
        """Generate correlated U(0, 1) samples via a Gaussian copula."""

        cov_matrix = correlation_matrix.copy()
        np.fill_diagonal(cov_matrix, 1.0)
        mean = np.zeros(self.n_receivers)

        try:
            mv_normal_samples = self.rng.multivariate_normal(
                mean, cov_matrix, size=self.n_steps
            )
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals[eigvals < 1e-6] = 1e-6
            reconstructed_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            d = np.sqrt(np.diag(reconstructed_cov))
            reconstructed_corr = reconstructed_cov / np.outer(d, d)
            mv_normal_samples = self.rng.multivariate_normal(
                mean, reconstructed_corr, size=self.n_steps
            )

        correlated_uniforms = norm.cdf(mv_normal_samples).T
        return correlated_uniforms

    def _scalar_shared_fraction_matrix(self, f: Union[float, np.ndarray]) -> np.ndarray:
        """Return a correlation/shared-fraction matrix from a scalar or matrix input."""

        if np.isscalar(f):
            corr_matrix = np.full(
                (self.n_receivers, self.n_receivers), float(f), dtype=np.float64
            )
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            corr_matrix = np.asarray(f, dtype=np.float64)
        return corr_matrix

    def get_global_p(
        self, dt: float, rate: Union[float, np.ndarray], rho: float
    ) -> np.ndarray:
        """Compute global probability trace (length ``n_steps``) from rate and ``rho``."""

        dt_s = dt * 1e-3
        if np.ndim(rate) > 0:
            rate_arr = np.asarray(rate, dtype=np.float64)
            if rate_arr.shape[0] != self.n_steps:
                raise ValueError(
                    f"rate length {rate_arr.shape[0]} != n_steps {self.n_steps}"
                )
            return rate_arr * dt_s

        p_mean = float(rate) * dt_s
        if rho <= 0.0:
            return np.full((self.n_steps,), p_mean, dtype=np.float64)
        if rho >= 1.0:
            rho = 0.99

        alpha_global, beta_global = beta_params_from_p_rho(p_mean, rho)
        return self.rng.beta(alpha_global, beta_global, size=self.n_steps)

    def generate_p_matrix(
        self,
        global_p_curve: np.ndarray,
        shared_input: Union[float, np.ndarray],
        concentration: float,
    ) -> np.ndarray:
        """Generate receiver-specific probabilities with shared fluctuations."""

        correlation_matrix = self._scalar_shared_fraction_matrix(f=shared_input)
        uniforms = self._get_correlated_uniforms(correlation_matrix)

        epsilon = 1e-9
        safe_p = np.clip(global_p_curve, epsilon, 1.0 - epsilon)
        alpha = safe_p * concentration
        beta_param = (1.0 - safe_p) * concentration
        return beta.ppf(uniforms, alpha, beta_param)

    def simulate(
        self, p_matrix: np.ndarray, shared_input: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Simulate counts with shared variability using correlated Binomial draws."""

        correlation_matrix = self._scalar_shared_fraction_matrix(f=shared_input)
        uniforms = self._get_correlated_uniforms(correlation_matrix)
        final_counts = binom.ppf(uniforms, self.n_trials, p_matrix)
        return final_counts


def beta_params_from_p_rho(p: float, rho: float) -> Tuple[float, float]:
    """Compute Beta distribution parameters for mean ``p`` and variance ``rho * p * (1-p)``."""

    if rho <= 0.0:
        raise ValueError(
            "rho must be > 0 to compute Beta parameters (handle rho==0 separately)."
        )
    if rho >= 1.0:
        raise ValueError("rho must be < 1 for Beta params (handle rho==1 separately).")
    s = 1.0 / rho - 1.0
    alpha = p * s
    beta_val = (1.0 - p) * s
    return alpha, beta_val


@dataclass(frozen=True)
class DistanceCorrelationState:
    receiver_positions: np.ndarray
    L: float
    correlation_matrix: np.ndarray
    f_target_samples: Dict[float, float]


def _periodic_component(delta: float, L: float) -> float:
    return min(abs(delta), L - abs(delta))


def _periodic_distance_float(a: np.ndarray, b: np.ndarray, L: float) -> float:
    dx = _periodic_component(a[0] - b[0], L)
    dy = _periodic_component(a[1] - b[1], L)
    dz = _periodic_component(a[2] - b[2], L)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _pairwise_periodic_distances(
    receiver_positions: np.ndarray, L: float
) -> np.ndarray:
    R = receiver_positions.shape[0]
    dist = np.zeros((R, R), dtype=np.float64)
    for i in range(R):
        for j in range(i + 1, R):
            d = _periodic_distance_float(
                receiver_positions[i], receiver_positions[j], L
            )
            dist[i, j] = dist[j, i] = d
    return dist


def build_distance_correlation_state(
    receiver_positions: np.ndarray,
    bounding_box_width: float,
    f_target: Callable[[float], float],
) -> DistanceCorrelationState:
    """Build a shared-fraction matrix from receiver positions and target ``f(d)``."""

    positions = np.asarray(receiver_positions, dtype=np.float64)
    R = positions.shape[0]
    distances = _pairwise_periodic_distances(positions, bounding_box_width)

    corr = np.eye(R, dtype=np.float64)
    bins: Dict[float, list] = {}
    for i in range(R):
        for j in range(i + 1, R):
            d = float(distances[i, j])
            d_key = float(round(d, 6))
            f_val = float(np.clip(f_target(d), 0.0, 1.0))
            corr[i, j] = corr[j, i] = f_val
            bins.setdefault(d_key, []).append(f_val)

    f_samples = {d: float(np.mean(vals)) for d, vals in bins.items()}
    return DistanceCorrelationState(
        receiver_positions=positions.astype(np.float32),
        L=float(bounding_box_width),
        correlation_matrix=corr,
        f_target_samples=f_samples,
    )


def simulate_receiver_counts_homogeneous(
    R: int,
    N: int,
    shared_input: Union[float, np.ndarray],
    rate: Union[float, np.ndarray],
    dt: float,
    rho: float,
    num_bins: int,
    rng: np.random.Generator,
    concentration: float = 1.0,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Simulate receiver counts using a scalar or matrix shared fraction."""

    sim = ReceiverSimulator(n_receivers=R, n_steps=num_bins, n_trials=N, rng=rng)
    global_p = sim.get_global_p(dt=dt, rate=rate, rho=rho)
    p_matrix = sim.generate_p_matrix(
        global_p_curve=global_p, shared_input=shared_input, concentration=concentration
    )
    counts = sim.simulate(p_matrix=p_matrix, shared_input=shared_input)
    return counts.astype(dtype, copy=False) if dtype else counts


def simulate_receiver_counts_homogeneous_to_memmap(
    filename: str,
    R: int,
    N: int,
    shared_input: Union[float, np.ndarray],
    rate: Union[float, np.ndarray],
    dt: float,
    rho: float,
    num_bins: int,
    receiver_dtype: np.dtype,
    rng: np.random.Generator,
    concentration: float = 1.0,
    verbose: bool = False,
) -> None:
    """Stream homogeneous receiver counts directly to a memmap file."""

    rate_is_timeseries = np.ndim(rate) > 0
    rate_array = None
    if rate_is_timeseries:
        rate_array = np.asarray(rate, dtype=np.float64)
        if rate_array.shape[0] != num_bins:
            raise ValueError(
                f"rate time series must have length num_bins (got {rate_array.shape[0]} vs {num_bins})"
            )

    mm = np.memmap(filename, dtype=receiver_dtype, mode="w+", shape=(R, num_bins))
    target_chunk_bytes = 128 * 1024 * 1024
    # Peak working set has ~4 float64 R x chunk arrays (normal samples, uniforms,
    # p_matrix, counts before casting), so size estimates use float64 regardless of
    # receiver_dtype to avoid underestimation.
    internal_arrays = 4
    peak_bytes_per_bin = R * np.dtype(np.float64).itemsize * internal_arrays
    chunk_size = max(1, min(num_bins, target_chunk_bytes // max(1, peak_bytes_per_bin)))

    if verbose:
        peak_mb = peak_bytes_per_bin * chunk_size / (1024 * 1024)
        print(f"Simulating receiver counts in chunks of size {chunk_size} bins...")
        print(f"Estimated peak RAM per chunk: {peak_mb:.2f} MB")
        print(f"Total chunks: {math.ceil(num_bins / chunk_size)}")

    for start in tqdm(range(0, num_bins, chunk_size)):
        end = min(start + chunk_size, num_bins)
        nchunk = end - start
        rate_chunk = rate_array[start:end] if rate_is_timeseries else rate

        sim = ReceiverSimulator(n_receivers=R, n_steps=nchunk, n_trials=N, rng=rng)
        global_p = sim.get_global_p(dt=dt, rate=rate_chunk, rho=rho)
        p_matrix = sim.generate_p_matrix(
            global_p_curve=global_p,
            shared_input=shared_input,
            concentration=concentration,
        )
        counts = sim.simulate(p_matrix=p_matrix, shared_input=shared_input).astype(
            receiver_dtype, copy=False
        )
        mm[:, start:end] = counts

    mm.flush()


def simulate_receiver_counts_distance_dependent(
    correlation_matrix: np.ndarray,
    N: int,
    rate: Union[float, np.ndarray],
    dt: float,
    rho: float,
    num_bins: int,
    rng: np.random.Generator,
    concentration: float = 1.0,
    dtype: Optional[np.dtype] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Simulate receiver counts using an explicit shared-fraction matrix."""

    corr = np.asarray(correlation_matrix, dtype=np.float64)
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation_matrix must be square")
    R = corr.shape[0]

    sim = ReceiverSimulator(n_receivers=R, n_steps=num_bins, n_trials=N, rng=rng)
    global_p = sim.get_global_p(dt=dt, rate=rate, rho=rho)
    if verbose:
        print(f"Mean global p: {global_p.mean():.6f}")
    p_matrix = sim.generate_p_matrix(
        global_p_curve=global_p, shared_input=corr, concentration=concentration
    )
    counts = sim.simulate(p_matrix=p_matrix, shared_input=corr)
    return counts.astype(dtype, copy=False) if dtype else counts


def simulate_receiver_counts_distance_dependent_to_memmap(
    filename: str,
    correlation_matrix: np.ndarray,
    N: int,
    rate: Union[float, np.ndarray],
    dt: float,
    rho: float,
    num_bins: int,
    receiver_dtype: np.dtype,
    rng: np.random.Generator,
    concentration: float = 1.0,
    verbose: bool = False,
) -> None:
    """Stream distance-dependent receiver counts directly to a memmap file."""

    corr = np.asarray(correlation_matrix, dtype=np.float64)
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("correlation_matrix must be square")
    R = corr.shape[0]

    rate_is_timeseries = np.ndim(rate) > 0
    rate_array = None
    if rate_is_timeseries:
        rate_array = np.asarray(rate, dtype=np.float64)
        if rate_array.shape[0] != num_bins:
            raise ValueError(
                f"rate time series must have length num_bins (got {rate_array.shape[0]} vs {num_bins})"
            )

    mm = np.memmap(filename, dtype=receiver_dtype, mode="w+", shape=(R, num_bins))
    target_chunk_bytes = 128 * 1024 * 1024
    internal_arrays = 4  # normal samples, uniforms, p_matrix, counts (float64)
    peak_bytes_per_bin = R * np.dtype(np.float64).itemsize * internal_arrays
    chunk_size = max(1, min(num_bins, target_chunk_bytes // max(1, peak_bytes_per_bin)))

    if verbose:
        peak_mb = peak_bytes_per_bin * chunk_size / (1024 * 1024)
        print(f"Simulating receiver counts in chunks of size {chunk_size} bins...")
        print(f"Estimated peak RAM per chunk: {peak_mb:.2f} MB")
        print(f"Total chunks: {math.ceil(num_bins / chunk_size)}")

    for start in tqdm(range(0, num_bins, chunk_size)):
        end = min(start + chunk_size, num_bins)
        nchunk = end - start
        rate_chunk = rate_array[start:end] if rate_is_timeseries else rate

        sim = ReceiverSimulator(n_receivers=R, n_steps=nchunk, n_trials=N, rng=rng)
        global_p = sim.get_global_p(dt=dt, rate=rate_chunk, rho=rho)
        p_matrix = sim.generate_p_matrix(
            global_p_curve=global_p, shared_input=corr, concentration=concentration
        )
        counts = sim.simulate(p_matrix=p_matrix, shared_input=corr).astype(
            receiver_dtype, copy=False
        )
        mm[:, start:end] = counts

    mm.flush()


def iter_memmap_spike_counts(
    filename: str,
    R: int,
    num_bins: int,
    receiver_dtype: np.dtype,
    chunk_size: int = 1000,
    copy: bool = False,
    verbose: bool = False,
):
    if verbose:
        print(
            f"Reading memmap file {filename}; per-chunk memory ~{R*chunk_size*np.dtype(receiver_dtype).itemsize/(1024*1024):.2f} MB."
        )
    mm = np.memmap(filename, dtype=receiver_dtype, mode="r", shape=(R, num_bins))
    for start in range(0, num_bins, chunk_size):
        end = min(start + chunk_size, num_bins)
        chunk = mm[:, start:end]
        yield chunk.copy() if copy else chunk


__all__ = [
    "ReceiverSimulator",
    "beta_params_from_p_rho",
    "DistanceCorrelationState",
    "build_distance_correlation_state",
    "simulate_receiver_counts_homogeneous",
    "simulate_receiver_counts_homogeneous_to_memmap",
    "simulate_receiver_counts_distance_dependent",
    "simulate_receiver_counts_distance_dependent_to_memmap",
    "iter_memmap_spike_counts",
]


def _print_stats(title: str, counts: np.ndarray, rate_hz: float, dt_ms: float, N: int):
    dt_s = dt_ms / 1000.0
    expected = rate_hz * dt_s * N * counts.shape[1]
    mean_per_receiver = counts.sum(axis=1).mean()
    print(f"--- {title} ---")
    print(f"Expected total counts/receiver: {expected:.2f}")
    print(f"Observed total counts/receiver: {mean_per_receiver:.2f}")
    print(
        f"Counts shape: {counts.shape}, per-bin mean: {counts.mean():.4f}, std: {counts.std():.4f}"
    )


def _plot_counts(counts: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(counts, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Receiver")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Spike count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    rng = np.random.default_rng()
    dt = 1.0  # ms
    num_bins = 1000
    R = 550
    N = 500
    rate = 15.0  # Hz

    # 1) No rho, no sharing
    counts_none = simulate_receiver_counts_homogeneous(
        R=R,
        N=N,
        shared_input=0.0,
        rate=rate,
        dt=dt,
        rho=0.0,
        num_bins=num_bins,
        rng=rng,
        concentration=1,
    )
    _print_stats("No rho, no sharing", counts_none, rate, dt, N)
    _plot_counts(counts_none[: min(R, 30)], "No rho, no sharing")

    # 2) Only rho
    counts_rho = simulate_receiver_counts_homogeneous(
        R=R,
        N=N,
        shared_input=0.0,
        rate=rate,
        dt=dt,
        rho=0.8,
        num_bins=num_bins,
        rng=rng,
        concentration=1,
    )
    _print_stats("Only rho", counts_rho, rate, dt, N)
    _plot_counts(counts_rho[: min(R, 30)], "Only rho")

    # 3) Only sharing
    counts_share = simulate_receiver_counts_homogeneous(
        R=R,
        N=N,
        shared_input=0.3,
        rate=rate,
        dt=dt,
        rho=0.0,
        num_bins=num_bins,
        rng=rng,
    )
    _print_stats("Only sharing", counts_share, rate, dt, N)
    _plot_counts(counts_share[: min(R, 30)], "Only sharing")

    # 4) rho + homogeneous sharing
    counts_rho_share = simulate_receiver_counts_homogeneous(
        R=R,
        N=N,
        shared_input=0.3,
        rate=rate,
        dt=dt,
        rho=0.4,
        num_bins=num_bins,
        rng=rng,
    )
    _print_stats("rho + sharing", counts_rho_share, rate, dt, N)
    _plot_counts(counts_rho_share[: min(R, 30)], "rho + sharing")

    # 5) Distance-dependent sharing on 1mm^3 grid, half-Gaussian target
    side = 4
    coords = np.linspace(0.0, 1.0, side, endpoint=False) + 0.5 / side
    receiver_positions = np.array(
        [(x, y, z) for x in coords for y in coords for z in coords], dtype=np.float32
    )

    def f_target(d: float) -> float:
        sigma = 0.25
        return float(np.exp(-0.5 * (d / sigma) ** 2))

    dist_state = build_distance_correlation_state(
        receiver_positions=receiver_positions,
        bounding_box_width=1.0,
        f_target=f_target,
    )

    R_dist = dist_state.correlation_matrix.shape[0]
    N_dist = 60
    num_bins_dist = 300
    rate_dist = 12.0

    counts_dist = simulate_receiver_counts_distance_dependent(
        correlation_matrix=dist_state.correlation_matrix,
        N=N_dist,
        rate=rate_dist,
        dt=dt,
        rho=0.0,
        num_bins=num_bins_dist,
        rng=rng,
        verbose=True,
    )
    _print_stats("Distance dependent (no rho)", counts_dist, rate_dist, dt, N_dist)
    _plot_counts(counts_dist[: min(R_dist, 40)], "Distance dependent (no rho)")

    counts_dist_rho = simulate_receiver_counts_distance_dependent(
        correlation_matrix=dist_state.correlation_matrix,
        N=N_dist,
        rate=rate_dist,
        dt=dt,
        rho=0.35,
        num_bins=num_bins_dist,
        rng=rng,
        verbose=True,
    )
    _print_stats(
        "Distance dependent (with rho)", counts_dist_rho, rate_dist, dt, N_dist
    )
    _plot_counts(counts_dist_rho[: min(R_dist, 40)], "Distance dependent (with rho)")

    # 6) Memmap equivalence for homogeneous and distance-dependent
    with tempfile.TemporaryDirectory() as tmpdir:
        fname_h = os.path.join(tmpdir, "homogeneous.dat")
        simulate_receiver_counts_homogeneous_to_memmap(
            filename=fname_h,
            R=R,
            N=N,
            shared_input=0.2,
            rate=rate,
            dt=dt,
            rho=0.2,
            num_bins=num_bins,
            receiver_dtype=np.float64,
            rng=np.random.default_rng(999),
            verbose=True,
        )
        mem_counts_h = next(
            iter_memmap_spike_counts(
                filename=fname_h,
                R=R,
                num_bins=num_bins,
                receiver_dtype=np.float64,
                chunk_size=num_bins,
            )
        )
        direct_counts_h = simulate_receiver_counts_homogeneous(
            R=R,
            N=N,
            shared_input=0.2,
            rate=rate,
            dt=dt,
            rho=0.2,
            num_bins=num_bins,
            rng=np.random.default_rng(999),
        )
        diff_h = np.abs(mem_counts_h - direct_counts_h).max()
        print(f"Memmap vs direct (homogeneous) max diff: {diff_h}")

        fname_d = os.path.join(tmpdir, "distance.dat")
        simulate_receiver_counts_distance_dependent_to_memmap(
            filename=fname_d,
            correlation_matrix=dist_state.correlation_matrix,
            N=N_dist,
            rate=rate_dist,
            dt=dt,
            rho=0.25,
            num_bins=num_bins_dist,
            receiver_dtype=np.float64,
            rng=np.random.default_rng(1001),
            verbose=True,
        )
        mem_counts_d = next(
            iter_memmap_spike_counts(
                filename=fname_d,
                R=R_dist,
                num_bins=num_bins_dist,
                receiver_dtype=np.float64,
                chunk_size=num_bins_dist,
            )
        )
        direct_counts_d = simulate_receiver_counts_distance_dependent(
            correlation_matrix=dist_state.correlation_matrix,
            N=N_dist,
            rate=rate_dist,
            dt=dt,
            rho=0.25,
            num_bins=num_bins_dist,
            rng=np.random.default_rng(1001),
        )
        diff_d = np.abs(mem_counts_d - direct_counts_d).max()
        print(f"Memmap vs direct (distance-dependent) max diff: {diff_d}")
