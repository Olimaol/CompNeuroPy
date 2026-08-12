"""Spike-count streams standing in for presynaptic pools that are not simulated.

A stream is an ``(R, n_steps)`` matrix of counts: how many of a pair's ``N_eff``
presynaptic neurons spiked into receiver ``i`` during bin ``t``. What such a
stream must reproduce is fixed by what it stands in for -- ``N_eff`` neurons
firing at a stated rate whose afferent pools overlap between receivers by a
stated fraction ``f`` -- and is therefore not a modelling choice:

    mean = N * p(t)
    var  = N*p*(1 - p*(1 + sigma^2)) + (N*p)^2 * sigma^2     Fano = var / (N*p)
    cov  = f*N*p*(1 - p*(1 + sigma^2)) + (N*p)^2 * sigma^2   corr = cov / var

where ``sigma^2`` is the shared-modulation variance (``stream_target_statistics``
implements exactly these). In terms of the presynaptic pairwise correlation
``rho = p*sigma^2/(1-p)`` the same law reads ``Fano = (1-p)*((1-rho) + N*rho)``
and ``corr = (f*(1-rho) + N*rho) / ((1-rho) + N*rho)``, which is the form the
contract document states; at ``rho = sigma = 0`` both reduce to
``Fano = 1 - p``, ``corr = f``.

See ``BGM_22/experimental_data/input_streams/README.md`` for the full contract,
the sources of each input quantity, and the list of things this approach
deliberately cannot represent.

Three constructions are provided. All are exact in the marginal, and all work
the same way: realise the presynaptic pool explicitly and let the overlaps
produce the correlations, rather than computing a correlation and imposing it.

``simulate_cortical_axon_pool_streams_to_memmap``
    The striatal cortical streams. All receiver types of one region sample the
    same pool of cortical axons, so the shared fractions *between* types --
    dSPN to iSPN, FS to SPN -- follow from the pool size instead of being free
    parameters.

``simulate_receiver_counts_geometric_to_memmap``
    The missing-GABA streams. Places virtual source neurons around the receiver
    lattice and connects each receiver to them with the same distance kernel the
    model is built from, so the shared fraction ``f(d)`` *emerges* from the
    overlap. Correct at any lattice size or density.

``simulate_receiver_counts_homogeneous_to_memmap``
    A single population with a flat shared fraction (the ``CorticalInputs``
    streams for thal/GPe/STN). Splits each receiver's afferents into shared and
    private and draws each with a Binomial.

The homogeneous construction is not made redundant by the pool: shared+private
with fraction ``f`` gives the same pairwise statistics as a pool of
``M = N/f`` axons, so the pool subsumes it -- except at ``f = 0``, which would
need an infinite pool and falls out of the split naturally as a pure-private
draw (``CorticalInputs`` runs at exactly that). Conversely, one pool ties the
fractions of every receiver type sampling it to ``f_i = N_i/M``, where the
shared+private split leaves each type's ``f`` free.

Both take an optional shared rate modulation carrying a real correlation
timescale (``tau_c``), because a spike-count correlation is only meaningful
together with the measurement window it was observed at.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union
import math

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import gamma as gamma_dist, norm
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Shared rate modulation
# ---------------------------------------------------------------------------


def ou_sum_variance(m: int, a: float) -> float:
    """Variance of the sum of ``m`` consecutive samples of a unit-variance AR(1).

    The process has autocorrelation ``a**k``, so
    ``W(m) = m + 2 * sum_{k=1}^{m-1} (m - k) * a**k``. Evaluated in closed form
    because ``m`` is the measurement window in units of ``dt`` and is routinely
    10 000 or more.

    Args:
        m (int):
            Number of consecutive samples.

        a (float):
            AR(1) coefficient in [0, 1).

    Returns:
        W (float):
            The variance. ``W(m) = m`` when ``a == 0``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if not (0.0 <= a < 1.0):
        raise ValueError(f"AR(1) coefficient must be in [0, 1), got {a}")
    if a == 0.0:
        return float(m)
    one_minus_a = 1.0 - a
    geom = a * (1.0 - a ** (m - 1)) / one_minus_a
    weighted = a * (1.0 - m * a ** (m - 1) + (m - 1) * a**m) / one_minus_a**2
    return float(m + 2.0 * (m * geom - weighted))


def solve_modulation_amplitude(
    r_sc: float, t_meas_ms: float, tau_c_ms: float, dt: float, p_bar: float
) -> float:
    """Amplitude of the shared rate modulation that yields a target ``r_sc``.

    The presynaptic neurons are taken to share a multiplicative rate modulation
    ``M(t)`` with mean 1, variance ``sigma**2`` and autocorrelation timescale
    ``tau_c``. Two such neurons then have a spike-count correlation, over a
    window of ``m = t_meas / dt`` bins,

        r_sc = p * sigma^2 * W(m) / (m + p * sigma^2 * W(m))

    which this inverts. Note the correlation is *only* defined together with the
    window it is measured at: with ``tau_c = 0`` the value is the same at every
    window, which is exactly why the old white-noise implementation could not be
    right at both the simulation timestep and the hundreds-of-milliseconds
    windows the literature reports.

    Args:
        r_sc (float):
            Target pairwise spike-count correlation between two presynaptic
            neurons, in [0, 1).

        t_meas_ms (float):
            Measurement window the target was observed at, in ms.

        tau_c_ms (float):
            Correlation timescale in ms. 0 gives white modulation.

        dt (float):
            Simulation timestep in ms.

        p_bar (float):
            Mean per-neuron spike probability per bin.

    Returns:
        sigma (float):
            Standard deviation of the modulation.
    """
    if not (0.0 <= r_sc < 1.0):
        raise ValueError(f"r_sc must be in [0, 1), got {r_sc}")
    if r_sc == 0.0:
        return 0.0
    if t_meas_ms <= 0.0:
        raise ValueError(
            "t_meas_ms must be > 0: a spike-count correlation is meaningless "
            "without the measurement window it was observed at."
        )
    if p_bar <= 0.0:
        raise ValueError(f"p_bar must be > 0, got {p_bar}")

    m = int(round(t_meas_ms / dt))
    if m < 1:
        raise ValueError(
            f"measurement window {t_meas_ms} ms is shorter than dt {dt} ms"
        )
    a = math.exp(-dt / tau_c_ms) if tau_c_ms > 0.0 else 0.0
    w = ou_sum_variance(m, a)
    sigma2 = r_sc * m / ((1.0 - r_sc) * p_bar * w)
    return float(math.sqrt(sigma2))


def make_global_p_trace(
    rate: Union[float, np.ndarray],
    dt: float,
    num_bins: int,
    rng: np.random.Generator,
    r_sc: float = 0.0,
    tau_c_ms: float = 0.0,
    t_meas_ms: Optional[float] = None,
    carry: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Per-bin spike probability, shared by every receiver of one stream.

    Two things enter. The drive itself -- ``p(t) = rate(t) * dt / 1000`` -- and,
    if ``r_sc > 0``, a shared multiplicative modulation standing in for the
    pairwise correlation among the presynaptic neurons themselves.

    The modulation has a Gamma marginal with mean 1 and variance ``sigma**2``,
    so it is strictly positive for any amplitude and leaves the mean drive
    untouched, and it is autocorrelated by driving it through an AR(1) Gaussian
    copula. Cost is ``O(num_bins)``, not ``O(R * num_bins)``, because it is one
    trace per stream rather than one per receiver.

    Args:
        rate (float or np.ndarray):
            Presynaptic firing rate in Hz. A scalar, or one value per bin.

        dt (float):
            Timestep in ms.

        num_bins (int):
            Number of bins to produce.

        rng (np.random.Generator):
            Random source.

        r_sc (float):
            Pairwise spike-count correlation among the presynaptic neurons.

        tau_c_ms (float):
            Correlation timescale in ms.

        t_meas_ms (float, optional):
            Window ``r_sc`` was measured at. Required when ``r_sc > 0``.

        carry (float, optional):
            Last AR(1) value of the previous chunk, so the modulation is
            continuous across chunk boundaries.

    Returns:
        p_t (np.ndarray):
            Per-bin probability, shape ``(num_bins,)``.

        sigma (float):
            Modulation amplitude actually used.

        carry_out (float):
            AR(1) state to hand to the next chunk.
    """
    dt_s = dt * 1e-3
    if np.ndim(rate) > 0:
        rate_arr = np.asarray(rate, dtype=np.float64)
        if rate_arr.shape[0] != num_bins:
            raise ValueError(
                f"rate length {rate_arr.shape[0]} != num_bins {num_bins}"
            )
        p_drive = rate_arr * dt_s
    else:
        p_drive = np.full((num_bins,), float(rate) * dt_s, dtype=np.float64)

    if r_sc <= 0.0:
        return p_drive, 0.0, 0.0

    p_bar = float(np.mean(p_drive))
    sigma = solve_modulation_amplitude(
        r_sc=r_sc, t_meas_ms=t_meas_ms, tau_c_ms=tau_c_ms, dt=dt, p_bar=p_bar
    )

    a = math.exp(-dt / tau_c_ms) if tau_c_ms > 0.0 else 0.0
    z = _ar1_trace(num_bins=num_bins, a=a, rng=rng, carry=carry)
    # Gamma marginal, mean 1, variance sigma^2
    shape_k = 1.0 / (sigma * sigma)
    # kept off 0 and 1 because norm.cdf saturates past |z| ~ 8.3 and
    # gamma.ppf(1.0) is infinite
    uniforms = np.clip(norm.cdf(z), 1e-12, 1.0 - 1e-12)
    modulation = gamma_dist.ppf(uniforms, a=shape_k, scale=sigma * sigma)
    p_t = np.clip(p_drive * modulation, 0.0, 1.0)
    return p_t, sigma, float(z[-1])


def _ar1_trace(
    num_bins: int, a: float, rng: np.random.Generator, carry: Optional[float]
) -> np.ndarray:
    """Unit-variance AR(1) Gaussian trace, continuous across chunk boundaries."""
    xi = rng.standard_normal(num_bins)
    if a == 0.0:
        return xi
    from scipy.signal import lfilter

    # lfilter starts from a zero initial state, so the leading samples would have
    # reduced variance. Seeding the first chunk from the stationary distribution
    # keeps the process stationary from step 1, which matters because the
    # self-check measures the first bins written.
    if carry is None:
        carry = float(rng.standard_normal())
    z = lfilter([math.sqrt(1.0 - a * a)], [1.0, -a], xi)
    return z + carry * a ** np.arange(1, num_bins + 1)


# ---------------------------------------------------------------------------
# Target and measured statistics
# ---------------------------------------------------------------------------


def stream_target_statistics(
    n_eff: float, p_bar: float, shared_fraction: float, sigma: float
) -> Dict[str, float]:
    """Closed-form statistics a stream with these parameters must show.

    Derived from the definition, not fitted. ``build_input_caches`` checks every
    stream against these and raises on a mismatch -- the half of ``TODO.md``
    section 22 that said nothing checks the result.

    Args:
        n_eff (float):
            Number of presynaptic neurons per receiver. May be fractional: the
            geometric construction gives every receiver an integer degree but
            their mean is not an integer, and rounding it biases the target
            noticeably on the small pairs (FS to FS has only ~12 afferents).

        p_bar (float):
            Mean per-neuron spike probability per bin.

        shared_fraction (float):
            Mean fraction of the pool two receivers have in common.

        sigma (float):
            Amplitude of the shared rate modulation (0 if none).

    Returns:
        stats (dict):
            ``mean``, ``fano`` and ``corr``.
    """
    var_mod = sigma * sigma
    mean = n_eff * p_bar
    # Binomial(N, p*M) with E[M] = 1, Var[M] = sigma^2:
    #   Var = N*p*(1 - p*(1 + sigma^2)) + N^2 * p^2 * sigma^2
    var = n_eff * p_bar * (1.0 - p_bar * (1.0 + var_mod)) + (
        n_eff * p_bar
    ) ** 2 * var_mod
    fano = var / mean if mean > 0 else float("nan")
    cov = shared_fraction * n_eff * p_bar * (
        1.0 - p_bar * (1.0 + var_mod)
    ) + (n_eff * p_bar) ** 2 * var_mod
    corr = cov / var if var > 0 else float("nan")
    return {"mean": float(mean), "fano": float(fano), "corr": float(corr)}


def measure_stream_statistics(
    counts: np.ndarray, max_receivers: int = 150
) -> Dict[str, float]:
    """Mean, Fano factor and mean pairwise correlation of a stream sample.

    Args:
        counts (np.ndarray):
            ``(R, n)`` block of counts from a stretch over which the drive is
            stationary, so the measured Fano factor is comparable with the
            single-bin target.

        max_receivers (int):
            Cap on the receivers entering the correlation, which is O(R^2). The
            subset is drawn at random rather than taken from the front: receiver
            index order follows the lattice, which is x-major, so the leading
            rows are spatially clustered. On a stream whose shared fraction falls
            with distance that biases the measured correlation upwards -- for
            FS to iSPN, where f(d) spans 0.20 to 0.067, by about 25 %.

    Returns:
        stats (dict):
            ``mean``, ``fano`` and ``corr``.
    """
    block = np.asarray(counts, dtype=np.float64)
    mean = float(block.mean())
    fano = float(block.var() / mean) if mean > 0 else float("nan")

    r = min(block.shape[0], max_receivers)
    corr = float("nan")
    if r >= 2:
        picked = np.random.default_rng(0).choice(
            block.shape[0], size=r, replace=False
        )
        sub = block[np.sort(picked)]
        # drop receivers with no variance, they make corrcoef produce NaN
        keep = sub.std(axis=1) > 0
        if keep.sum() >= 2:
            cc = np.corrcoef(sub[keep])
            iu = np.triu_indices(cc.shape[0], 1)
            corr = float(np.nanmean(cc[iu]))
    return {"mean": mean, "fano": fano, "corr": corr}


def check_stream_statistics(
    label: str,
    target: Dict[str, float],
    measured: Dict[str, float],
    sample_shape: Optional[Tuple[int, int]] = None,
    mean_rtol: float = 0.02,
    fano_rtol: float = 0.10,
    corr_atol: float = 0.02,
    corr_rtol: float = 0.20,
) -> None:
    """Raise if a stream does not show the statistics its parameters imply.

    Args:
        label (str):
            Stream name, used in the error message.

        target (dict):
            From ``stream_target_statistics``.

        measured (dict):
            From ``measure_stream_statistics``.

        sample_shape (tuple, optional):
            ``(n_receivers, n_bins)`` of the sample the measurement came from.
            When given, the mean tolerance is widened to four standard errors of
            the sample mean, which for a sparse stream is much larger than the
            flat relative tolerance. A stream like M1 to FS -- 29 receivers, 56
            afferents, 0.02 counts per bin -- has a 1 % standard error on its
            mean, so a flat 2 % would fire on a perfectly good draw.

        mean_rtol, fano_rtol (float):
            Relative tolerances. ``mean_rtol`` acts as a floor.

        corr_atol, corr_rtol (float):
            The correlation is checked on whichever of the two is looser; it is
            the noisiest of the three, and the shared fractions are small enough
            that a purely relative test would be dominated by sampling error.

    Raises:
        ValueError: if any statistic is out of tolerance.
    """
    problems = []
    effective_mean_rtol = mean_rtol
    if sample_shape is not None and target["mean"] > 0:
        n_receivers, n_bins = sample_shape
        # Bins are independent; receivers are not, so the effective sample size
        # shrinks by the usual 1 + (R - 1) * corr factor.
        inflation = 1.0 + max(n_receivers - 1, 0) * max(target["corr"], 0.0)
        rel_se = math.sqrt(
            target["fano"] * inflation / (target["mean"] * n_receivers * n_bins)
        )
        effective_mean_rtol = max(mean_rtol, 4.0 * rel_se)
    if not np.isclose(
        measured["mean"], target["mean"], rtol=effective_mean_rtol, atol=0.0
    ):
        problems.append(
            f"mean {measured['mean']:.5g} != target {target['mean']:.5g} "
            f"(rtol {effective_mean_rtol:.4g})"
        )
    if not np.isclose(measured["fano"], target["fano"], rtol=fano_rtol, atol=0.0):
        problems.append(
            f"Fano {measured['fano']:.5g} != target {target['fano']:.5g} "
            f"(rtol {fano_rtol})"
        )
    if not np.isnan(measured["corr"]):
        tol = max(corr_atol, corr_rtol * abs(target["corr"]))
        if abs(measured["corr"] - target["corr"]) > tol:
            problems.append(
                f"corr {measured['corr']:.5g} != target {target['corr']:.5g} "
                f"(tol {tol:.5g})"
            )
    if problems:
        raise ValueError(
            f"Stream '{label}' does not match the statistics its own parameters "
            f"imply: " + "; ".join(problems) + ". See "
            "BGM_22/experimental_data/input_streams/README.md for what a stream "
            "is required to reproduce."
        )


# ---------------------------------------------------------------------------
# Flat shared fraction: shared + private Binomial
# ---------------------------------------------------------------------------


def simulate_receiver_counts_homogeneous_to_memmap(
    filename: str,
    R: int,
    N: int,
    shared_input: float,
    rate: Union[float, np.ndarray],
    dt: float,
    num_bins: int,
    receiver_dtype: np.dtype,
    rng: np.random.Generator,
    r_sc: float = 0.0,
    tau_c_ms: float = 0.0,
    t_meas_ms: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Stream counts for a pool with a flat shared fraction, straight to a memmap.

    Each receiver's ``N`` afferents split into ``round(f * N)`` it shares with
    every other receiver and ``N - round(f * N)`` of its own. The shared ones are
    drawn once per bin, the private ones per receiver per bin. The sum is exactly
    ``Binomial(N, p)`` and the pairwise correlation is exactly ``f``, because
    that is literally what a shared sub-pool plus a private sub-pool means.

    Args:
        filename (str):
            Destination memmap path, shape ``(R, num_bins)``.

        R (int):
            Number of receivers.

        N (int):
            Presynaptic neurons per receiver.

        shared_input (float):
            Shared fraction ``f`` in [0, 1].

        rate (float or np.ndarray):
            Presynaptic rate in Hz, scalar or one value per bin.

        dt (float):
            Timestep in ms.

        num_bins (int):
            Length of the stream.

        receiver_dtype (np.dtype):
            Storage dtype.

        rng (np.random.Generator):
            Random source.

        r_sc, tau_c_ms, t_meas_ms:
            Shared rate modulation, see ``make_global_p_trace``.

        verbose (bool):
            Print chunking information.

    Returns:
        stats (dict):
            ``target`` and ``measured`` statistics, and ``sigma``.
    """
    if not (0.0 <= shared_input <= 1.0):
        raise ValueError(f"shared_input must be in [0, 1], got {shared_input}")
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")

    n_shared = int(round(shared_input * N))
    n_private = N - n_shared

    rate_is_series = np.ndim(rate) > 0
    rate_arr = np.asarray(rate, dtype=np.float64) if rate_is_series else None
    if rate_is_series and rate_arr.shape[0] != num_bins:
        raise ValueError(
            f"rate time series must have length num_bins "
            f"(got {rate_arr.shape[0]} vs {num_bins})"
        )

    mm = np.memmap(filename, dtype=receiver_dtype, mode="w+", shape=(R, num_bins))
    chunk_size = _chunk_size_for(R, num_bins)
    if verbose:
        print(
            f"  shared+private: N={N} (shared {n_shared}, private {n_private}), "
            f"R={R}, chunks of {chunk_size} bins"
        )

    p_sum = 0.0
    sigma = 0.0
    carry = None
    sample_block = None
    sample_p_bar = None
    for start in tqdm(range(0, num_bins, chunk_size), disable=not verbose):
        end = min(start + chunk_size, num_bins)
        n = end - start
        rate_chunk = rate_arr[start:end] if rate_is_series else rate
        p_t, sigma, carry = make_global_p_trace(
            rate=rate_chunk,
            dt=dt,
            num_bins=n,
            rng=rng,
            r_sc=r_sc,
            tau_c_ms=tau_c_ms,
            t_meas_ms=t_meas_ms,
            carry=carry,
        )
        p_sum += float(p_t.sum())

        counts = rng.binomial(n_private, np.broadcast_to(p_t, (R, n)))
        if n_shared > 0:
            counts = counts + rng.binomial(n_shared, p_t)[None, :]
        mm[:, start:end] = counts.astype(receiver_dtype, copy=False)
        if sample_block is None:
            n_sample = min(n, _SAMPLE_BINS)
            sample_block = counts[:, :n_sample].copy()
            sample_p_bar = float(p_t[:n_sample].mean())

    mm.flush()
    del mm

    p_bar = p_sum / num_bins
    target = stream_target_statistics(
        n_eff=N, p_bar=sample_p_bar, shared_fraction=shared_input, sigma=sigma
    )
    measured = measure_stream_statistics(sample_block)
    return {
        "target": target,
        "measured": measured,
        "sigma": sigma,
        "p_bar": p_bar,
        "sample_shape": tuple(sample_block.shape),
    }


# How many bins of the first chunk are kept for the self-check. The drive is
# constant across a TR (23 100 bins), so a sample this size sits inside one TR
# and the measured Fano factor is comparable with the single-bin target.
_SAMPLE_BINS = 20000


def _chunk_size_for(R: int, num_bins: int, target_bytes: int = 128 * 1024 * 1024) -> int:
    """Bins per chunk so the working set stays near ``target_bytes``."""
    per_bin = max(1, R * np.dtype(np.int64).itemsize * 3)
    return max(1, min(num_bins, target_bytes // per_bin))


def axon_pool_size(n_reference: int, shared_fraction: float) -> int:
    """Size of the cortical axon pool implied by a measured shared fraction.

    Kincaid et al. 1998 report that one corticostriatal axon contacts <=1.4 % of
    the cells in its arborization. If a pool of ``M`` axons is sampled by every
    striatal neuron, and neuron ``i`` draws ``N_i`` of them, then two neurons
    share ``N_i * N_j / M`` axons and their counts correlate at
    ``sqrt(N_i * N_j) / M``. For two neurons of the reference type that is
    ``N / M``, so ``M = N / f``.

    With ``N = 7000`` and ``f = 0.014`` this gives 500 000 axons, the same order
    as the ~380 000 cortical axons Kincaid reports innervating the dendritic
    volume of one spiny neuron.

    Args:
        n_reference (int):
            Afferent count of the cell type the shared fraction was measured on
            (the SPNs).

        shared_fraction (float):
            Measured shared fraction for two neurons of that type.

    Returns:
        M (int):
            Axon pool size.
    """
    if not (0.0 < shared_fraction <= 1.0):
        raise ValueError(
            f"shared_fraction must be in (0, 1] to define a pool size, got "
            f"{shared_fraction}"
        )
    return int(round(n_reference / shared_fraction))


def simulate_cortical_axon_pool_streams_to_memmap(
    streams: Dict[str, Dict],
    pool_size: int,
    rate: Union[float, np.ndarray],
    dt: float,
    num_bins: int,
    receiver_dtype: np.dtype,
    rng: np.random.Generator,
    r_sc: float = 0.0,
    tau_c_ms: float = 0.0,
    t_meas_ms: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Dict]:
    """Stream counts for several receiver types drawing on one cortical axon pool.

    All receiver types of one cortical region sample the *same* pool of
    ``pool_size`` axons, so their inputs are correlated with each other and not
    only within a type. That matters: drawn independently per receiver type, a
    dSPN and an iSPN sitting in the same tissue and sampling the same axons would
    share nothing at all, and an FS would share nothing with the SPNs it
    inhibits.

    Per bin, ``k ~ Binomial(pool_size, p)`` axons fire, and receiver ``i`` sees
    ``Hypergeometric(pool_size, N_i, k)`` of them. That is exact in every
    respect: marginals are ``Binomial(N_i, p)`` and the correlation between any
    two receivers is ``sqrt(N_i * N_j) / pool_size``, whether they are the same
    type or not. The cross-type shared fractions are therefore *derived* from the
    pool size rather than being free parameters.

    Args:
        streams (dict):
            ``name -> {"filename": str, "R": int, "N": int}``. ``N`` is that
            receiver type's afferent count from this region.

        pool_size (int):
            Number of axons in the region's pool, from ``axon_pool_size``.

        rate (float or np.ndarray):
            Presynaptic rate in Hz, scalar or one value per bin.

        dt (float):
            Timestep in ms.

        num_bins (int):
            Length of the streams.

        receiver_dtype (np.dtype):
            Storage dtype.

        rng (np.random.Generator):
            Random source.

        r_sc, tau_c_ms, t_meas_ms:
            Shared rate modulation, see ``make_global_p_trace``.

        verbose (bool):
            Print chunking information.

    Returns:
        stats (dict):
            ``name -> {"target", "measured", "sigma", "p_bar",
            "shared_fraction"}``, plus a ``"cross_type_shared_fractions"`` entry
            giving the derived correlation for every pair of receiver types.
    """
    if not streams:
        raise ValueError("streams must be non-empty")
    for name, spec in streams.items():
        if spec["N"] > pool_size:
            raise ValueError(
                f"stream '{name}' wants {spec['N']} afferents from a pool of "
                f"{pool_size} axons; the pool must be at least as large as the "
                "largest afferent count."
            )

    rate_is_series = np.ndim(rate) > 0
    rate_arr = np.asarray(rate, dtype=np.float64) if rate_is_series else None
    if rate_is_series and rate_arr.shape[0] != num_bins:
        raise ValueError(
            f"rate time series must have length num_bins "
            f"(got {rate_arr.shape[0]} vs {num_bins})"
        )

    memmaps = {
        name: np.memmap(
            spec["filename"],
            dtype=receiver_dtype,
            mode="w+",
            shape=(spec["R"], num_bins),
        )
        for name, spec in streams.items()
    }
    r_total = sum(spec["R"] for spec in streams.values())
    chunk_size = _chunk_size_for(r_total, num_bins)
    if verbose:
        listed = ", ".join(
            "{} (N={})".format(name, spec["N"]) for name, spec in streams.items()
        )
        print(
            f"  axon pool of {pool_size} axons shared by {listed}, "
            f"chunks of {chunk_size} bins"
        )

    p_sum = 0.0
    sigma = 0.0
    carry = None
    samples = {name: None for name in streams}
    sample_p_bar = None
    for start in tqdm(range(0, num_bins, chunk_size), disable=not verbose):
        end = min(start + chunk_size, num_bins)
        n = end - start
        rate_chunk = rate_arr[start:end] if rate_is_series else rate
        p_t, sigma, carry = make_global_p_trace(
            rate=rate_chunk,
            dt=dt,
            num_bins=n,
            rng=rng,
            r_sc=r_sc,
            tau_c_ms=tau_c_ms,
            t_meas_ms=t_meas_ms,
            carry=carry,
        )
        p_sum += float(p_t.sum())

        # which axons of the shared pool fire, this bin
        k = rng.binomial(pool_size, p_t)
        for name, spec in streams.items():
            n_eff = spec["N"]
            counts = rng.hypergeometric(
                n_eff, pool_size - n_eff, np.broadcast_to(k, (spec["R"], n))
            )
            memmaps[name][:, start:end] = counts.astype(receiver_dtype, copy=False)
            if samples[name] is None:
                samples[name] = counts[:, : min(n, _SAMPLE_BINS)].copy()
        if sample_p_bar is None:
            sample_p_bar = float(p_t[: min(n, _SAMPLE_BINS)].mean())

    for mm in memmaps.values():
        mm.flush()
    memmaps.clear()

    out = {}
    for name, spec in streams.items():
        f_within = spec["N"] / pool_size
        target = stream_target_statistics(
            n_eff=spec["N"],
            p_bar=sample_p_bar,
            shared_fraction=f_within,
            sigma=sigma,
        )
        out[name] = {
            "target": target,
            "measured": measure_stream_statistics(samples[name]),
            "sigma": sigma,
            "p_bar": p_sum / num_bins,
            "shared_fraction": f_within,
            "sample_shape": tuple(samples[name].shape),
        }
    names = list(streams)
    out["cross_type_shared_fractions"] = {
        f"{a}|{b}": math.sqrt(streams[a]["N"] * streams[b]["N"]) / pool_size
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    }
    out["pool_size"] = pool_size
    return out


# ---------------------------------------------------------------------------
# Distance-dependent shared fraction: geometric source pools
# ---------------------------------------------------------------------------


@dataclass
class GeometricSourcePools:
    """Which virtual source neurons feed which receiver.

    Built by connecting each receiver to a common cloud of source neurons with
    the model's own distance kernel, so the shared fraction between two
    receivers is the overlap of their pools and therefore reproduces ``f(d)``
    without it ever being computed. ``multiplicity`` lets one virtual source
    stand for ``k`` real neurons, which keeps the cloud small; a receiver's
    afferent count is ``degree * multiplicity``.
    """

    src_indptr: np.ndarray
    src_receivers: np.ndarray
    n_sources: int
    multiplicity: int
    degrees: np.ndarray
    n_receivers: int

    @property
    def mean_n_eff(self) -> float:
        return float(self.degrees.mean() * self.multiplicity)

    def realised_shared_fractions(self) -> np.ndarray:
        """The ``(R, R)`` shared-fraction matrix the construction actually gives.

        Only for validation and reporting -- the simulation never needs it.
        """
        source_of_entry = np.repeat(
            np.arange(self.n_sources), np.diff(self.src_indptr)
        )
        a = sp.coo_matrix(
            (
                np.ones(self.src_receivers.shape[0], dtype=np.float32),
                (self.src_receivers, source_of_entry),
            ),
            shape=(self.n_receivers, self.n_sources),
        ).tocsr()
        overlap = np.asarray((a @ a.T).todense(), dtype=np.float64)
        overlap *= self.multiplicity
        # The correlation of two Binomial counts sharing `n` of N_i and N_j
        # afferents is n / sqrt(N_i N_j), so the normaliser is the geometric mean
        # of the two degrees, not either one of them. Degrees vary by ~8 % across
        # receivers, so this is not quite the same as dividing by the row degree.
        deg = np.maximum(self.degrees * self.multiplicity, 1e-12)
        return overlap / np.sqrt(np.outer(deg, deg))


def build_geometric_source_pools(
    receiver_positions: np.ndarray,
    p_func: Callable[[np.ndarray], np.ndarray],
    r_in: float,
    r_out: float,
    density_pre: float,
    rng: np.random.Generator,
    multiplicity: int = 10,
    verbose: bool = False,
) -> GeometricSourcePools:
    """Realise the unsimulated presynaptic pool as actual points in space.

    Sources are scattered at ``density_pre / multiplicity`` through a box that
    extends ``r_out`` beyond the receiver bounding box in every direction, so no
    receiver sees an edge. Each receiver connects to each source at distance
    ``d`` in ``[r_in, r_out]`` with probability ``p_func(d)`` -- the same kernel
    the simulated connectivity uses. The expected degree therefore reproduces
    ``E_outer`` and the expected overlap reproduces ``E_shared(d)``, with no
    integral evaluated anywhere.

    Args:
        receiver_positions (np.ndarray):
            ``(R, 3)`` positions in mm.

        p_func (callable):
            Connection probability as a function of distance in mm. Must accept
            an array.

        r_in (float):
            Inner radius in mm. Sources closer than this are already simulated.

        r_out (float):
            Outer cutoff in mm.

        density_pre (float):
            Density of the presynaptic cell type in neurons per mm^3.

        rng (np.random.Generator):
            Random source.

        multiplicity (int):
            How many real neurons one virtual source stands for.

        verbose (bool):
            Print the realised degree against expectation.

    Returns:
        pools (GeometricSourcePools):
            Source-major connectivity.
    """
    positions = np.asarray(receiver_positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"receiver_positions must be (R, 3), got {positions.shape}"
        )
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")
    if r_out <= r_in:
        raise ValueError(f"r_out ({r_out}) must exceed r_in ({r_in})")

    lo = positions.min(axis=0) - r_out
    hi = positions.max(axis=0) + r_out
    volume = float(np.prod(hi - lo))
    n_sources = int(round(density_pre * volume / multiplicity))
    if n_sources < 1:
        raise ValueError(
            f"geometric pool would hold {n_sources} sources; lower multiplicity "
            f"(currently {multiplicity}) or check density_pre={density_pre}"
        )

    sources = rng.uniform(lo, hi, size=(n_sources, 3))
    tree = cKDTree(sources)

    rows = []
    cols = []
    for i in range(positions.shape[0]):
        idx = np.fromiter(
            tree.query_ball_point(positions[i], r_out), dtype=np.int64
        )
        if idx.size == 0:
            continue
        d = np.linalg.norm(sources[idx] - positions[i], axis=1)
        accept = (d >= r_in) & (rng.random(idx.size) < p_func(d))
        keep = idx[accept]
        if keep.size:
            rows.append(np.full(keep.size, i, dtype=np.int32))
            cols.append(keep.astype(np.int32))

    if not rows:
        raise ValueError(
            "geometric pool produced no connections at all; check p_func, "
            f"r_in={r_in}, r_out={r_out}"
        )
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)

    # store source-major: source -> the receivers it feeds
    order = np.argsort(cols, kind="stable")
    src_receivers = rows[order].astype(np.int32)
    counts = np.bincount(cols, minlength=n_sources)
    src_indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    degrees = np.bincount(rows, minlength=positions.shape[0]).astype(np.int64)

    pools = GeometricSourcePools(
        src_indptr=src_indptr,
        src_receivers=src_receivers,
        n_sources=n_sources,
        multiplicity=multiplicity,
        degrees=degrees,
        n_receivers=positions.shape[0],
    )
    if verbose:
        print(
            f"  geometric pool: {n_sources} sources x{multiplicity}, "
            f"degree {pools.mean_n_eff:.0f} +- "
            f"{degrees.std() * multiplicity:.0f} afferents per receiver"
        )
    return pools


def simulate_receiver_counts_geometric_to_memmap(
    filename: str,
    pools: GeometricSourcePools,
    rate: Union[float, np.ndarray],
    dt: float,
    num_bins: int,
    receiver_dtype: np.dtype,
    rng: np.random.Generator,
    r_sc: float = 0.0,
    tau_c_ms: float = 0.0,
    t_meas_ms: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, float]:
    """Stream counts from an explicit source pool, straight to a memmap.

    Per bin, draw how many of the ``n_sources * multiplicity`` real neurons
    spiked, scatter those spikes uniformly over the sources, and add each one to
    every receiver that source feeds. Marginals are exactly Binomial and the
    receiver correlation is exactly the pool overlap, both by construction.

    Args:
        filename (str):
            Destination memmap path, shape ``(R, num_bins)``.

        pools (GeometricSourcePools):
            From ``build_geometric_source_pools``.

        rate (float or np.ndarray):
            Presynaptic rate in Hz.

        dt (float):
            Timestep in ms.

        num_bins (int):
            Length of the stream.

        receiver_dtype (np.dtype):
            Storage dtype.

        rng (np.random.Generator):
            Random source.

        r_sc, tau_c_ms, t_meas_ms:
            Shared rate modulation, see ``make_global_p_trace``.

        verbose (bool):
            Print chunking information.

    Returns:
        stats (dict):
            ``target`` and ``measured`` statistics, ``sigma`` and the realised
            mean shared fraction.
    """
    R = pools.n_receivers
    k = pools.multiplicity
    n_neurons = pools.n_sources * k

    rate_is_series = np.ndim(rate) > 0
    rate_arr = np.asarray(rate, dtype=np.float64) if rate_is_series else None
    if rate_is_series and rate_arr.shape[0] != num_bins:
        raise ValueError(
            f"rate time series must have length num_bins "
            f"(got {rate_arr.shape[0]} vs {num_bins})"
        )

    mm = np.memmap(filename, dtype=receiver_dtype, mode="w+", shape=(R, num_bins))
    chunk_size = _chunk_size_for(R, num_bins, target_bytes=32 * 1024 * 1024)
    if verbose:
        print(
            f"  geometric: {pools.n_sources} sources x{k}, R={R}, "
            f"chunks of {chunk_size} bins"
        )

    src_deg = np.diff(pools.src_indptr)
    p_sum = 0.0
    sigma = 0.0
    carry = None
    sample_block = None
    sample_p_bar = None
    for start in tqdm(range(0, num_bins, chunk_size), disable=not verbose):
        end = min(start + chunk_size, num_bins)
        n = end - start
        rate_chunk = rate_arr[start:end] if rate_is_series else rate
        p_t, sigma, carry = make_global_p_trace(
            rate=rate_chunk,
            dt=dt,
            num_bins=n,
            rng=rng,
            r_sc=r_sc,
            tau_c_ms=tau_c_ms,
            t_meas_ms=t_meas_ms,
            carry=carry,
        )
        p_sum += float(p_t.sum())

        n_events = rng.binomial(n_neurons, p_t)
        total = int(n_events.sum())
        block = np.zeros((R, n), dtype=np.int32)
        if total:
            src = rng.integers(0, pools.n_sources, size=total)
            bin_of_event = np.repeat(np.arange(n, dtype=np.int64), n_events)
            deg = src_deg[src]
            live = deg > 0
            src, bin_of_event, deg = src[live], bin_of_event[live], deg[live]
            if src.size:
                total_pairs = int(deg.sum())
                offsets = np.arange(total_pairs) - np.repeat(
                    np.cumsum(deg) - deg, deg
                )
                rec = pools.src_receivers[
                    np.repeat(pools.src_indptr[src], deg) + offsets
                ]
                bins_expanded = np.repeat(bin_of_event, deg)
                flat = rec.astype(np.int64) * n + bins_expanded
                block = np.bincount(flat, minlength=R * n).reshape(R, n).astype(
                    np.int32
                )
        mm[:, start:end] = block.astype(receiver_dtype, copy=False)
        if sample_block is None:
            n_sample = min(n, _SAMPLE_BINS)
            sample_block = block[:, :n_sample].copy()
            sample_p_bar = float(p_t[:n_sample].mean())

    mm.flush()
    del mm

    p_bar = p_sum / num_bins
    f_matrix = pools.realised_shared_fractions()
    iu = np.triu_indices(R, 1)
    mean_f = float(f_matrix[iu].mean()) if R > 1 else 0.0

    target = stream_target_statistics(
        n_eff=pools.mean_n_eff,
        p_bar=sample_p_bar,
        shared_fraction=mean_f,
        sigma=sigma,
    )
    measured = measure_stream_statistics(sample_block)
    return {
        "target": target,
        "measured": measured,
        "sigma": sigma,
        "p_bar": p_bar,
        "mean_shared_fraction": mean_f,
        "mean_n_eff": pools.mean_n_eff,
        "sample_shape": tuple(sample_block.shape),
    }


# ---------------------------------------------------------------------------
# Reading back
# ---------------------------------------------------------------------------


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
            f"Reading memmap file {filename}; per-chunk memory "
            f"~{R * chunk_size * np.dtype(receiver_dtype).itemsize / (1024 * 1024):.2f} MB."
        )
    mm = np.memmap(filename, dtype=receiver_dtype, mode="r", shape=(R, num_bins))
    for start in range(0, num_bins, chunk_size):
        end = min(start + chunk_size, num_bins)
        chunk = mm[:, start:end]
        yield chunk.copy() if copy else chunk


# ---------------------------------------------------------------------------
# Cortical proportions
# ---------------------------------------------------------------------------


def validate_cortical_proportions(
    cortical_proportions_dict: Optional[Dict[str, float]], owner: str
) -> Dict[str, float]:
    """Check a per-region cortical proportion mapping and return it unchanged.

    There is deliberately no default. The proportions are the only physical
    difference between the caudate and the putamen loop, and the same numbers
    also weight the mix that produces the ``caudate_rate``/``putamen_rate``
    series in the cortical rate ``.npz``. A default here would be a second copy
    that could silently disagree with the one the rate file was built from, so
    the caller has to say which proportions it means. In BGM_22 they live in
    ``BOLD_optimization/parameters.py`` under ``cortical_proportions_dict``.

    Args:
        cortical_proportions_dict (dict):
            Region name -> share of a receiver's cortical afferents. Shares must
            be non-negative and sum to 1; a region at 0 gets no stream at all.

        owner (str):
            Name of the calling class, used in the error messages.

    Returns:
        cortical_proportions_dict (dict):
            The validated mapping.
    """
    if cortical_proportions_dict is None:
        raise ValueError(
            f"{owner} requires cortical_proportions_dict; there is no default. "
            "Pass the mapping for this loop -- in BGM_22 it is "
            "parameters.py['cortical_proportions_dict'][loop]. Note that the "
            "cortical rate .npz was mixed with these same numbers, so changing "
            "them means regenerating it."
        )
    if not isinstance(cortical_proportions_dict, dict) or not cortical_proportions_dict:
        raise ValueError(
            f"{owner}: cortical_proportions_dict must be a non-empty dict, got "
            f"{cortical_proportions_dict!r}."
        )
    negative = {k: v for k, v in cortical_proportions_dict.items() if v < 0}
    if negative:
        raise ValueError(
            f"{owner}: cortical proportions must be non-negative, got {negative}."
        )
    total = float(sum(cortical_proportions_dict.values()))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{owner}: cortical proportions must sum to 1, got {total!r} for "
            f"{cortical_proportions_dict}. They split a fixed number of cortical "
            "afferents, so a sum below or above 1 silently rescales the drive."
        )
    return cortical_proportions_dict


__all__ = [
    "ou_sum_variance",
    "solve_modulation_amplitude",
    "make_global_p_trace",
    "stream_target_statistics",
    "measure_stream_statistics",
    "check_stream_statistics",
    "simulate_receiver_counts_homogeneous_to_memmap",
    "axon_pool_size",
    "simulate_cortical_axon_pool_streams_to_memmap",
    "GeometricSourcePools",
    "build_geometric_source_pools",
    "simulate_receiver_counts_geometric_to_memmap",
    "iter_memmap_spike_counts",
    "validate_cortical_proportions",
]
