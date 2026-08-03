"""Cortical input generation and streaming for existing ANNarchy populations.

This helper mirrors the cortical-input path in ``Microcircuit`` but strips away
all connectivity and local inhibitory input generation. It loads the BOLD-derived
rates, simulates spike-count time series without shared input, stores them on
memmap files, and exposes ``create_model``/``update`` to feed the provided
populations via ANNarchy ``TimedArray``/``CurrentInjection`` objects.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from ANNarchy import CurrentInjection, Population, TimedArray, simulate

# import from the local spike_input_cortex module the functions iter_memmap_spike_counts and
# simulate_receiver_counts_homogeneous_to_memmap:
from CompNeuroPy.striatal_microcircuit.spike_input_cortex import (
    iter_memmap_spike_counts,
    simulate_receiver_counts_homogeneous_to_memmap,
)


class CorticalInputs:
    """Generate and stream cortical spike inputs for pre-existing populations.

    Args:
        populations: Iterable of ANNarchy ``Population`` objects that should receive
            cortical input. Their ``name`` must match the keys in
            ``N_cortical_inputs_dict`` (case-insensitive).
        N_cortical_inputs_dict: Expected number of cortical input neurons per receiver
            neuron for each population type (e.g., {"dSPN": 7000, "iSPN": 7000}).
        dt: Simulation timestep in ms.
        update_time: Time window (ms) covered by each update chunk.
        T: Total simulation time in ms.
        name: Striatal region label, ``"caudate"`` or ``"putamen"``.
        dbs_condition: DBS condition string used to pick the rate file (``"on"``/``"off"``).
        storage_dir: Optional directory for memmaps and state; defaults to a hidden
            folder next to this module.
        build_cortical_input: When False, skip simulation and load existing memmaps/state.
        seed: RNG seed for reproducible spike sampling.
        verbose: Print progress information when True.
    """

    def __init__(
        self,
        populations: Iterable[Population],
        N_cortical_inputs_dict: Dict[str, int],
        dt: float,
        update_time: float,
        T: float,
        name: str,
        dbs_condition: str = "on",
        cortical_rate_path: Optional[str] = None,
        storage_dir: Optional[str] = None,
        build_cortical_input: bool = True,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.name = name
        if self.name not in ("caudate", "putamen"):
            raise ValueError("name must be either 'caudate' or 'putamen'")
        if dbs_condition not in {"on", "off"}:
            raise ValueError("dbs_condition must be 'on' or 'off'")
        self.dbs_condition = dbs_condition

        self.cortical_rate_path = (
            Path(cortical_rate_path).expanduser()
            if cortical_rate_path is not None
            else Path(__file__).resolve().parent
            / "external_input"
            / "results_cortical_drive_by_bold"
            / f"firing_rates_matlab_condition-{self.dbs_condition}.npz"
        )

        self.dt = float(dt)
        self.update_time = float(update_time)
        self.T = float(T)
        if self.dt <= 0 or self.update_time <= 0 or self.T <= 0:
            raise ValueError("dt, update_time, and T must be positive")
        self.n_steps = int(round(self.T / self.dt))
        if not np.isclose(self.n_steps * self.dt, self.T, rtol=1e-9, atol=1e-9):
            raise ValueError("T must be an integer multiple of dt")

        self.verbose = verbose
        self.rng = np.random.default_rng(seed)

        self.cortical_proportions_dict = self._default_cortical_proportions()[self.name]
        self.N_cortical_inputs_dict = N_cortical_inputs_dict

        script_dir = os.path.dirname(__file__)
        self.storage_dir = storage_dir or os.path.join(
            script_dir, f".cortical_inputs_{self.name}_{self.dbs_condition}"
        )
        self.inputs_dir = os.path.join(self.storage_dir, "inputs")
        os.makedirs(self.inputs_dir, exist_ok=True)

        # Map population type labels to ANNarchy populations
        self.populations: Dict[str, Population] = {}
        for pop in populations:
            pop_type = self._infer_type(pop.name)
            if pop_type in self.populations:
                raise ValueError(f"Duplicate population type '{pop_type}' provided")
            self.populations[pop_type] = pop
        self.cell_types = set(self.populations.keys())

        self.cor_input_memmap_dict: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.annarchy_inp_populations: Dict[Tuple[str, str], TimedArray] = {}
        self.annarchy_inp_projections: Dict[Tuple[str, str], CurrentInjection] = {}
        self.inp_iterator_dict: Dict[Tuple[str, str], object] = {}
        self.model_created = False

        # prepare dictionary to store the weights of inputs, fill with default values
        self.mean_weights_by_type: Dict[Tuple[str, str], float] = {}
        for post_type in self.cell_types:
            for cortical_region in self.cortical_proportions_dict.keys():
                key = (cortical_region, post_type)
                # default mean weight for cortical inputs, can be scaled later
                self.mean_weights_by_type[key] = 0.001

        if build_cortical_input:
            self._simulate_cortical_spike_counts()
            self._save_cortical_input_state()
        else:
            self._load_cortical_input_state()

    # ----------------------
    # Public API
    # ----------------------
    def create_model(self) -> None:
        """Create ANNarchy TimedArray inputs and CurrentInjection projections."""
        if self.model_created:
            raise RuntimeError("create_model() can only be called once")
        if not self.cor_input_memmap_dict:
            raise RuntimeError("No cortical spike-count memmaps available")

        n_steps_input = int(self.update_time / self.dt)
        if n_steps_input <= 0:
            raise ValueError("update_time must be at least one dt long")

        for key, memmap_info in self.cor_input_memmap_dict.items():
            pre_region, post_type = key
            post_pop = self.populations.get(post_type)
            if post_pop is None:
                continue

            inp = TimedArray(
                rates=np.zeros((n_steps_input, post_pop.size)),
                name=f"TimedInput_{pre_region}_{post_type}_{self.name}",
            )
            proj = CurrentInjection(
                pre=inp,
                post=post_pop,
                target="ampa",
                name=f"CurrentInjection_{pre_region}_{post_type}_{self.name}",
            )
            proj.connect_current()

            spike_file = self._spike_counts_path(pre_region, post_type)
            inp_iterator = iter_memmap_spike_counts(
                filename=spike_file,
                R=memmap_info["R"],
                num_bins=self.n_steps,
                receiver_dtype=memmap_info["receiver_dtype"],
                chunk_size=n_steps_input,
                copy=False,
                verbose=self.verbose,
            )
            self.annarchy_inp_populations[key] = inp
            self.annarchy_inp_projections[key] = proj
            self.inp_iterator_dict[key] = inp_iterator

        self.model_created = True

    def get_model_component_names(self) -> Dict[str, list[str]]:
        """Return names of TimedArrays and projections created by ``create_model``."""
        if not self.model_created:
            raise RuntimeError(
                "create_model() must be called before accessing component names."
            )

        population_names = [inp.name for inp in self.annarchy_inp_populations.values()]
        projection_names = [
            proj.name for proj in self.annarchy_inp_projections.values()
        ]

        return {
            "populations": sorted(population_names),
            "projections": sorted(projection_names),
        }

    def update(self, run_simulation: bool = False) -> None:
        """Advance cortical input streams by one update window."""
        if not self.model_created:
            raise RuntimeError("create_model() must be called before update()")

        for key, inp_iterator in self.inp_iterator_dict.items():
            inp_population = self.annarchy_inp_populations[key]
            inputs = next(inp_iterator).T  # (n_steps, n_neurons)

            # rewind the internal timers so the new chunk is played from its first block
            inp_population.update(
                rates=inputs * self.mean_weights_by_type[key], reset=True
            )

        if run_simulation:
            simulate(self.update_time)

    def reset(self) -> None:
        """Reset cortical input iterators so the next ``update`` starts from the beginning."""
        if not self.model_created:
            raise RuntimeError("create_model() must be called before reset()")

        n_steps_input = int(self.update_time / self.dt)
        if n_steps_input <= 0:
            raise ValueError("update_time must be at least one dt long")

        self.inp_iterator_dict = {}
        for key in self.annarchy_inp_populations.keys():
            memmap_info = self.cor_input_memmap_dict.get(key)
            if memmap_info is None:
                continue

            spike_file = self._spike_counts_path(*key)
            self.inp_iterator_dict[key] = iter_memmap_spike_counts(
                filename=spike_file,
                R=memmap_info["R"],
                num_bins=self.n_steps,
                receiver_dtype=memmap_info["receiver_dtype"],
                chunk_size=n_steps_input,
                copy=False,
                verbose=self.verbose,
            )

    # ----------------------
    # Internal helpers
    # ----------------------
    def _infer_type(self, pop_name: str) -> str:
        """Infer the logical type label from a population name."""
        candidates = {k.lower(): k for k in self.N_cortical_inputs_dict.keys()}
        key = pop_name.lower()
        if key not in candidates:
            raise ValueError(
                f"Population name '{pop_name}' does not match any known type keys {list(self.N_cortical_inputs_dict.keys())}"
            )
        return candidates[key]

    def _spike_counts_path(self, pre_type: str, post_type: str) -> str:
        return os.path.join(
            self.inputs_dir, f"receiver_counts_{pre_type}_{post_type}.dat"
        )

    def _cortical_input_state_path(self) -> str:
        return os.path.join(self.inputs_dir, "cortical_input_state.pkl")

    def _default_cortical_proportions(self) -> Dict[str, Dict[str, float]]:
        return {
            "caudate": {
                "dlPFC": 0.45,
                "preSMA": 0.25,
                "PMd": 0.15,
                "PMv": 0.10,
                "SMA": 0.04,
                "M1": 0.01,
                "S1": 0.00,
            },
            "putamen": {
                "dlPFC": 0.05,
                "preSMA": 0.10,
                "PMd": 0.15,
                "PMv": 0.05,
                "SMA": 0.25,
                "M1": 0.30,
                "S1": 0.10,
            },
        }

    def _save_cortical_input_state(self) -> None:
        if not self.cor_input_memmap_dict:
            return

        payload = {
            "cor_input_memmap_dict": self.cor_input_memmap_dict,
            "rng_state": self.rng.bit_generator.state,
            "dbs_condition": self.dbs_condition,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "cortical_proportions_dict": self.cortical_proportions_dict,
            "N_cortical_inputs_dict": self.N_cortical_inputs_dict,
            "name": self.name,
            "cortical_rate_path": str(self.cortical_rate_path),
            "keys": [f"{pre}-{post}" for (pre, post) in self.cor_input_memmap_dict],
        }

        with open(self._cortical_input_state_path(), "wb") as f:
            pickle.dump(payload, f)

    def _load_cortical_input_state(self) -> None:
        state_path = self._cortical_input_state_path()
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Cortical input cache not found at {state_path}. Rebuild by setting build_cortical_input=True."
            )

        with open(state_path, "rb") as f:
            payload = pickle.load(f)

        if payload.get("name") != self.name:
            raise ValueError(
                "Cached cortical-input state uses different region name; rebuild cortical inputs."
            )

        dt_saved = payload.get("dt", self.dt)
        if not np.isclose(dt_saved, self.dt, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"Cached cortical-input dt ({dt_saved}) does not match current dt ({self.dt}); rebuild cortical inputs."
            )

        n_steps_saved = payload.get("n_steps", self.n_steps)
        if n_steps_saved != self.n_steps:
            raise ValueError(
                f"Cached cortical-input n_steps ({n_steps_saved}) does not match current n_steps ({self.n_steps}); rebuild cortical inputs."
            )

        cond_saved = payload.get("dbs_condition", self.dbs_condition)
        if cond_saved != self.dbs_condition:
            raise ValueError(
                "Cached cortical-input state was built for a different dbs_condition; rebuild cortical inputs."
            )

        rate_path_saved = payload.get("cortical_rate_path")
        if rate_path_saved is not None:
            current_rate_path = str(Path(self.cortical_rate_path))
            if rate_path_saved != current_rate_path:
                raise ValueError(
                    "Cached cortical-input state uses different cortical_rate_path; rebuild cortical inputs."
                )

        proportions_saved = payload.get("cortical_proportions_dict")
        if proportions_saved and proportions_saved != self.cortical_proportions_dict:
            raise ValueError(
                "Cached cortical-input state uses different cortical_proportions_dict; rebuild cortical inputs."
            )

        N_inputs_saved = payload.get("N_cortical_inputs_dict")
        if N_inputs_saved and N_inputs_saved != self.N_cortical_inputs_dict:
            raise ValueError(
                "Cached cortical-input state uses different N_cortical_inputs_dict; rebuild cortical inputs."
            )

        cor_memmap = payload.get("cor_input_memmap_dict")
        if cor_memmap is None:
            raise ValueError(
                "Cached cortical-input memmap info dict is empty; rebuild cortical inputs."
            )

        saved_keys = set(tuple(k.split("-")) for k in payload.get("keys", []))
        if saved_keys and set(cor_memmap.keys()) != saved_keys:
            raise ValueError(
                "Cached cortical-input state keys mismatch stored state; rebuild cortical inputs."
            )

        for pre_type, post_type in cor_memmap.keys():
            path = self._spike_counts_path(pre_type, post_type)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Spike-count file for {pre_type}->{post_type} not found at {path}; rebuild cortical inputs."
                )

        self.cor_input_memmap_dict = cor_memmap

        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _simulate_cortical_spike_counts(self) -> None:
        rate_path = Path(self.cortical_rate_path)
        if not rate_path.exists():
            raise FileNotFoundError(
                f"Cortical drive rates not found at {rate_path}; run cortical_drive_by_bold.py first."
            )

        with np.load(rate_path) as data:
            # pick any time array to infer dt
            sample_time_key = None
            for cortical_region in self.cortical_proportions_dict:
                key_candidate = f"{cortical_region}_time"
                if key_candidate in data:
                    sample_time_key = key_candidate
                    break
            if sample_time_key is None:
                raise ValueError(
                    "No cortical time arrays found in cortical drive file."
                )

            time_arr = np.asarray(data[sample_time_key])
            if time_arr.size < 2:
                raise ValueError(
                    f"Time array '{sample_time_key}' too short to determine dt."
                )
            dt_seconds = float(time_arr[1] - time_arr[0])
            dt_ms = dt_seconds * 1000.0
            if not np.isclose(dt_ms, self.dt, rtol=1e-6, atol=1e-9):
                raise ValueError(
                    f"Cortical drive dt ({dt_ms:.6f} ms) does not match requested dt ({self.dt:.6f} ms)."
                )

            for post_type, post_pop in self.populations.items():
                N_total = self.N_cortical_inputs_dict.get(post_type)
                if N_total is None or N_total <= 0:
                    if self.verbose:
                        print(
                            f"Skipping {post_type}: no expected cortical inputs specified."
                        )
                    continue

                for (
                    cortical_region,
                    proportion,
                ) in self.cortical_proportions_dict.items():
                    rate_key = f"{cortical_region}_rate"
                    if rate_key not in data:
                        raise KeyError(
                            f"Rate key '{rate_key}' missing in cortical drive file {rate_path}."
                        )
                    rate_series = np.asarray(data[rate_key])
                    if rate_series.size < self.n_steps:
                        raise ValueError(
                            f"Rate series for {cortical_region} has only {rate_series.size} samples; expected at least {self.n_steps}."
                        )
                    rate_segment = rate_series[: self.n_steps]

                    N_eff = int(np.round(proportion * N_total))
                    if N_eff == 0:
                        continue

                    spike_file = self._spike_counts_path(cortical_region, post_type)
                    simulate_receiver_counts_homogeneous_to_memmap(
                        filename=spike_file,
                        R=post_pop.size,
                        N=N_eff,
                        shared_input=0.0,  # no shared fraction
                        rate=rate_segment,
                        dt=self.dt,
                        rho=0.0,
                        num_bins=self.n_steps,
                        receiver_dtype=np.float64,
                        rng=self.rng,
                        verbose=self.verbose,
                    )

                    self.cor_input_memmap_dict[(cortical_region, post_type)] = {
                        "R": post_pop.size,
                        "receiver_dtype": np.float64,
                    }
