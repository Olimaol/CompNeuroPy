import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.sparse import lil_matrix, load_npz, save_npz

# ANNarchy imports
from ANNarchy import (
    CurrentInjection,
    Population,
    Projection,
    TimedArray,
    simulate,
)

# Local imports
from CompNeuroPy.striatal_microcircuit.spike_input_cortex import (
    simulate_receiver_counts_homogeneous_to_memmap,
    simulate_receiver_counts_distance_dependent_to_memmap,
    iter_memmap_spike_counts,
)
from CompNeuroPy.striatal_microcircuit.get_weights import (
    components_spn_spn,
    components_fsi_spn,
    components_fsi_fsi,
)

# CompNeuroPy imports
from CompNeuroPy.extra_functions import CombinedSampler
from CompNeuroPy.neuron_models import (
    Izhikevich2007Humphries2009SPND1,
    Izhikevich2007Humphries2009SPND2,
    Izhikevich2007Humphries2009FSI,
)


class Microcircuit:
    """
    Microcircuit class that constructs a 3D periodic microcircuit with distance-dependent
    connection probabilities and provides analysis/visualization methods.

    Key attributes after initialization:
    - weights_by_type: dict[(pre_type, post_type)] -> sparse lil_matrix with shape
        (n_pre_type_cells, n_post_type_cells); indices local to each population.
    - adj: dict[pre_type][post_type] -> list of (i_pre, j_post) index pairs
    - con_probs: dict[(pre_type, post_type)] -> list of tuples (p, connected_flag)
    - positions: (n_total, 3) array of neuron positions in mm
    - types: (n_total,) array of neuron type strings
    - cell_types: list of type labels
    - output_dir: path where plots are saved
    """

    # ----------------------
    # Storage helpers
    # ----------------------
    def _connectivity_state_path(self) -> str:
        return os.path.join(self.connectivity_dir, "connectivity_state.pkl")

    def _weight_matrix_path(self, pre_type: str, post_type: str) -> str:
        return os.path.join(
            self.connectivity_dir, f"weights_{pre_type}_{post_type}.npz"
        )

    def _missing_input_state_path(self) -> str:
        return os.path.join(self.inputs_dir, "missing_input_state.pkl")

    def _cortical_input_state_path(self) -> str:
        return os.path.join(self.inputs_dir, "cortical_input_state.pkl")

    def _spike_counts_path(self, pre_type: str, post_type: str) -> str:
        return os.path.join(
            self.inputs_dir,
            f"receiver_counts_{pre_type}_{post_type}.dat",  # TODO remove _distance_dependent
        )

    def __init__(
        self,
        name: str = "caudate",
        nx: int = 10,
        b: int = 10,
        density: float = 84900.0,
        firing_rate_dict: dict | None = None,
        correlation_dict: dict | None = None,
        N_cortical_inputs_dict: dict | None = None,
        cortical_proportions_dict: dict | None = None,
        cortical_rate_path: str | Path | None = None,
        dt: float = 0.1,
        T: float = 1000.0,
        update_time: float = 100.0,
        seed: int = 42,
        props_delRey: np.ndarray | None = None,
        fitted_params_path: str | None = None,
        storage_dir: str | None = None,
        output_dir: str | None = None,
        build_connectivity: bool = True,
        build_missing_gaba_input: bool = True,
        build_cortical_input: bool = True,
        verbose: bool = True,
        dbs_condition: str = "on",
    ) -> None:
        # --- Parameters ---
        self.debug = False
        self.update_time = update_time  # ms for how long inputs are defined
        self.name = name
        self.nx = nx
        self.b = b
        self.density = density
        self.seed = seed
        self.verbose = verbose
        if dbs_condition not in {"on", "off"}:
            raise ValueError("dbs_condition must be 'on' or 'off'")
        self.dbs_condition = dbs_condition

        # name should be either 'caudate' or 'putamen'
        if self.name not in ("caudate", "putamen"):
            raise ValueError("name must be either 'caudate' or 'putamen'")

        # firing rates per cell type (Hz)
        # default for D1 and D2 extracted from: (Liang et al., 2008) using with levodopa treatment, see experimental_data/activity_striatum/extract_from_liang_etal_2008.py
        # default for FS: 10 Hz based on: (Yamada et al., 2016; Marche und Apicella, 2021; Adler et al., 2013; Hernandez et al., 2013; He et al., 2024)
        if firing_rate_dict is None:
            firing_rate_dict = {"FS": 10.0, "dSPN": 37.07, "iSPN": 29.07}
        self.firing_rate_dict = firing_rate_dict

        # average correlations between pairs of cell types
        if correlation_dict is None:
            # default based on (Adler et al., 2013):
            correlation_dict = {"FS": 0.06, "dSPN": 0.004, "iSPN": 0.004}
        self.correlation_dict = correlation_dict

        # expected number of input neurons from cortex per receiver neuron per cell type
        # default based on my calculations (see zotero/goolge/notebooks)
        if N_cortical_inputs_dict is None:
            N_cortical_inputs_dict = {"FS": 2800, "dSPN": 7000, "iSPN": 7000}
        self.N_cortical_inputs_dict = N_cortical_inputs_dict

        # the cortical proportions dict for our given cortical regions from the BOLD data:
        if cortical_proportions_dict is None:
            proportions = {
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
            cortical_proportions_dict = proportions[self.name]
        self.cortical_proportions_dict = cortical_proportions_dict

        # shared fraction of inputs between striatal neurons based on Kincaid et al., 1998
        self.shared_fraction = 0.014

        # timestep for simulation in ms
        self.dt = dt

        # total simulation time in ms and simulation steps
        self.T = T
        self.n_steps = int(T / dt)

        # proportions (del Rey et al. 2022)
        if props_delRey is None:
            props_delRey = np.array([0.026, 0.86 / 2, 0.86 / 2])
        props = props_delRey / np.sum(props_delRey)
        self.cortical_rate_path = (
            Path(cortical_rate_path).expanduser()
            if cortical_rate_path is not None
            else Path(__file__).resolve().parent
            / "external_input"
            / "results_cortical_drive_by_bold"
            / f"firing_rates_matlab_condition-{self.dbs_condition}.npz"
        )
        self.props = {"FS": props[0], "dSPN": props[1], "iSPN": props[2]}
        self.cell_types = list(self.props.keys())

        # paths
        script_dir = os.path.dirname(__file__)
        self.storage_dir = storage_dir or os.path.join(
            script_dir, f".microcircuit_{self.name}_{self.dbs_condition}"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self.connectivity_dir = os.path.join(self.storage_dir, "connectivity")
        self.inputs_dir = os.path.join(self.storage_dir, "inputs")
        os.makedirs(self.connectivity_dir, exist_ok=True)
        os.makedirs(self.inputs_dir, exist_ok=True)
        if fitted_params_path is None:
            fitted_params_path = os.path.join(
                script_dir, "connectivity_fits", "fitted_params.json"
            )
        figures_subdir = output_dir if output_dir is not None else "figures"
        figures_subdir = os.path.basename(figures_subdir)
        self.output_dir = os.path.join(self.storage_dir, figures_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        # load connectivity parameters
        with open(fitted_params_path) as f:
            fitted = json.load(f)
        self.conn_params: dict[tuple[str, str], tuple[float, float]] = {
            tuple(key.split("-")): (val["amplitude"], val["sigma_um"])
            for key, val in fitted.items()
        }

        # --- Lattice and neuron types ---
        self.n_total = self.nx * self.b * self.b
        # spacing per cell (mm)
        self.d = (1.0 / self.density) ** (1 / 3)  # mm

        # positions (mm)
        xs = np.arange(self.nx) * self.d
        ys = np.arange(self.b) * self.d
        zs = np.arange(self.b) * self.d
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z
        self.positions = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

        # RNG
        self.rng = np.random.default_rng(self.seed)

        # assign neuron types
        type_counts = {ct: int(self.props[ct] * self.n_total) for ct in self.cell_types}
        rem = self.n_total - sum(type_counts.values())
        for ct in sorted(self.cell_types, key=lambda x: self.props[x], reverse=True)[
            :rem
        ]:
            type_counts[ct] += 1
        types = np.array(
            [ct for ct, count in type_counts.items() for _ in range(count)]
        )
        self.rng.shuffle(types)
        self.types = types

        # store counts for delayed ANNarchy population creation
        self.type_counts = type_counts

        # derived geometry
        self.d_um = self.d * 1e3
        self.dim_x_um = self.nx * self.d_um
        self.dim_y_um = self.b * self.d_um
        self.dim_z_um = self.b * self.d_um
        self.volume_mm3 = self.n_total / self.density

        # periodic KDTree (original cube + 26 neighbors)
        self.L = np.array([self.nx * self.d, self.b * self.d, self.b * self.d])
        shifts = np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        )
        self.ext_positions = np.vstack(
            [self.positions + shift * self.L for shift in shifts]
        )
        self.tree = sp.cKDTree(self.ext_positions)

        # adjacency storage and default center index (center-right as before)
        self.adj = {ct: {ct2: [] for ct2 in self.cell_types} for ct in self.cell_types}
        mid_y = self.b // 2
        mid_z = self.b // 2
        self.j_center = (self.nx - 1) * self.b * self.b + mid_y * self.b + mid_z

        # containers for analysis
        self.con_probs = {
            (pre_type, post_type): []
            for pre_type in self.cell_types
            for post_type in self.cell_types
        }
        self.neighbor_sizes: list[int] = []

        # Prepare per-type index mappings (global -> local) and initialize weight matrices
        self.indices_by_type: dict[str, np.ndarray] = {
            ct: np.flatnonzero(self.types == ct) for ct in self.cell_types
        }
        self.local_index_map: dict[str, dict[int, int]] = {
            ct: {g_idx: l_idx for l_idx, g_idx in enumerate(self.indices_by_type[ct])}
            for ct in self.cell_types
        }
        # Create weight matrices only for pairs present in connectivity params for later use in ANNarchy connection
        self.weights_by_type: dict[tuple[str, str], lil_matrix] = {}
        for pre_type, post_type in self.conn_params.keys():
            n_pre = len(self.indices_by_type[pre_type])
            n_post = len(self.indices_by_type[post_type])
            self.weights_by_type[(pre_type, post_type)] = lil_matrix((n_pre, n_post))

        # prepare the samplers from which the values of the weights are sampled
        self.weight_samplers: dict[tuple[str, str], CombinedSampler] = {}
        for pre_type, post_type in self.conn_params.keys():
            if pre_type in ("dSPN", "iSPN") and post_type in ("dSPN", "iSPN"):
                components = components_spn_spn
            elif pre_type == "FS" and post_type in ("dSPN", "iSPN"):
                components = components_fsi_spn
            elif pre_type == "FS" and post_type == "FS":
                components = components_fsi_fsi
            else:
                raise ValueError(
                    f"No weight components defined for pair {pre_type}->{post_type}"
                )
            sampler = CombinedSampler(components=components, rng=self.rng)
            self.weight_samplers[(pre_type, post_type)] = sampler

        # prepare the dictionaries to hold the input iterators and corresponding TimedArray populations for each pre-post type pair and the cortical inputs (from dlPFC, PM/SMA, M1) for later use in the update function
        self.inp_iterator_dict: dict[tuple[str, str], iter] = (
            {}
        )  # key: (pre_type, post_type)
        self.annarchy_inp_populations: dict[tuple[str, str], TimedArray] = (
            {}
        )  # key: (pre_type, post_type)
        self.annarchy_inp_projections: dict[tuple[str, str], CurrentInjection] = (
            {}
        )  # key: (pre_type, post_type)

        # prepare dictionary to store the weights of eternal inputs, fill it with default values for the cortical inputs
        self.mean_weights_by_type: dict[tuple[str, str], float] = {}
        for post_type in self.cell_types:
            for cortical_region in self.cortical_proportions_dict.keys():
                key = (cortical_region, post_type)
                # default mean weight for cortical inputs, can be scaled later
                self.mean_weights_by_type[key] = 0.001

        # container for ANNarchy populations; created lazily in create_model
        self.annarchy_populations: dict[str, Population] = {}
        self.model_created: bool = False
        self.local_input_memmap_dict: dict | None = None
        self.cor_input_memmap_dict: dict | None = None

        # container to store distances (mm) for actual formed connections per pair
        self.connection_distances_by_pair: dict[tuple[str, str], list[float]] = {
            key: [] for key in self.conn_params.keys()
        }

        # get the radius for the simulated neighborhood per post type (mm) and
        # theoretical max sigma, required for _missing_local_input() and _build_connectivity()
        self.neighborhood_radii_mm, self.max_sigma_mm = (
            self._get_neighborhood_radii_mm()
        )

        # build or load connectivity and fill per-type weight matrices
        self.built_connectivity = False
        if build_connectivity:
            self._build_connectivity()
            self._save_connectivity_state()
        else:
            self._load_connectivity_state()
        self.built_connectivity = True

        if self.verbose:
            self.summary()

        # prepare or load missing local gaba inputs (spike counts) for all neurons
        if build_missing_gaba_input:
            self._missing_local_input()
            self._save_missing_input_state()
        else:
            self._load_missing_input_state()

        # define excitatory inputs (spike counts) for all neurons
        if build_cortical_input:
            self._excitatory_inputs()
            self._save_cortical_input_state()
        else:
            self._load_cortical_input_state()

    def update(self, run_simulation: bool = False) -> None:
        """Update function to be called during simulation to update the input populations."""
        if not self.model_created:
            raise RuntimeError(
                "create_model() must be called before update to build ANNarchy objects."
            )
        fs_debug_cache: dict[str, dict[str, np.ndarray]] | None = (
            {} if self.verbose and self.debug else None
        )
        fs_scaling_factors = None
        if self.verbose and self.debug:
            fs_scaling_factors = self._compute_fs_scaling_factors()
        dSPN_inputs_sum = []
        # Loop over all input iterators and update the corresponding TimedArray populations
        for key, inp_iterator in self.inp_iterator_dict.items():
            inp_population = self.annarchy_inp_populations[key]
            # get next chunk of inputs (incoming spike counts)
            inputs = next(inp_iterator)
            # reshape inputs from (n_neurons, n_steps) into (n_steps, n_neurons)
            inputs = inputs.T

            if self.verbose and self.debug and key[0] == "dlPFC" and key[1] == "dSPN":
                # plot the inputs as raster plot with time on x-axis and neuron index on y-axis
                plt.figure(figsize=(12, 6))

                # We transpose the data (.T) so that:
                #   Rows (Y-axis) = Neurons
                #   Columns (X-axis) = Time steps
                # origin='lower' ensures Neuron 0 is at the bottom.
                plt.imshow(
                    inputs.T * self.mean_weights_by_type[key],
                    aspect="auto",
                    cmap="viridis",
                    origin="lower",
                    interpolation="nearest",
                )
                # neuron with idx zero should be at the top
                plt.gca().invert_yaxis()

                plt.colorbar(label="Input Count")
                plt.xlabel("Time (steps)")
                plt.ylabel("Neuron Index")
                plt.title("Neuron Input Counts Over Time")
                plt.title(
                    f"Inputs for timed array: {inp_population.name}\nshape={inputs.shape}, weight={self.mean_weights_by_type[key]}, max input={np.max(inputs)}"
                )
                plt.tight_layout()
                plt.show()

            # sum up all cortical inputs of dSPN receiver 0
            if key[0] in self.cortical_proportions_dict.keys() and key[1] == "dSPN":
                dSPN_inputs_sum.append(inputs[:, 0])

            # collect cortical chunks for FS validation
            if fs_debug_cache is not None and key[0] in self.cortical_proportions_dict:
                if key[1] in {"dSPN", "iSPN", "FS"}:
                    fs_debug_cache.setdefault(key[0], {})[key[1]] = inputs

            # update the TimedArray population with weighted inputs, rewinding the
            # internal timers so the new chunk is played from its first block
            inp_population.update(
                rates=inputs * self.mean_weights_by_type[key], reset=True
            )

        if self.verbose and self.debug and dSPN_inputs_sum:
            # plot the summed cortical inputs to dSPN neuron 0 over time
            plt.figure(figsize=(12, 6))
            total_inputs = np.sum(dSPN_inputs_sum, axis=0)
            plt.plot(
                np.arange(len(total_inputs)) * self.dt,
                total_inputs,
            )
            plt.xlabel("Time (ms)")
            plt.ylabel("Total Cortical Input Count to dSPN Neuron 0")
            plt.title(
                "Total Cortical Inputs to dSPN Neuron 0 Over Time (summed over regions)"
            )
            plt.tight_layout()
            plt.show()

        if (
            self.verbose
            and self.debug
            and fs_debug_cache
            and fs_scaling_factors is not None
        ):
            self._debug_validate_fs_inputs(
                fs_debug_cache=fs_debug_cache,
                scaling_factors=fs_scaling_factors,
            )

        # Optional simulate the network for the update_time
        if run_simulation:
            simulate(self.update_time)

    def reset(self) -> None:
        """Reset input iterators so the next ``update`` starts from the first chunk."""
        if not self.model_created:
            raise RuntimeError("create_model() must be called before reset().")

        n_steps_input = int(self.update_time / self.dt)
        if n_steps_input <= 0:
            raise ValueError("update_time must be at least one dt long")

        # Rebuild iterators for all ANNarchy input populations that were created.
        self.inp_iterator_dict = {}
        for key in self.annarchy_inp_populations.keys():
            memmap_info = None
            if self.local_input_memmap_dict and key in self.local_input_memmap_dict:
                memmap_info = self.local_input_memmap_dict[key]
            elif self.cor_input_memmap_dict and key in self.cor_input_memmap_dict:
                memmap_info = self.cor_input_memmap_dict[key]

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

    def _compute_fs_scaling_factors(self) -> np.ndarray | None:
        """Recompute the FS scaling factors used during FS input generation."""
        if ("FS", "dSPN") not in self.weights_by_type or (
            "FS",
            "iSPN",
        ) not in self.weights_by_type:
            if self.verbose:
                print(
                    "[FS debug] Missing FS->dSPN or FS->iSPN weights; cannot validate FS inputs."
                )
            return None

        W_fs_dspn = self.weights_by_type[("FS", "dSPN")]
        W_fs_ispn = self.weights_by_type[("FS", "iSPN")]

        N_spn_total = self.N_cortical_inputs_dict.get("dSPN", 0)
        N_fs_total = self.N_cortical_inputs_dict.get("FS", 0)
        if N_spn_total <= 0 or N_fs_total <= 0:
            if self.verbose:
                print(
                    "[FS debug] Invalid cortical input expectations for SPN/FS; cannot validate."
                )
            return None

        sum_w_dspn = np.array(W_fs_dspn.sum(axis=1)).flatten()
        sum_w_ispn = np.array(W_fs_ispn.sum(axis=1)).flatten()
        total_weighted_capacity = (sum_w_dspn * N_spn_total) + (
            sum_w_ispn * N_spn_total
        )

        scaling_factors = np.zeros_like(total_weighted_capacity, dtype=np.float64)
        mask = total_weighted_capacity > 0
        scaling_factors[mask] = N_fs_total / total_weighted_capacity[mask]
        return scaling_factors

    def _debug_validate_fs_inputs(
        self,
        fs_debug_cache: dict[str, dict[str, np.ndarray]],
        scaling_factors: np.ndarray,
    ) -> None:
        """Validate that FS inputs follow from dSPN/iSPN inputs and visualize the relation."""

        W_fs_dspn = self.weights_by_type[("FS", "dSPN")].tocsr()
        W_fs_ispn = self.weights_by_type[("FS", "iSPN")].tocsr()

        for region, region_chunks in fs_debug_cache.items():
            dspn_chunk = region_chunks.get("dSPN")
            ispn_chunk = region_chunks.get("iSPN")
            fs_chunk = region_chunks.get("FS")

            if dspn_chunk is None or ispn_chunk is None or fs_chunk is None:
                if self.verbose:
                    print(
                        f"[FS debug] {region}: missing chunks for validation. Have keys {list(region_chunks.keys())}."
                    )
                continue

            # inputs are stored as (steps, neurons); transpose for matrix multiplication
            dspn_inputs = dspn_chunk.T
            ispn_inputs = ispn_chunk.T
            fs_inputs_actual = fs_chunk.T

            expected_mean = (W_fs_dspn @ dspn_inputs) + (W_fs_ispn @ ispn_inputs)
            expected_mean = expected_mean * scaling_factors[:, None]

            if expected_mean.shape != fs_inputs_actual.shape:
                if self.verbose:
                    print(
                        f"[FS debug] {region}: shape mismatch expected {expected_mean.shape} vs actual {fs_inputs_actual.shape}."
                    )
                continue

            total_expected = float(expected_mean.sum())
            total_actual = float(fs_inputs_actual.sum())
            ratio_total = (
                total_actual / total_expected if total_expected > 0 else np.nan
            )

            flat_expected = expected_mean.ravel()
            flat_actual = fs_inputs_actual.ravel()
            corr = np.nan
            if flat_expected.std() > 0 and flat_actual.std() > 0:
                corr = float(np.corrcoef(flat_expected, flat_actual)[0, 1])

            per_neuron_expected_mean = expected_mean.mean(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                per_neuron_ratio = np.divide(
                    fs_inputs_actual.mean(axis=1),
                    per_neuron_expected_mean,
                    out=np.full_like(per_neuron_expected_mean, np.nan),
                    where=per_neuron_expected_mean > 0,
                )

            corr_display = "nan" if np.isnan(corr) else f"{corr:.3f}"
            mean_ratio = float(np.nanmean(per_neuron_ratio))
            std_ratio = float(np.nanstd(per_neuron_ratio))

            if self.verbose:
                print(
                    f"[FS debug] {region}: steps={fs_inputs_actual.shape[1]}, FS neurons={fs_inputs_actual.shape[0]}, "
                    f"total_expected={total_expected:.2f}, total_actual={total_actual:.2f}, "
                    f"total_ratio={ratio_total:.3f}, corr={corr_display}, "
                    f"mean_neuron_ratio={mean_ratio:.3f}+/-{std_ratio:.3f}"
                )

            time_axis = np.arange(fs_inputs_actual.shape[1]) * self.dt
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

            axes[0, 0].plot(
                time_axis, expected_mean.sum(axis=0), label="expected (from SPNs)"
            )
            axes[0, 0].plot(
                time_axis,
                fs_inputs_actual.sum(axis=0),
                label="actual (FS memmap)",
                alpha=0.7,
            )
            axes[0, 0].set_xlabel("Time (ms)")
            axes[0, 0].set_ylabel("Input count")
            axes[0, 0].set_title(f"{region}: total FS input per timestep")
            axes[0, 0].legend()

            sample_size = min(3000, flat_expected.size)
            sample_idx = (
                np.linspace(0, flat_expected.size - 1, num=sample_size, dtype=int)
                if sample_size > 0
                else np.array([], dtype=int)
            )
            axes[0, 1].scatter(
                flat_expected[sample_idx],
                flat_actual[sample_idx],
                s=6,
                alpha=0.6,
                label="samples",
            )
            if sample_size > 0:
                max_val = max(
                    flat_expected[sample_idx].max(), flat_actual[sample_idx].max()
                )
                axes[0, 1].plot([0, max_val], [0, max_val], "r--", lw=1, label="y=x")
            axes[0, 1].set_xlabel("Expected (Poisson mean)")
            axes[0, 1].set_ylabel("Actual (sampled)")
            axes[0, 1].set_title("Expected vs actual (sampled points)")
            axes[0, 1].legend()

            axes[1, 0].hist(
                per_neuron_ratio[~np.isnan(per_neuron_ratio)],
                bins=30,
                color="steelblue",
                edgecolor="black",
            )
            axes[1, 0].axvline(1.0, color="red", linestyle="--", label="ideal")
            axes[1, 0].set_xlabel("Mean(actual)/Mean(expected)")
            axes[1, 0].set_ylabel("FS neuron count")
            axes[1, 0].set_title("FS neuron-wise ratio")
            axes[1, 0].legend()

            n_show = min(3, fs_inputs_actual.shape[0])
            for idx in range(n_show):
                axes[1, 1].plot(
                    time_axis,
                    expected_mean[idx],
                    label=f"expected n{idx}",
                    linestyle="--",
                    alpha=0.8,
                )
                axes[1, 1].plot(
                    time_axis,
                    fs_inputs_actual[idx],
                    label=f"actual n{idx}",
                    alpha=0.8,
                )
            axes[1, 1].set_xlabel("Time (ms)")
            axes[1, 1].set_ylabel("Input count")
            axes[1, 1].set_title("Example FS neurons")
            axes[1, 1].legend()

            plt.suptitle(f"FS input validation for {region}")
            plt.show()

            # Detailed per-neuron view: one FS neuron and its connected SPNs
            fs_idx = None
            for candidate in range(fs_inputs_actual.shape[0]):
                if (
                    W_fs_dspn.getrow(candidate).nnz > 0
                    or W_fs_ispn.getrow(candidate).nnz > 0
                ):
                    fs_idx = candidate
                    break

            if fs_idx is None:
                if self.verbose:
                    print(
                        f"[FS debug] {region}: no FS neuron with SPN connections found for detailed plot."
                    )
                continue

            w_dspn_row = np.array(W_fs_dspn.getrow(fs_idx).toarray()).ravel()
            w_ispn_row = np.array(W_fs_ispn.getrow(fs_idx).toarray()).ravel()

            conn_list = []
            for idx, w in enumerate(w_dspn_row):
                if w > 0:
                    conn_list.append(("dSPN", idx, w))
            for idx, w in enumerate(w_ispn_row):
                if w > 0:
                    conn_list.append(("iSPN", idx, w))

            if not conn_list:
                if self.verbose:
                    print(
                        f"[FS debug] {region}: FS neuron {fs_idx} has no SPN connections for detailed plot."
                    )
                continue

            # Keep plot readable: show strongest connections first
            conn_list.sort(key=lambda x: x[2], reverse=True)
            max_traces = 6
            conn_list = conn_list[:max_traces]

            colors = plt.cm.tab10(np.linspace(0, 1, len(conn_list)))
            fig_detail, axes_detail = plt.subplots(
                1, 3, figsize=(15, 4), constrained_layout=True
            )

            # FS neuron spikes
            axes_detail[0].plot(
                time_axis,
                fs_inputs_actual[fs_idx],
                color="black",
                label=f"FS {fs_idx} actual",
            )
            axes_detail[0].plot(
                time_axis,
                expected_mean[fs_idx],
                color="gray",
                linestyle="--",
                label="expected",
            )
            axes_detail[0].set_title(f"FS neuron {fs_idx} input counts")
            axes_detail[0].set_xlabel("Time (ms)")
            axes_detail[0].set_ylabel("Input count")
            axes_detail[0].legend()

            # SPN spike counts (raw)
            for color, (ctype, idx, w) in zip(colors, conn_list):
                if ctype == "dSPN":
                    axes_detail[1].plot(
                        time_axis,
                        dspn_inputs[idx],
                        color=color,
                        label=f"dSPN {idx} (w={w:.3f})",
                    )
                else:
                    axes_detail[1].plot(
                        time_axis,
                        ispn_inputs[idx],
                        color=color,
                        label=f"iSPN {idx} (w={w:.3f})",
                    )
            axes_detail[1].set_title("Connected SPN spike counts")
            axes_detail[1].set_xlabel("Time (ms)")
            axes_detail[1].set_ylabel("Spike count")
            axes_detail[1].legend()

            # Weighted SPN spike counts
            for color, (ctype, idx, w) in zip(colors, conn_list):
                if ctype == "dSPN":
                    weighted = dspn_inputs[idx] * w
                else:
                    weighted = ispn_inputs[idx] * w
                axes_detail[2].plot(
                    time_axis,
                    weighted,
                    color=color,
                    label=f"{ctype} {idx} (w={w:.3f})",
                )
            axes_detail[2].set_title("Weighted SPN spike counts")
            axes_detail[2].set_xlabel("Time (ms)")
            axes_detail[2].set_ylabel("Weighted count")
            axes_detail[2].legend()

            fig_detail.suptitle(
                f"FS {fs_idx} and connected SPNs ({region})", fontsize=12
            )
            plt.show()

    def get_input_receiver_populations(self) -> dict[str, Population]:
        """
        Return the ANNarchy populations that receive external inputs.

        Returns
        -------
        dict
            Mapping {"dSPN": Population, "iSPN": Population, "FS": Population}
        """
        return {
            ct: self.annarchy_populations[ct]
            for ct in ("dSPN", "iSPN", "FS")
            if ct in self.annarchy_populations
        }

    def create_model(self) -> dict[str, Population]:
        """Instantiate ANNarchy objects (populations, inputs/projections).

        Call this after constructing the Microcircuit to keep heavy ANNarchy
        objects separate from data preparation.

        Returns
        -------
        dict
            Mapping {"dSPN": Population, "iSPN": Population, "FS": Population}
        """
        if self.model_created:
            if self.verbose:
                print("ANNarchy model already built; skipping create_model().")
            return

        # Create neuron populations
        self.create_populations_annarchy(type_counts=self.type_counts)

        # create projections between striatal populations
        self.create_local_projections_annarchy()

        # Ensure the input memmap dicts exist
        if self.local_input_memmap_dict is None:
            self._missing_local_input()
        if self.cor_input_memmap_dict is None:
            self._excitatory_inputs()

        # Build ANNarchy TimedArray inputs and projections for local gaba inputs, SPN neuron models have gaba as target for gaba currents
        self._create_inputs_annarchy(memmap_dict=self.local_input_memmap_dict)

        # Build ANNarchy TimedArray inputs and projections for cortical excitatory inputs, SPN neuron models have glut as target for nmda and ampa currents
        self._create_inputs_annarchy(memmap_dict=self.cor_input_memmap_dict)

        self.model_created = True

        return self.get_input_receiver_populations()

    def get_model_component_names(self) -> dict[str, list[str]]:
        """Return names of populations and projections built by ``create_model``."""
        if not self.model_created:
            raise RuntimeError(
                "create_model() must be called before accessing component names."
            )

        population_names = [pop.name for pop in self.annarchy_populations.values()]
        population_names.extend(
            inp.name for inp in self.annarchy_inp_populations.values()
        )

        projection_names = []
        projection_names.extend(
            proj.name for proj in getattr(self, "annarchy_projections", {}).values()
        )
        projection_names.extend(
            proj.name for proj in self.annarchy_inp_projections.values()
        )

        return {
            "populations": sorted(population_names),
            "projections": sorted(projection_names),
        }

    def create_local_projections_annarchy(self) -> None:
        """Create ANNarchy Projections between the striatal populations based on the
        sampled connectivity and weights.
        """
        self.annarchy_projections: dict[tuple[str, str], Projection] = {}
        for (pre_type, post_type), weight_matrix in self.weights_by_type.items():
            pre_pop = self.annarchy_populations[pre_type]
            post_pop = self.annarchy_populations[post_type]

            # Create projection
            proj = Projection(
                pre=pre_pop,
                post=post_pop,
                target="gaba",
                name=f"Proj_{pre_type}_{post_type}_{self.name}",
            )
            proj.connect_from_sparse(weight_matrix)
            self.annarchy_projections[(pre_type, post_type)] = proj

    def create_populations_annarchy(self, type_counts: dict[str, int]) -> None:
        """
        Create ANNarchy populations for each cell type with the specified counts.

        Args:
            type_counts: Dictionary mapping cell type labels to their respective counts.
        """

        self.annarchy_populations = {}
        for cell_type, count in type_counts.items():
            # TODO maybe need to change the dopamine parameter
            if cell_type == "dSPN":
                neuron_model = Izhikevich2007Humphries2009SPND1(
                    current_based_excitation=True
                )
            elif cell_type == "iSPN":
                neuron_model = Izhikevich2007Humphries2009SPND2(
                    current_based_excitation=True
                )
            elif cell_type == "FS":
                neuron_model = Izhikevich2007Humphries2009FSI(
                    current_based_excitation=True
                )
            else:
                raise ValueError(f"No neuron model for cell type: {cell_type}")

            population = Population(
                geometry=count, neuron=neuron_model, name=f"{self.name}_{cell_type}"
            )
            self.annarchy_populations[cell_type] = population

    # ----------------------
    # Connectivity creation
    # ----------------------
    def _get_neighborhood_radii_mm(self):
        """
        Get the neighborhood radius (mm) for each post type based on max sigma. It can
        not exceed half the max dimension of the periodic cube.
        """

        # compute max sigma (um) per post type (for neighbor query radius)
        max_sigma_um = {
            post_type: max(
                sigma_um
                for (_, post_type2), (_, sigma_um) in self.conn_params.items()
                if post_type2 == post_type
            )
            for post_type in self.cell_types
        }
        # 3 times max sigma
        max_sigma_mm = {pt: max_sigma_um[pt] * 3 * 1e-3 for pt in self.cell_types}

        # compute neighborhood radii (mm) which cannot exceed half the box size
        radii_mm = {
            post_type: min(max_sigma_mm[post_type], self.L.max() / 2)
            for post_type in self.cell_types
        }
        return radii_mm, max_sigma_mm

    def _neighbors_within(self, j: int, r_mm: float) -> set:
        """Return indices of neurons within radius ``r_mm`` from neuron ``j`` under periodic boundaries."""
        idxs = self.tree.query_ball_point(self.positions[j], r=r_mm)
        return set(idx % self.n_total for idx in idxs)

    def _periodic_distance(self, i: int, j: int) -> float:
        """Compute true periodic Euclidean distance (mm) between neurons ``i`` and ``j``."""
        delta = self.positions[i] - self.positions[j]
        delta = delta - self.L * np.round(delta / self.L)
        return float(np.linalg.norm(delta))

    def _build_connectivity(self) -> None:
        """Instantiate probabilistic connections and sample weights for all permitted pre/post type pairs."""
        # loop over postsynaptic neurons (global indices)
        for post_global in range(self.n_total):
            post_type = self.types[post_global]
            neighbor_idxs = self._neighbors_within(
                post_global, r_mm=self.neighborhood_radii_mm[post_type]
            )
            self.neighbor_sizes.append(len(neighbor_idxs))

            # loop over presynaptic candidates (global indices)
            for pre_global in neighbor_idxs:
                if pre_global == post_global:
                    continue
                pre_type = self.types[pre_global]
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue
                P0, sigma_um = self.conn_params[key]
                sigma = sigma_um * 1e-3
                dist = self._periodic_distance(pre_global, post_global)
                p = self._p_exp(dist, P0, sigma)
                self.con_probs[key].append((p, 0))
                if self.rng.random() < p:
                    # store global adjacency
                    self.adj[pre_type][post_type].append((pre_global, post_global))
                    self.con_probs[key][-1] = (p, 1)
                    # translate to local indices per type and set weight
                    pre_local = self.local_index_map[pre_type][pre_global]
                    post_local = self.local_index_map[post_type][post_global]
                    self.weights_by_type[key][pre_local, post_local] = (
                        self.weight_samplers[key].sample()
                    )
                    # record distance (mm)
                    self.connection_distances_by_pair[key].append(dist)

    def _save_connectivity_state(self) -> None:
        """Persist connectivity-related data for reuse without rebuilding."""
        meta = {
            "neighbor_sizes": self.neighbor_sizes,
            "con_probs": {f"{k[0]}-{k[1]}": v for k, v in self.con_probs.items()},
            "connection_distances_by_pair": {
                f"{k[0]}-{k[1]}": v
                for k, v in self.connection_distances_by_pair.items()
            },
            "adj": {
                pre: {post: pairs for post, pairs in inner.items()}
                for pre, inner in self.adj.items()
            },
            "rng_state": self.rng.bit_generator.state,
            "types": self.types,
            "type_counts": self.type_counts,
            "cell_types": self.cell_types,
            "n_total": self.n_total,
            "nx": self.nx,
            "b": self.b,
            "density": self.density,
            "conn_params": {
                f"{pre}-{post}": vals for (pre, post), vals in self.conn_params.items()
            },
        }
        with open(self._connectivity_state_path(), "wb") as f:
            pickle.dump(meta, f)
        for (pre_type, post_type), W in self.weights_by_type.items():
            save_npz(self._weight_matrix_path(pre_type, post_type), W.tocsr())

    def _load_connectivity_state(self) -> None:
        """Load connectivity data from disk if available; raise if missing."""
        state_path = self._connectivity_state_path()
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Connectivity cache not found at {state_path}. Rebuild by setting build_connectivity=True."
            )
        with open(state_path, "rb") as f:
            meta = pickle.load(f)

        # basic compatibility checks
        if meta.get("n_total") != self.n_total or meta.get("nx") != self.nx:
            raise ValueError(
                "Cached connectivity was generated for a different lattice; rebuild connectivity."
            )
        if not np.array_equal(meta.get("types"), self.types):
            raise ValueError(
                "Cached connectivity uses a different type assignment; rebuild connectivity or reuse the same seed/params."
            )

        self.neighbor_sizes = meta.get("neighbor_sizes", [])
        self.con_probs = {
            tuple(k.split("-")): v for k, v in meta.get("con_probs", {}).items()
        }
        self.connection_distances_by_pair = {
            tuple(k.split("-")): v
            for k, v in meta.get("connection_distances_by_pair", {}).items()
        }
        self.adj = meta.get("adj", self.adj)
        # ensure keys exist for all defined pairs
        for key in self.conn_params.keys():
            self.con_probs.setdefault(key, [])
            self.connection_distances_by_pair.setdefault(key, [])
            pre_type, post_type = key
            self.adj.setdefault(pre_type, {})
            self.adj[pre_type].setdefault(post_type, [])
        # load weights
        self.weights_by_type = {}
        for key in self.conn_params.keys():
            pre_type, post_type = key
            weight_path = self._weight_matrix_path(pre_type, post_type)
            if not os.path.exists(weight_path):
                raise FileNotFoundError(
                    f"Missing cached weight matrix at {weight_path}; rebuild connectivity."
                )
            self.weights_by_type[key] = load_npz(weight_path).tolil()

        rng_state = meta.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _expected_outer(self, rho, Rin, Rout, p_func):
        """
        Expected number of inputs from the outer shell [Rin, Rout] for a single receiver.

        Math:
            E[N_outer] = rho * ∫_{shell} p(r) dV
                    = 4*pi * rho * ∫_{Rin}^{Rout} p(r) r^2 dr

        Parameters
        ----------
        rho : float
            Presynaptic neuron density (neurons per unit volume).
        Rin : float
            Inner radius of the outer shell (units consistent with distance).
        Rout : float
            Outer cutoff radius.
        p_func : callable
            p_func(r) returns connection probability at distance r.

        Returns
        -------
        float
            Expected number of connections originating in the outer shell.

        Practical notes
        ---------------
        - Evaluate the integral with scipy.integrate.quad (adaptive).
        - Ensure units are consistent.
        - If p_func returns 0 beyond some radius, that is fine; integration will account for it.

        Example
        -------
        >>> E_outer = expected_outer(1e5, 0.05, 0.8, lambda r: p_exp(r, 0.5, 0.2))
        """
        integrand = lambda r: p_func(r) * r**2
        val, err = integrate.quad(
            integrand, Rin, Rout, epsabs=1e-8, epsrel=1e-6, limit=200
        )
        return 4 * np.pi * rho * val

    def _expected_shared_for_d(self, rho, Rin, Rout, p_func, d):
        """
        Expected number of shared presynaptic inputs from the outer shells of two receivers separated by distance d.

        Math derivation (summary):
            E[N_shared(d)] = rho * ∫ p(r_A) p(r_B) dV
        Place receiver A at origin, receiver B on polar axis at distance d.
            E[N_shared(d)] = 2*pi * rho * ∫_{r=Rin}^{Rout} r^2 ∫_{theta=0}^{pi}
                            p(r) p(r_B) sin(theta) dtheta dr
        where r_B = sqrt(r^2 + d^2 - 2*r*d*cos(theta)).

        Parameters
        ----------
        rho : float
            Presynaptic density (neurons per unit volume).
        Rin, Rout : float
            Inner and outer radii defining the shell of interest.
        p_func : callable
            Connection kernel p(r).
        d : float
            Distance between the two receiving neurons (units consistent with radii).

        Returns
        -------
        float
            Expected number of shared presynaptic neurons that are in both outer shells and connect to both receivers.

        Edge cases & checks
        -------------------
        - If d >= 2*Rout there is no overlap of the outer shells -> returns 0.
        - If d == 0, this reduces to E[N_shared(0)] = 4*pi*rho * ∫_{Rin}^{Rout} p(r)^2 r^2 dr.

        Numerical considerations
        ------------------------
        - This is a nested integral (r then theta). Use quad for the inner theta integral and then quad for r.
        - For many d values, consider caching/interpolating results.
        """
        if d >= 2 * Rout:
            return 0.0

        def inner_theta(theta, r):
            # distance to receiver B
            rB = np.sqrt(max(0.0, r * r + d * d - 2 * r * d * np.cos(theta)))
            if (rB < Rin) or (rB > Rout):
                return 0.0
            return p_func(r) * p_func(rB) * (r**2) * np.sin(theta)

        def integrand_r(r):
            val_theta, _ = integrate.quad(
                lambda th: inner_theta(th, r),
                0.0,
                np.pi,
                epsabs=1e-6,
                epsrel=1e-5,
                limit=200,
            )
            return val_theta

        val_r, _ = integrate.quad(
            integrand_r, Rin, Rout, epsabs=1e-6, epsrel=1e-5, limit=200
        )
        return 2 * np.pi * rho * val_r

    def _p_exp(
        self, d: float | np.ndarray, P0: float, sigma: float
    ) -> float | np.ndarray:
        """
        Exponential connection probability function.
        """
        return P0 * np.exp(-(d**2) / (sigma**2))

    def _excitatory_inputs(self):
        """Define cortical excitatory inputs based on BOLD-driven firing rates."""

        # Simulate and store the spike counts for the cortical inputs
        self.cor_input_memmap_dict = self._simulate_cor_input_spike_counts()

    def _simulate_cor_input_spike_counts(self):
        """Simulate spike counts for cortical input streams and store them."""
        # As rates for cortical drive, load the precomputed rates based on BOLD data
        rate_path = Path(self.cortical_rate_path)
        if not rate_path.exists():
            raise FileNotFoundError(
                f"Cortical drive rates not found at {rate_path}; run cortical_drive_by_bold.py first."
            )

        with np.load(rate_path) as data:
            # print keys in the loaded data
            if self.verbose:
                print(f"Loaded cortical drive data keys: {list(data.keys())}")
            # Infer dt (ms) from any region's time array to ensure consistency with the model dt
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

            if np.isclose(dt_ms, self.dt, rtol=1e-6, atol=1e-9):
                expansion_factor = 1
            elif dt_ms > self.dt:
                ratio = dt_ms / self.dt
                expansion_factor = int(round(ratio))
                if not np.isclose(ratio, expansion_factor, rtol=1e-6, atol=1e-9):
                    raise ValueError(
                        f"Cortical drive dt ({dt_ms:.6f} ms) is not an integer multiple of Microcircuit dt ({self.dt:.6f} ms)."
                    )
                if self.verbose:
                    print(
                        f"Cortical drive dt ({dt_ms:.6f} ms) is {expansion_factor}x the Microcircuit dt; repeating rate values accordingly."
                    )
            else:
                raise ValueError(
                    f"Cortical drive dt ({dt_ms:.6f} ms) is finer than Microcircuit dt ({self.dt:.6f} ms); please regenerate cortical drive data."
                )

            # Simulate spike counts per cortical region for dSPN and iSPN receivers
            cor_input_memmap_dict = {}
            for receiver_type in ("dSPN", "iSPN"):
                for (
                    cortical_region,
                    proportion,
                ) in self.cortical_proportions_dict.items():

                    if self.verbose:
                        print(
                            f"Simulating cortical input spike counts for region '{cortical_region}' to receiver type '{receiver_type}'."
                        )

                    spike_file = self._spike_counts_path(cortical_region, receiver_type)

                    rate_key = f"{cortical_region}_rate"
                    if rate_key not in data:
                        raise KeyError(
                            f"Rate key '{rate_key}' missing in cortical drive file {rate_path}."
                        )
                    rate_series = np.asarray(data[rate_key])
                    available_steps = rate_series.size * expansion_factor
                    if available_steps < self.n_steps:
                        raise ValueError(
                            f"Rate series for {cortical_region} provides {available_steps} microcircuit-sized steps after expansion; "
                            f"expected at least {self.n_steps}."
                        )
                    if expansion_factor > 1:
                        rate_series_expanded = np.repeat(rate_series, expansion_factor)
                        rate_segment = rate_series_expanded[: self.n_steps]
                    else:
                        rate_segment = rate_series[: self.n_steps]

                    # number of expected inputs from this cortical region
                    N_total = self.N_cortical_inputs_dict[receiver_type]
                    N = proportion * N_total
                    N_eff = int(np.round(N))
                    if N_eff == 0:
                        continue
                    # number of receivers R of the receiver type
                    R = self.type_counts[receiver_type]
                    # key is (pre, post)
                    key = (cortical_region, receiver_type)

                    simulate_receiver_counts_homogeneous_to_memmap(
                        filename=spike_file,
                        R=R,
                        N=N_eff,
                        shared_input=self.shared_fraction,
                        rate=rate_segment,
                        dt=self.dt,
                        rho=0.0,  # rho is ignored because fluctuations come rate time series based on BOLD
                        num_bins=self.n_steps,
                        receiver_dtype=np.float64,
                        rng=self.rng,
                        # concentration=1000.0,
                        verbose=self.verbose,
                    )

                    # Debugging check for dlPFC -> dSPN inputs: compare expected vs simulated counts in first chunk
                    if cortical_region == "dlPFC" and receiver_type == "dSPN":
                        chunk_steps = int(self.update_time / self.dt)
                        if chunk_steps > 0:
                            rate_chunk = rate_segment[:chunk_steps]
                            dt_seconds = self.dt / 1000.0
                            # Expected count per input neuron over the first chunk
                            expected_per_input = float(np.sum(rate_chunk) * dt_seconds)
                            expected_total = expected_per_input * N_eff

                            # Load first chunk of simulated counts from memmap
                            first_chunk_iter = iter_memmap_spike_counts(
                                filename=spike_file,
                                R=R,
                                num_bins=self.n_steps,
                                receiver_dtype=np.float64,
                                chunk_size=chunk_steps,
                                copy=False,
                                verbose=False,
                            )
                            first_chunk = next(first_chunk_iter)
                            simulated_sum_per_receiver = np.sum(first_chunk, axis=1)
                            sim_mean = float(np.mean(simulated_sum_per_receiver))
                            sim_std = float(np.std(simulated_sum_per_receiver))

                            if self.verbose:
                                print("[dlPFC->dSPN debug] First chunk diagnostics:")
                                print(
                                    f"  chunk_steps={chunk_steps}, dt_ms={self.dt}, chunk_time_ms={chunk_steps * self.dt}"
                                )
                                print(
                                    f"  N_eff (inputs)={N_eff}, rate_chunk_mean_Hz={np.mean(rate_chunk):.4f}, rate_chunk_sum_Hz={np.sum(rate_chunk):.4f}"
                                )
                                print(
                                    f"  expected_per_input_count={expected_per_input:.4f}, expected_total_count={expected_total:.4f}"
                                )
                                print(
                                    f"  simulated_sum_per_receiver: mean={sim_mean:.4f}, std={sim_std:.4f} (over {len(simulated_sum_per_receiver)} receivers)"
                                )
                        else:
                            if self.verbose:
                                print(
                                    "[dlPFC->dSPN debug] Skipped diagnostics because chunk_steps computed as 0."
                                )

                    # store infos in cor_input_memmap_dict
                    cor_input_memmap_dict[key] = {
                        "R": R,
                        "receiver_dtype": np.float64,
                    }

        # After generating SPN inputs, derive FS inputs from them
        cor_input_memmap_dict = self._derive_fs_cortical_inputs(cor_input_memmap_dict)

        return cor_input_memmap_dict

    def _calculate_max_chunk_size(
        self, N_dspn: int, N_ispn: int, N_fs: int, max_ram_mb: float = 512.0
    ) -> int:
        """
        Calculate safe chunk size based on available RAM.

        Args:
            N_dspn, N_ispn, N_fs: Number of neurons per type
            max_ram_mb: Target maximum RAM usage in MB
        """
        # Bytes per float64
        bytes_per_float = 8

        # Memory per time step:
        # We need to load chunks for dSPN and iSPN, compute sums, and store FS
        # Plus overhead for sparse matrix multiplication (intermediate buffers)
        # Conservative estimate: (N_d + N_i + 2 * N_fs) * bytes_per_float * safety_factor
        safety_factor = 4.0
        bytes_per_step = (N_dspn + N_ispn + 2 * N_fs) * bytes_per_float * safety_factor

        # Total available bytes
        total_bytes = max_ram_mb * 1024 * 1024

        # Steps fitting in memory
        chunk_size = int(total_bytes / bytes_per_step)

        # Ensure at least 1 step
        return max(1, chunk_size)

    def _derive_fs_cortical_inputs(self, cor_input_memmap_dict):
        """
        Derive cortical inputs for FS neurons from their connected dSPN/iSPN targets.

        The FS inputs are generated by:
        1. Identifying all dSPNs and iSPNs connected to each FS neuron.
        2. Computing a WEIGHTED sum of the cortical spike counts of these connected targets.
           Stronger connections contribute more to the "input pool".
        3. Scaling this weighted sum to match the expected FS input count (N_FS) via Poisson sampling.
           This ensures input magnitude is correct while preserving the relative influence of strong vs weak connections.
        """
        if self.verbose:
            print("Deriving FS cortical inputs from connected dSPN/iSPN targets...")

        # 1. Prepare connectivity matrices (FS -> dSPN and FS -> iSPN)
        # We use CSR format for efficient row slicing / matrix multiplication
        if ("FS", "dSPN") not in self.weights_by_type or (
            "FS",
            "iSPN",
        ) not in self.weights_by_type:
            if self.verbose:
                print(
                    "Warning: FS->dSPN or FS->iSPN weights missing. Cannot derive FS inputs."
                )
            return cor_input_memmap_dict

        W_fs_dspn = self.weights_by_type[("FS", "dSPN")].tocsr()
        W_fs_ispn = self.weights_by_type[("FS", "iSPN")].tocsr()

        # 2. Compute Scaling Factors
        # We want the expected output count to be N_FS_total.
        # The weighted sum has an expected value proportional to the sum of weights.
        # Scale_i = N_FS / (Sum_j(W_ij * N_SPN))

        N_spn_total = self.N_cortical_inputs_dict[
            "dSPN"
        ]  # Assuming dSPN and iSPN have same N (7000)
        N_fs_total = self.N_cortical_inputs_dict["FS"]  # 2800

        # Sum of weights per FS neuron (axis 1 = sum over columns/targets)
        # Result is shape (n_fs, 1)
        sum_w_dspn = np.array(W_fs_dspn.sum(axis=1)).flatten()
        sum_w_ispn = np.array(W_fs_ispn.sum(axis=1)).flatten()

        # Total weighted input capacity per FS neuron
        total_weighted_capacity = (sum_w_dspn * N_spn_total) + (
            sum_w_ispn * N_spn_total
        )

        # Avoid division by zero for FS neurons with no connections
        scaling_factors = np.zeros_like(total_weighted_capacity)
        mask = total_weighted_capacity > 0
        scaling_factors[mask] = N_fs_total / total_weighted_capacity[mask]

        # 3. Iterate over cortical regions
        R_fs = self.type_counts["FS"]
        R_dspn = self.type_counts["dSPN"]
        R_ispn = self.type_counts["iSPN"]

        # Calculate dynamic chunk size
        chunk_size = self._calculate_max_chunk_size(
            R_dspn, R_ispn, R_fs, max_ram_mb=512.0
        )
        # clip chunk size to not exceed total steps
        chunk_size = min(chunk_size, self.n_steps)

        if self.verbose:
            print(f"  Calculated chunk size: {chunk_size} steps")

        for cortical_region in self.cortical_proportions_dict.keys():
            # Only process if this region actually provides input
            if self.cortical_proportions_dict[cortical_region] <= 0:
                continue

            key_dspn = (cortical_region, "dSPN")
            key_ispn = (cortical_region, "iSPN")
            key_fs = (cortical_region, "FS")

            # Ensure SPN inputs exist
            if (
                key_dspn not in cor_input_memmap_dict
                or key_ispn not in cor_input_memmap_dict
            ):
                continue

            if self.verbose:
                print(f"  Processing {cortical_region} -> FS inputs...")

            # Output filename for FS inputs
            fs_spike_file = self._spike_counts_path(cortical_region, "FS")
            # Create memmap for writing FS inputs
            fs_mm = np.memmap(
                fs_spike_file, dtype=np.float64, mode="w+", shape=(R_fs, self.n_steps)
            )

            # Input iterators for SPN inputs
            iter_dspn = iter_memmap_spike_counts(
                filename=self._spike_counts_path(cortical_region, "dSPN"),
                R=R_dspn,
                num_bins=self.n_steps,
                receiver_dtype=np.float64,
                chunk_size=chunk_size,
            )
            iter_ispn = iter_memmap_spike_counts(
                filename=self._spike_counts_path(cortical_region, "iSPN"),
                R=R_ispn,
                num_bins=self.n_steps,
                receiver_dtype=np.float64,
                chunk_size=chunk_size,
            )

            # Iterate through time chunks
            current_step = 0
            for chunk_dspn, chunk_ispn in zip(iter_dspn, iter_ispn):
                # chunk shape: (n_neurons, n_steps_in_chunk)
                steps_in_chunk = chunk_dspn.shape[1]

                # Weighted Sum of inputs from connected targets
                # (n_fs, n_dspn) @ (n_dspn, steps) -> (n_fs, steps)
                weighted_sum = (W_fs_dspn @ chunk_dspn) + (W_fs_ispn @ chunk_ispn)

                # Apply scaling factor to match target N_FS expectation
                # scaling_factors shape (n_fs,), broadcast over time steps
                expected_fs_counts = weighted_sum * scaling_factors[:, None]

                # Sample discrete spikes using Poisson
                # This preserves the mean rate while generating integer counts
                fs_inputs = self.rng.poisson(expected_fs_counts).astype(np.float64)

                # Write to memmap
                fs_mm[:, current_step : current_step + steps_in_chunk] = fs_inputs
                current_step += steps_in_chunk

            fs_mm.flush()

            # Update dictionary
            cor_input_memmap_dict[key_fs] = {
                "R": R_fs,
                "receiver_dtype": np.float64,
            }

        return cor_input_memmap_dict

    def _build_distance_dependent_shared_fraction_matrices(self, f_d_interp_dict):
        """Build distance-dependent shared fraction matrices for all pre/post type pairs."""
        f_d_matrices_dict = {}
        for key, f_d_interp in f_d_interp_dict.items():
            _, post_type = key
            R = self.type_counts[post_type]
            # initialize shared fraction matrix
            f_d_matrix = np.zeros((R, R), dtype=np.float32)
            # loop over all receiver pairs
            for i_local, i_global in enumerate(self.indices_by_type[post_type]):
                for j_local, j_global in enumerate(self.indices_by_type[post_type]):
                    if i_global == j_global:
                        f_d_matrix[i_local, j_local] = 1.0
                    else:
                        d = self._periodic_distance(i_global, j_global)
                        f_d_matrix[i_local, j_local] = f_d_interp(d)
            # keep correlation/shared-fraction values within [0, 1]
            np.fill_diagonal(f_d_matrix, 1.0)
            np.clip(f_d_matrix, 0.0, 1.0, out=f_d_matrix)
            f_d_matrices_dict[key] = f_d_matrix
        return f_d_matrices_dict

    def _missing_local_input(self):
        """Construct distance-dependent shared input matrices and simulate local inhibitory spike counts."""
        # Get distance dependent shared input curves f(d)
        (
            f_d_interp_dict,
            f_d_raw_dict,
            expected_outer_dict,
            expected_shared_dict,
        ) = self._define_distance_dependent_shared_input_curves()

        # Given the shared input curves f(d) combined with receiver positions obtain shared fraction matrices
        f_d_matrices_dict = self._build_distance_dependent_shared_fraction_matrices(
            f_d_interp_dict=f_d_interp_dict
        )

        # Simulate spike counts for these matrices and assign to receivers and store them
        self.local_input_memmap_dict = self._simulate_distance_dependent_spike_counts(
            f_d_matrices_dict=f_d_matrices_dict, expected_outer_dict=expected_outer_dict
        )

        # store the mean of the weights per pre-post type pair
        for key in self.conn_params.keys():
            weight_samples = self.weight_samplers[key].sample(n=10000)
            self.mean_weights_by_type[key] = float(np.mean(weight_samples))

    def _create_inputs_annarchy(self, memmap_dict):
        """Create ANNarchy TimedArray input populations for spike counts and the
        corresponding input iterators for setting the inputs during simulation using stored data.

        Target selection is decided here:
        - Local striatal inputs (pre_type in self.cell_types) use `gaba`.
        - Cortical/external inputs (pre_type not in self.cell_types) use `glut`,
          except FS receivers, which should be driven via `ampa`.
        """
        if memmap_dict is None:
            raise ValueError(
                "memmap_dict is None; build or load spike-count inputs before creating ANNarchy inputs."
            )
        # Loop over the pre/post keys of the memmap-backed spike count files
        for key, memmap_info in memmap_dict.items():
            pre_type, post_type = key
            # skip if the postsynaptic population does not exist (safety for unexpected keys)
            if post_type not in self.annarchy_populations:
                continue
            # get the receiver population
            post_pop: Population = self.annarchy_populations[post_type]

            # Determine target based on pre/post identity
            if pre_type in self.cell_types:
                actual_target = "gaba"
            else:
                actual_target = "ampa" if post_type == "FS" else "glut"

            # create the input population and connect it to the receiver population
            # the input is initialized with placeholder zeros, this needs to be updated before simulation
            n_steps_input = int(self.update_time / self.dt)
            inp = TimedArray(
                rates=np.zeros((n_steps_input, post_pop.size)),
                name=f"TimedInput_{pre_type}_{post_type}_{self.name}",
            )
            proj = CurrentInjection(
                pre=inp,
                post=post_pop,
                target=actual_target,
                name=f"CurrentInjection_{pre_type}_{post_type}_{self.name}",
            )
            proj.connect_current()

            # create the input iterator for the update function
            spike_file = self._spike_counts_path(pre_type, post_type)
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

    def _simulate_distance_dependent_spike_counts(
        self, f_d_matrices_dict, expected_outer_dict
    ):
        """Generate spike-count time series for each f_d matrix."""
        local_input_memmap_dict = {}
        # Loop over postsynaptic neuron type
        for post_type in self.cell_types:
            # loop over presynaptic neuron type
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue

                spike_file = self._spike_counts_path(pre_type, post_type)

                # use an integer number of effective presynaptic sources; avoid fractional trials that can yield NaNs
                N_eff = int(round(expected_outer_dict[key]))
                if N_eff == 0:
                    continue

                simulate_receiver_counts_distance_dependent_to_memmap(
                    filename=spike_file,
                    correlation_matrix=f_d_matrices_dict[key],
                    N=N_eff,
                    rate=self.firing_rate_dict[pre_type],
                    dt=self.dt,
                    rho=self.correlation_dict[pre_type],
                    num_bins=self.n_steps,
                    receiver_dtype=np.float64,
                    rng=self.rng,
                    # concentration=1000.0,
                    verbose=self.verbose,
                )

                # store infos in local_input_memmap_dict
                R = self.type_counts[post_type]
                local_input_memmap_dict[key] = {
                    "R": R,
                    "receiver_dtype": np.float64,
                }
        return local_input_memmap_dict

    def _save_missing_input_state(self) -> None:
        """Persist distance-dependent input metadata to allow reloading without recomputation."""
        if self.local_input_memmap_dict is None:
            return
        payload = {
            "local_input_memmap_dict": self.local_input_memmap_dict,
            "rng_state": self.rng.bit_generator.state,
            "conn_keys": [f"{pre}-{post}" for (pre, post) in self.conn_params.keys()],
            "mean_weights_by_type": dict(self.mean_weights_by_type),
            "dt": self.dt,
            "n_steps": self.n_steps,
        }
        with open(self._missing_input_state_path(), "wb") as f:
            pickle.dump(payload, f)

    def _load_missing_input_state(self) -> None:
        """Load distance-dependent input state; expect spike-count files to exist."""
        state_path = self._missing_input_state_path()
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Missing cached missing-input state at {state_path}. Rebuild by setting build_missing_gaba_input=True."
            )
        with open(state_path, "rb") as f:
            payload = pickle.load(f)

        expected_keys = set(tuple(k.split("-")) for k in payload.get("conn_keys", []))
        if expected_keys and expected_keys != set(self.conn_params.keys()):
            raise ValueError(
                "Cached missing-input state does not match current connectivity parameters; rebuild missing inputs."
            )

        dt_saved = payload.get("dt", self.dt)
        if not np.isclose(dt_saved, self.dt, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"Cached missing-input dt ({dt_saved}) does not match current dt ({self.dt}); rebuild missing inputs."
            )

        n_steps_saved = payload.get("n_steps", self.n_steps)
        if n_steps_saved != self.n_steps:
            raise ValueError(
                f"Cached missing-input n_steps ({n_steps_saved}) does not match current n_steps ({self.n_steps}); rebuild missing inputs."
            )

        self.local_input_memmap_dict = payload.get("local_input_memmap_dict")
        if self.local_input_memmap_dict is None:
            raise ValueError(
                "Cached missing-input memmap info dict is empty; rebuild missing inputs."
            )

        mean_weights_saved = payload.get("mean_weights_by_type")
        if mean_weights_saved is None:
            raise ValueError(
                "Cached missing-input state is missing mean weights; rebuild missing inputs."
            )
        # merge local conectivity weights into existing defaults
        for key in self.conn_params.keys():
            self.mean_weights_by_type[key] = mean_weights_saved[key]

        # ensure spike-count files exist for all required pairs in the cached state
        for pre_type, post_type in self.local_input_memmap_dict.keys():
            path = self._spike_counts_path(pre_type, post_type)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Spike-count file for {pre_type}->{post_type} not found at {path}; rebuild missing inputs."
                )

        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _save_cortical_input_state(self) -> None:
        """Persist cortical input state to allow reloading without recomputation."""
        if self.cor_input_memmap_dict is None:
            return

        payload = {
            "cor_input_memmap_dict": self.cor_input_memmap_dict,
            "rng_state": self.rng.bit_generator.state,
            "dbs_condition": self.dbs_condition,
            "dt": self.dt,
            "n_steps": self.n_steps,
            "cortical_proportions_dict": self.cortical_proportions_dict,
            "N_cortical_inputs_dict": self.N_cortical_inputs_dict,
            "shared_fraction": self.shared_fraction,
            "cortical_rate_path": str(self.cortical_rate_path),
            "keys": [f"{pre}-{post}" for (pre, post) in self.cor_input_memmap_dict],
        }

        with open(self._cortical_input_state_path(), "wb") as f:
            pickle.dump(payload, f)

    def _load_cortical_input_state(self) -> None:
        """Load cortical input state; expect spike-count files to exist."""
        state_path = self._cortical_input_state_path()
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Cortical input cache not found at {state_path}. Rebuild by setting build_cortical_input=True."
            )

        with open(state_path, "rb") as f:
            payload = pickle.load(f)

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

        shared_fraction_saved = payload.get("shared_fraction")
        if (shared_fraction_saved is not None) and not np.isclose(
            shared_fraction_saved, self.shared_fraction
        ):
            raise ValueError(
                "Cached cortical-input state uses different shared_fraction; rebuild cortical inputs."
            )

        rate_path_saved = payload.get("cortical_rate_path")
        if rate_path_saved is not None:
            current_rate_path = str(Path(self.cortical_rate_path))
            if rate_path_saved != current_rate_path:
                raise ValueError(
                    "Cached cortical-input state uses different cortical_rate_path; rebuild cortical inputs."
                )

        self.cor_input_memmap_dict = payload.get("cor_input_memmap_dict")
        if self.cor_input_memmap_dict is None:
            raise ValueError(
                "Cached cortical-input memmap info dict is empty; rebuild cortical inputs."
            )

        saved_keys = set(tuple(k.split("-")) for k in payload.get("keys", []))
        if saved_keys and set(self.cor_input_memmap_dict.keys()) != saved_keys:
            raise ValueError(
                "Cached cortical-input state keys mismatch stored state; rebuild cortical inputs."
            )

        for pre_type, post_type in self.cor_input_memmap_dict.keys():
            path = self._spike_counts_path(pre_type, post_type)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Spike-count file for {pre_type}->{post_type} not found at {path}; rebuild cortical inputs."
                )

        rng_state = payload.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    def _define_distance_dependent_shared_input_curves(self):
        """Compute distance-dependent shared-input fraction curves f(d) for all valid type pairs."""

        # Get shared input fraction depending on distance f(d) considering the size of
        # the simulated volume and the distance-dependent connection probability
        # Number of inputs from outer shell:
        Rin_mm = (
            self.neighborhood_radii_mm
        )  # inner radius of outer shell (mm), i.e. simulated volume around receiver neuron
        Rout_mm = (
            self.max_sigma_mm
        )  # outer cutoff radius (mm), i.e. theoretical max sigma
        rho_pre = {
            pre_type: self.props[pre_type] * self.density
            for pre_type in self.cell_types
        }  # presynaptic density (neurons/mm^3)

        # variables to store the returns
        f_d_interp_dict = {}
        f_d_raw_dict = {}
        expected_outer_dict = {}
        expected_shared_dict = {}

        # Loop over postsynaptic type
        for post_type in self.cell_types:
            # loop over presynaptic type
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue
                P0, sigma_um = self.conn_params[key]
                sigma_mm = sigma_um * 1e-3  # mm

                # define p_func
                p_func = lambda r_mm: self._p_exp(r_mm, P0, sigma_mm)

                # compute expected number of inputs from outer shell per receiver neuron
                expected_outer = self._expected_outer(
                    rho=rho_pre[pre_type],
                    Rin=Rin_mm[post_type],
                    Rout=Rout_mm[post_type],
                    p_func=p_func,
                )
                expected_inner = self._expected_outer(
                    rho=rho_pre[pre_type],
                    Rin=0.0,
                    Rout=Rin_mm[post_type],
                    p_func=p_func,
                )
                if self.verbose:
                    print(f"Computed E_outer for {pre_type}->{post_type}...")
                    print(
                        f"  Rin={Rin_mm[post_type]:.3f} mm, Rout={Rout_mm[post_type]:.3f} mm"
                    )
                    print(f"  rho_pre={rho_pre[pre_type]:.2f} neurons/mm^3")
                    print(
                        f"  p_func at 0 mm = {p_func(0):.4f}, p_func at Rin = {p_func(Rin_mm[post_type]):.4f}"
                    )
                    print(
                        f"  E_outer = {expected_outer:.4f} inputs per receiver neuron"
                    )
                    print(
                        f"  E_inner = {expected_inner:.4f} inputs per receiver neuron (has to match with local inputs in simulated volume)"
                    )
                    print("\n")

                # compute expected shared inputs for distance d between two neurons
                # precalculate the expected shared inputs for some distances to later interpolate
                dmax = (
                    np.sqrt(3) * self.L.max() / 2
                )  # maximum possible distance between pair of neurons in periodic cube, i.e. half the space diagonal
                d_vals = np.linspace(0, dmax, 50)
                expected_shared_vals = np.array(
                    [
                        self._expected_shared_for_d(
                            rho_pre[pre_type],
                            Rin_mm[post_type],
                            Rout_mm[post_type],
                            p_func,
                            d,
                        )
                        for d in d_vals
                    ]
                )
                if self.verbose:
                    print(f"Computed E_shared_outer for {pre_type}->{post_type}...")
                    print(f"  For distances between pairs d in [0, {dmax:.3f}] mm")
                    print(
                        f"  Expected shared inputs at d=0 mm: {expected_shared_vals[0]:.4f}"
                    )
                    print(
                        f"  Expected shared inputs at d={dmax:.3f} mm: {expected_shared_vals[-1]:.4f}"
                    )
                    print("\n")

                # distance-dependent shared input fraction f(d)
                f_d = expected_shared_vals / max(expected_outer, 1e-12)

                # store f(d) as an interpolating function
                f_d_interp_dict[key] = interp1d(
                    d_vals, f_d, kind="cubic", fill_value="extrapolate"
                )

                # store the raw f_d values and expected outer and shared numbers for later use
                f_d_raw_dict[key] = (d_vals, f_d)
                expected_outer_dict[key] = expected_outer
                expected_shared_dict[key] = (d_vals, expected_shared_vals)

                # visualization of f(d) (optional)
                # plt.figure(figsize=(8, 6))
                # plt.subplot(211)
                # d_vals_plot = np.linspace(0, dmax, 200)
                # plt.plot(d_vals_plot, f_d_dict[key](d_vals_plot))
                # plt.plot(d_vals, f_d, "o")
                # plt.title(
                #     f"Shared input fraction f(d) for {pre_type}->{post_type} \n E_outer={expected_outer:.2f}"
                # )
                # plt.xlabel("Distance d (mm)")
                # plt.ylabel("Shared input fraction f(d)")
                # plt.subplot(212)
                # plt.plot(d_vals, expected_shared_vals)
                # plt.title(f"Expected shared inputs for {pre_type}->{post_type}")
                # plt.xlabel("Distance d (mm)")
                # plt.ylabel("Expected shared inputs")
                # plt.show()

        return (
            f_d_interp_dict,
            f_d_raw_dict,
            expected_outer_dict,
            expected_shared_dict,
        )

    # ----------------------
    # Reporting & summaries
    # ----------------------
    def summary(self) -> None:
        counts = {ct: int(np.sum(self.types == ct)) for ct in self.cell_types}
        print(
            f"Cuboid dimensions (μm): X={self.dim_x_um:.2f}, Y={self.dim_y_um:.2f}, Z={self.dim_z_um:.2f}"
        )
        print(f"Volume: {self.volume_mm3:.3f} mm³")
        for ct, cnt in counts.items():
            print(f"Neurons ({ct}): {cnt}")
        print(f"Grid spacing: {self.d_um:.2f} μm")

    def print_connections_created(self) -> None:
        print("\nConnections created:")
        for pre_type in self.cell_types:
            for post_type in self.cell_types:
                if (pre_type, post_type) in self.conn_params:
                    count = len(self.adj[pre_type][post_type])
                    print(f"{pre_type} -> {post_type}: {count} connections")
        # Also report matrix shapes
        print("\nPer-type weight matrix shapes:")
        for (pre_type, post_type), W in self.weights_by_type.items():
            print(f"W[{pre_type}->{post_type}] shape = {W.shape}")

    def get_weight_matrix(self, pre_type: str, post_type: str):
        """Return the sparse weight matrix for a pre->post type pair."""
        key = (pre_type, post_type)
        if key not in self.weights_by_type:
            raise KeyError(f"No weight matrix for pair {pre_type}->{post_type}")
        return self.weights_by_type[key]

    # ----------------------
    # Plots & visualizations
    # ----------------------
    def plot_ext_kdtree_points(self, show: bool = False) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            self.ext_positions[:, 0] * 1e3,
            self.ext_positions[:, 1] * 1e3,
            self.ext_positions[:, 2] * 1e3,
            c="gray",
            s=5,
            alpha=0.6,
        )
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "kdtree_tiled_points.png"))
            plt.close()

    def plot_neighborhood(self, j: int | None = None, show: bool = False) -> None:
        if j is None:
            j = self.j_center
        post_type = self.types[j]
        neighbor_idxs = self._neighbors_within(
            j, r_mm=self.neighborhood_radii_mm[post_type]
        )

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            self.positions[:, 0] * 1e3,
            self.positions[:, 1] * 1e3,
            self.positions[:, 2] * 1e3,
            c="gray",
            s=5,
            alpha=0.6,
        )
        ax.scatter(
            self.positions[j, 0] * 1e3,
            self.positions[j, 1] * 1e3,
            self.positions[j, 2] * 1e3,
            c="orange",
            s=50,
        )
        for i in neighbor_idxs:
            if i == j:
                continue
            pre_type = self.types[i]
            color = (
                "blue"
                if pre_type == "FS"
                else ("green" if pre_type == "dSPN" else "purple")
            )
            ax.scatter(
                self.positions[i, 0] * 1e3,
                self.positions[i, 1] * 1e3,
                self.positions[i, 2] * 1e3,
                c=color,
                s=20,
            )
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, f"neighborhood_j{j}.png"))
            plt.close()

    def plot_neighbor_candidate_counts(self, show: bool = False) -> None:
        if not self.built_connectivity:
            raise RuntimeError(
                "Connectivity not yet built; cannot plot neighbor candidate counts."
            )
        plt.figure()
        plt.plot(sorted(self.neighbor_sizes))
        plt.title("Number of neighbor candidates per neuron")
        plt.xlabel("Neuron index (sorted)")
        plt.ylabel("Number of neighbors within dmax")
        plt.grid(True)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "neighbor_candidate_counts.png"))
            plt.close()

    def plot_connection_probability_boxplots(self, show: bool = False) -> None:
        if not self.built_connectivity:
            raise RuntimeError(
                "Connectivity not yet built; cannot plot connection probability boxplots."
            )
        for (pre_type, post_type), probs in self.con_probs.items():
            if not probs:
                continue
            connected = [p for p, conn in probs if conn == 1]
            unconnected = [p for p, conn in probs if conn == 0]
            plt.figure(figsize=(8, 4))
            plt.boxplot(
                [connected, unconnected],
                positions=[0, 1],
                widths=0.4,
                tick_labels=[
                    f"{pre_type} -> {post_type} (connected - {len(connected)})",
                    f"{pre_type} -> {post_type} (unconnected - {len(unconnected)})",
                ],
            )
            plt.title(f"Connection probabilities: {pre_type} -> {post_type}")
            plt.ylabel("Connection probability")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            fig_name = f"boxplot_{pre_type}_{post_type}.png"
            if show:
                plt.show()
            else:
                plt.savefig(os.path.join(self.output_dir, fig_name))
                plt.close()

    def analyze_degree_distributions(self, show: bool = False) -> dict:
        if not self.built_connectivity:
            raise RuntimeError(
                "Connectivity not yet built; cannot analyze degree distributions."
            )
        hist_data: dict[str, dict[str, np.ndarray]] = {}
        for pre_type in self.cell_types:
            hist_data[pre_type] = {}
            for post_type in self.cell_types:
                if (pre_type, post_type) not in self.conn_params:
                    continue
                pairs = self.adj[pre_type][post_type]
                counts = np.zeros(self.n_total, dtype=int)
                for pre_idx, post_idx in pairs:
                    counts[post_idx] += 1
                print(f"{pre_type} -> {post_type}: {int(np.sum(counts))} total inputs")
                mask = self.types == post_type
                hist_data[pre_type][post_type] = counts[mask]
                plt.figure()
                plt.hist(hist_data[pre_type][post_type], bins=30)
                plt.title(f"{pre_type} -> {post_type} input counts")
                plt.xlabel("Number of inputs")
                plt.ylabel("Cell count")
                fig_name = f"hist_{pre_type}_{post_type}.png"
                if show:
                    plt.show()
                else:
                    plt.savefig(os.path.join(self.output_dir, fig_name))
                    plt.close()
        return hist_data

    def plot_3d_neurons(self, show: bool = False) -> None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        colors = {"FS": "red", "dSPN": "green", "iSPN": "blue"}
        ax.scatter(
            self.positions[:, 0] * 1e3,
            self.positions[:, 1] * 1e3,
            self.positions[:, 2] * 1e3,
            c=[colors[t] for t in self.types],
            s=5,
            alpha=0.6,
        )
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        for ct, col in colors.items():
            ax.scatter([], [], [], c=col, label=ct)
        ax.legend(loc="upper right")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "3D_neurons.png"))
            plt.close()

    def plot_half_gaussian_curves(self, show: bool = False) -> None:
        plt.figure(figsize=(8, 4))
        max_x = self.dim_x_um / 2
        x_vals = np.linspace(0, max_x, 500)
        for (pre, post), (P0, sigma_um) in self.conn_params.items():
            p_vals = self._p_exp(x_vals, P0, sigma_um)
            plt.plot(x_vals, p_vals, label=f"{pre}→{post} (d={sigma_um:.0f}µm)")
        plt.axvline(max_x, color="black", linestyle="--", label="Max periodic dist")
        plt.xlabel("Distance (µm)")
        plt.ylabel("Connection probability")
        plt.title("Half-gaussian connection-probability curves")
        plt.legend()
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "conn_prob_curves.png"))
            plt.close()

    # ----------------------
    # Connectivity matrix visualizations
    # ----------------------
    def plot_weight_matrix(
        self,
        pre_type: str,
        post_type: str,
        show: bool = False,
        method: str = "spy",
        figsize: tuple[int, int] = (5, 5),
        markersize: float = 1.0,
    ) -> None:
        """Visualize the sparse connectivity matrix for a given pre->post pair.

        Parameters
        ----------
        pre_type, post_type : str
            Neuron type labels for pre and post populations.
        show : bool
            If True display interactively, else save to file.
        method : str
            "spy" (default) uses plt.spy; "density" renders a low-res density image.
        figsize : (int, int)
            Figure size in inches.
        markersize : float
            Marker size for plt.spy.
        """
        if not self.built_connectivity:
            raise RuntimeError(
                "Connectivity not yet built; cannot plot weight matrices."
            )
        key = (pre_type, post_type)
        if key not in self.weights_by_type:
            raise KeyError(f"No weight matrix for {pre_type}->{post_type}")
        W = self.weights_by_type[key]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if method == "spy":
            ax.spy(W, markersize=markersize)
        elif method == "density":
            # downsample for visualization to avoid huge dense conversion
            W_csr = W.tocsr()
            # Create a coarse grid
            n_pre, n_post = W.shape
            scale = max(1, int(max(n_pre, n_post) / 500))
            # aggregate blocks
            dense = np.zeros((n_pre // scale + 1, n_post // scale + 1), dtype=float)
            rows, cols = W_csr.nonzero()
            for r, c in zip(rows, cols):
                dense[r // scale, c // scale] += 1
            ax.imshow(dense, origin="lower", aspect="auto", cmap="viridis")
        else:
            raise ValueError("method must be 'spy' or 'density'")
        nnz = W.nnz
        density_val = (
            nnz / (W.shape[0] * W.shape[1]) if W.shape[0] and W.shape[1] else 0
        )
        ax.set_title(
            f"Connectivity {pre_type}→{post_type}\nshape={W.shape} nnz={nnz} dens={density_val:.3e}"
        )
        ax.set_xlabel(f"post ({post_type}) index")
        ax.set_ylabel(f"pre ({pre_type}) index")
        # make the axes square if different pre and post sizes
        pre_size, post_size = W.shape
        aspect = post_size / pre_size
        ax.set_aspect(aspect=aspect)
        plt.tight_layout()
        fname = f"weights_{pre_type}_{post_type}.png"
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, fname))
            plt.close()

    def plot_all_weight_matrices(
        self,
        show: bool = False,
        method: str = "spy",
        max_cols: int = 3,
        markersize: float = 1.0,
        figsize_per: float = 3.5,
    ) -> None:
        """Plot all existing weight matrices in a grid.

        Parameters
        ----------
        show : bool
            Display interactively instead of saving.
        method : str
            'spy' or 'density' (see plot_weight_matrix).
        max_cols : int
            Maximum number of subplot columns.
        markersize : float
            Marker size passed to spy.
        figsize_per : float
            Base size per subplot (width & height scale).
        """
        if not self.built_connectivity:
            raise RuntimeError(
                "Connectivity not yet built; cannot plot weight matrices."
            )
        keys = list(self.weights_by_type.keys())
        if not keys:
            print("No weight matrices to plot.")
            return
        n = len(keys)
        cols = min(max_cols, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * figsize_per, rows * figsize_per)
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(rows, cols)
        for idx, key in enumerate(keys):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            pre_type, post_type = key
            W = self.weights_by_type[key]
            if method == "spy":
                ax.spy(W, markersize=markersize)
            elif method == "density":
                W_csr = W.tocsr()
                n_pre, n_post = W.shape
                scale = max(1, int(max(n_pre, n_post) / 300))
                dense = np.zeros((n_pre // scale + 1, n_post // scale + 1), dtype=float)
                rows_n, cols_n = W_csr.nonzero()
                for rr, cc in zip(rows_n, cols_n):
                    dense[rr // scale, cc // scale] += 1
                ax.imshow(dense, origin="lower", aspect="auto", cmap="viridis")
            else:
                raise ValueError("method must be 'spy' or 'density'")
            nnz = W.nnz
            dens = nnz / (W.shape[0] * W.shape[1]) if W.shape[0] and W.shape[1] else 0
            ax.set_title(
                f"{pre_type}→{post_type}\n{W.shape} nnz={nnz} d={dens:.2e}", fontsize=8
            )
            ax.set_xlabel("post")
            ax.set_ylabel("pre")
            # print the connection type and the expected number of inputs for a single post neuron:
            print(f"{pre_type} -> {post_type}: expected inputs per post neuron:")
            n_inputs_per_post = W.sum(axis=0)
            print(f"  Mean: {n_inputs_per_post.mean():.2f}")
            print(f"  Std: {n_inputs_per_post.std():.2f}")
            print(f"  Min: {n_inputs_per_post.min():.2f}")
            print(f"  Max: {n_inputs_per_post.max():.2f}")
        # hide unused axes
        for extra in range(n, rows * cols):
            r = extra // cols
            c = extra % cols
            axes[r, c].axis("off")
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, "weights_all.png"), dpi=150)
            plt.close()

    # ----------------------
    # Distance distribution visualization
    # ----------------------
    def plot_connection_counts_vs_distance(
        self,
        bins: int | np.ndarray = 50,
        per_pair: bool = True,
        micrometers: bool = True,
        show: bool = False,
        density: bool = False,
        cumulative: bool = False,
    ) -> dict[tuple[str, str], dict[str, np.ndarray]]:
        """Plot connection counts versus true periodic Euclidean distance.

        Parameters
        ----------
        bins : int or array-like
            Number of bins or explicit bin edges (in mm if micrometers=False, else µm).
        per_pair : bool
            If True, create one subplot per (pre_type, post_type) pair; else aggregate all distances.
        micrometers : bool
            Convert distances to µm for plotting.
        show : bool
            Display instead of saving.
        density : bool
            If True, normalize histogram to form a probability density.
        cumulative : bool
            If True, plot cumulative counts/density.

        Returns
        -------
        dict mapping (pre_type, post_type) to {'bin_edges','counts','centers'} arrays (aggregated key 'ALL' if per_pair=False).
        """
        if not self.built_connectivity:
            raise RuntimeError(
                "Connectivity not yet built; cannot plot connection distance distributions."
            )
        # Prepare data
        scale = 1e3 if micrometers else 1.0
        label_unit = "µm" if micrometers else "mm"

        def _hist(dist_list):
            arr = np.asarray(dist_list) * scale
            if isinstance(bins, int):
                counts, edges = np.histogram(arr, bins=bins, density=density)
            else:
                counts, edges = np.histogram(arr, bins=bins, density=density)
            if cumulative:
                counts = np.cumsum(counts)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return counts, edges, centers

        results: dict[tuple[str, str], dict[str, np.ndarray]] = {}

        if per_pair:
            keys = [
                k
                for k in self.conn_params.keys()
                if self.connection_distances_by_pair[k]
            ]
            if not keys:
                print("No connections to plot distance distribution.")
                return {}
            n = len(keys)
            cols = min(3, n)
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.reshape(rows, cols)
            for idx, key in enumerate(keys):
                r = idx // cols
                c = idx % cols
                ax = axes[r, c]
                dist_list = self.connection_distances_by_pair[key]
                counts, edges, centers = _hist(dist_list)
                ax.plot(centers, counts, drawstyle="steps-mid")
                pre_type, post_type = key
                ax.set_title(f"{pre_type}→{post_type} (n={len(dist_list)})", fontsize=9)
                ax.set_xlabel(f"Distance ({label_unit})")
                ax.set_ylabel(
                    "Cumulative" if cumulative else ("Density" if density else "Count")
                )
                results[key] = {
                    "bin_edges": edges,
                    "counts": counts,
                    "centers": centers,
                }
            # hide unused axes
            for extra in range(n, rows * cols):
                axes[extra // cols, extra % cols].axis("off")
            plt.tight_layout()
            fname = "connection_distance_per_pair.png"
        else:
            # aggregate all distances
            all_dists = [
                d for lst in self.connection_distances_by_pair.values() for d in lst
            ]
            if not all_dists:
                print("No connections to plot distance distribution.")
                return {}
            counts, edges, centers = _hist(all_dists)
            plt.figure(figsize=(6, 4))
            plt.plot(centers, counts, drawstyle="steps-mid")
            plt.xlabel(f"Distance ({label_unit})")
            plt.ylabel(
                "Cumulative" if cumulative else ("Density" if density else "Count")
            )
            plt.title("Connection counts vs distance (ALL pairs)")
            plt.tight_layout()
            results[("ALL", "ALL")] = {
                "bin_edges": edges,
                "counts": counts,
                "centers": centers,
            }
            fname = "connection_distance_all.png"

        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(self.output_dir, fname), dpi=150)
            plt.close()
        return results


if __name__ == "__main__":
    # Example usage: build microcircuit and reproduce main analyses, saving to output_dir
    mc = Microcircuit(
        name="caudate",
        dbs_condition="off",
        nx=10,
        b=10,
        build_cortical_input=False,
        build_missing_gaba_input=False,
        build_connectivity=False,
        verbose=True,
    )
    mc.plot_ext_kdtree_points(show=False)
    mc.plot_neighborhood(show=False)
    mc.plot_neighbor_candidate_counts(show=False)
    mc.plot_connection_probability_boxplots(show=False)
    mc.print_connections_created()
    mc.analyze_degree_distributions(show=False)
    mc.plot_3d_neurons(show=False)
    mc.plot_half_gaussian_curves(show=False)
    mc.plot_all_weight_matrices(show=False)
    mc.plot_weight_matrix("dSPN", "dSPN", show=False)
    mc.plot_weight_matrix("FS", "iSPN", show=False)
    mc.plot_connection_counts_vs_distance(
        per_pair=False, micrometers=True, density=True
    )
    mc.plot_connection_counts_vs_distance(
        per_pair=True, micrometers=True, cumulative=False
    )
