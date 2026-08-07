import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as sp
from scipy import integrate
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
    axon_pool_size,
    simulate_cortical_axon_pool_streams_to_memmap,
    simulate_receiver_counts_geometric_to_memmap,
    build_geometric_source_pools,
    check_stream_statistics,
    iter_memmap_spike_counts,
    validate_cortical_proportions,
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

    # Smallest number of virtual sources a receiver may have in a geometric pool.
    # Below this the source multiplicity quantises the realised degree and shared
    # fraction coarsely enough to bias them.
    _MIN_SOURCES_PER_RECEIVER = 50

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
        correlation_window_ms: float | None = None,
        correlation_timescale_ms: float = 0.0,
        shared_fraction: float | None = None,
        cortical_correlation: float = 0.0,
        source_multiplicity: int = 10,
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
        # default for dSPN and iSPN: the parkinsonian Off state (levodopa withdrawn)
        # of (Liang et al., 2008), Table 1, taken directly. See the derivation and its
        # assumptions in BGM_22/experimental_data/activity_striatum/README.md
        # default for FS: no parkinsonian-primate FSI recording exists, so 10.5 Hz is
        # the n-weighted normal-primate level (Marche und Apicella, 2021; Yamada et al.,
        # 2016; Adler et al., 2013) times a chronic dopamine-depletion factor of 1.0
        # (Mallet et al., 2006; Hernandez et al., 2013; He et al., 2024). Same README.
        if firing_rate_dict is None:
            firing_rate_dict = {"FS": 10.5, "dSPN": 25.0, "iSPN": 33.0}
        self.firing_rate_dict = firing_rate_dict

        # Pairwise spike-count correlation among the unsimulated striatal neurons
        # that feed the missing-GABA streams. No default: a correlation is only
        # meaningful together with the window it was measured at, and the value
        # sets the simulated BOLD amplitude (see input_streams/README.md section 3).
        if correlation_dict is None:
            raise ValueError(
                "Microcircuit requires correlation_dict; there is no default. In "
                "BGM_22 it is parameters.py['mc.correlation_dict']. It must be "
                "given together with correlation_window_ms, the window the value "
                "was measured at."
            )
        self.correlation_dict = correlation_dict
        if correlation_window_ms is None and any(
            v > 0 for v in correlation_dict.values()
        ):
            raise ValueError(
                "correlation_dict has non-zero entries but correlation_window_ms "
                "is None. A spike-count correlation without its measurement "
                "window is not a well-defined quantity -- see "
                "experimental_data/input_streams/README.md section 2."
            )
        self.correlation_window_ms = correlation_window_ms
        self.correlation_timescale_ms = correlation_timescale_ms
        self.source_multiplicity = source_multiplicity

        # expected number of input neurons from cortex per receiver neuron per cell type
        # default based on my calculations (see zotero/goolge/notebooks)
        if N_cortical_inputs_dict is None:
            N_cortical_inputs_dict = {"FS": 2800, "dSPN": 7000, "iSPN": 7000}
        self.N_cortical_inputs_dict = N_cortical_inputs_dict

        # the cortical proportions dict for our given cortical regions from the BOLD
        # data; required, see validate_cortical_proportions for why there is no default
        self.cortical_proportions_dict = validate_cortical_proportions(
            cortical_proportions_dict, "Microcircuit"
        )

        # Fraction of cortical afferents two striatal neurons have in common.
        # Kincaid et al. 1998: one corticostriatal axon contacts <=1.4 % of the
        # cells in its arborization, and the shared fraction between two SPNs
        # equals that same figure. No default, for the same reason the
        # proportions have none -- see validate_cortical_proportions.
        if shared_fraction is None:
            raise ValueError(
                "Microcircuit requires shared_fraction; there is no default. In "
                "BGM_22 it is parameters.py['mc.shared_fraction']."
            )
        if not (0.0 <= shared_fraction <= 1.0):
            raise ValueError(
                f"shared_fraction must be in [0, 1], got {shared_fraction}"
            )
        self.shared_fraction = shared_fraction
        # Pairwise spike-count correlation among the cortical neurons themselves.
        # 0 means they are conditionally independent given the BOLD-derived drive,
        # which is the only shared fluctuation the model then has. Any non-zero
        # value needs correlation_window_ms, like correlation_dict does.
        self.cortical_correlation = cortical_correlation
        if cortical_correlation > 0 and correlation_window_ms is None:
            raise ValueError(
                "cortical_correlation > 0 requires correlation_window_ms, the "
                "window the value was measured at. Cohen & Kohn 2011 Table 1 "
                "reports windows of 66-3000 ms; nothing is measured at dt."
            )
        # Per-stream target and measured statistics, filled during the build and
        # written into the cache state so every cache carries an audit trail.
        self.stream_statistics: dict[str, dict] = {}

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

            # All receiver types of one region draw on the SAME cortical axon
            # pool, so the streams for a region are generated together. Drawn
            # independently per receiver type, a dSPN and an iSPN sitting in the
            # same tissue and sampling the same axons would share nothing at all,
            # and an FS would share nothing with the SPNs it inhibits.
            cor_input_memmap_dict = {}
            for (
                cortical_region,
                proportion,
            ) in self.cortical_proportions_dict.items():
                if proportion <= 0:
                    # a region with no share gets no stream at all, which is the
                    # only reason the two loops differ in stream count
                    continue

                rate_key = f"{cortical_region}_rate"
                if rate_key not in data:
                    raise KeyError(
                        f"Rate key '{rate_key}' missing in cortical drive file {rate_path}."
                    )
                rate_series = np.asarray(data[rate_key])
                available_steps = rate_series.size * expansion_factor
                if available_steps < self.n_steps:
                    raise ValueError(
                        f"Rate series for {cortical_region} provides {available_steps} "
                        f"microcircuit-sized steps after expansion; expected at least "
                        f"{self.n_steps}."
                    )
                if expansion_factor > 1:
                    rate_segment = np.repeat(rate_series, expansion_factor)[
                        : self.n_steps
                    ]
                else:
                    rate_segment = rate_series[: self.n_steps]

                # The pool size is fixed by the shared fraction measured on SPNs,
                # so every other shared fraction -- FS to FS, FS to SPN -- follows
                # from it instead of being a free parameter.
                n_eff_reference = int(
                    np.round(proportion * self.N_cortical_inputs_dict["dSPN"])
                )
                if n_eff_reference == 0:
                    continue
                pool_size = axon_pool_size(
                    n_reference=n_eff_reference,
                    shared_fraction=self.shared_fraction,
                )

                streams = {}
                for receiver_type in self.cell_types:
                    n_eff = int(
                        np.round(
                            proportion * self.N_cortical_inputs_dict[receiver_type]
                        )
                    )
                    if n_eff == 0:
                        continue
                    streams[receiver_type] = {
                        "filename": self._spike_counts_path(
                            cortical_region, receiver_type
                        ),
                        "R": self.type_counts[receiver_type],
                        "N": n_eff,
                    }
                if not streams:
                    continue

                if self.verbose:
                    print(
                        f"Simulating cortical input spike counts for region "
                        f"'{cortical_region}' ({', '.join(streams)})."
                    )

                stats = simulate_cortical_axon_pool_streams_to_memmap(
                    streams=streams,
                    pool_size=pool_size,
                    rate=rate_segment,
                    dt=self.dt,
                    num_bins=self.n_steps,
                    receiver_dtype=np.float64,
                    rng=self.rng,
                    r_sc=self.cortical_correlation,
                    tau_c_ms=self.correlation_timescale_ms,
                    t_meas_ms=self.correlation_window_ms,
                    verbose=self.verbose,
                )
                for receiver_type in streams:
                    check_stream_statistics(
                        f"{cortical_region}->{receiver_type} ({self.name})",
                        stats[receiver_type]["target"],
                        stats[receiver_type]["measured"],
                        sample_shape=stats[receiver_type]["sample_shape"],
                    )
                    self.stream_statistics[
                        f"{cortical_region}-{receiver_type}"
                    ] = stats[receiver_type]
                    cor_input_memmap_dict[(cortical_region, receiver_type)] = {
                        "R": self.type_counts[receiver_type],
                        "receiver_dtype": np.float64,
                    }
                self.stream_statistics[f"{cortical_region}-pool"] = {
                    "pool_size": stats["pool_size"],
                    "cross_type_shared_fractions": stats[
                        "cross_type_shared_fractions"
                    ],
                }

        return cor_input_memmap_dict

    def _missing_local_input(self):
        """Simulate the GABAergic input from striatal neurons outside the cube."""
        expected_outer_dict, expected_inner_dict, kernel_dict = (
            self._define_expected_input_counts()
        )
        self.expected_inner_dict = expected_inner_dict

        self.local_input_memmap_dict = self._simulate_distance_dependent_spike_counts(
            expected_outer_dict=expected_outer_dict, kernel_dict=kernel_dict
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
        self, expected_outer_dict, kernel_dict
    ):
        """Generate the missing-GABA streams from explicit geometric source pools.

        For each pair, virtual source neurons are scattered around the receiver
        lattice and connected with the pair's own distance kernel. The shared
        fraction between two receivers is then the overlap of their pools, so
        ``f(d)`` is reproduced without being computed, and the realised degree
        reproduces ``E_outer`` without being imposed. Both are checked.
        """
        local_input_memmap_dict = {}
        self.stream_statistics = getattr(self, "stream_statistics", {})

        for post_type in self.cell_types:
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue

                expected_outer = expected_outer_dict[key]
                if int(round(expected_outer)) == 0:
                    continue

                P0, sigma_mm, r_in, r_out, rho_pre = kernel_dict[key]
                positions = self.positions[self.indices_by_type[post_type]]

                # One virtual source stands for `multiplicity` real neurons, which
                # keeps the source cloud small. But it also quantises a receiver's
                # degree, so a pair with few afferents needs a finer grain: at
                # multiplicity 10 the 12-afferent FS->FS pair would have barely one
                # source per receiver and both its degree and its shared fraction
                # would be rounded away.
                multiplicity = max(
                    1,
                    min(
                        self.source_multiplicity,
                        int(expected_outer // self._MIN_SOURCES_PER_RECEIVER),
                    ),
                )

                if self.verbose:
                    print(f"Building geometric source pool for {pre_type}->{post_type}")
                pools = build_geometric_source_pools(
                    receiver_positions=positions,
                    p_func=lambda d, _P0=P0, _s=sigma_mm: self._p_exp(d, _P0, _s),
                    r_in=r_in,
                    r_out=r_out,
                    density_pre=rho_pre,
                    rng=self.rng,
                    multiplicity=multiplicity,
                    verbose=self.verbose,
                )

                # The pool is one random realisation of the source cloud, so its
                # mean degree only has to agree with E_outer to within the spread
                # of that draw. Receiver degrees are NOT independent -- nearby
                # receivers share sources -- so the spread is set by the cloud,
                # not by the receiver count. Measured over 8 seeds it is 1.6 % of
                # E_outer for dSPN->dSPN and 4.8 % for FS->dSPN, with a bias below
                # 1 %; the tolerance below is ~4 sigma of the worst case. Anything
                # that actually breaks the construction -- a wrong kernel, radius
                # or density -- is off by a factor, not by 20 %.
                if abs(pools.mean_n_eff - expected_outer) > 0.20 * expected_outer:
                    raise ValueError(
                        f"Geometric pool for {pre_type}->{post_type} realises "
                        f"{pools.mean_n_eff:.1f} afferents per receiver but the "
                        f"kernel integral expects {expected_outer:.1f}. The pool "
                        "does not reproduce E_outer; check the kernel, r_in/r_out "
                        "or the density."
                    )

                stats = simulate_receiver_counts_geometric_to_memmap(
                    filename=self._spike_counts_path(pre_type, post_type),
                    pools=pools,
                    rate=self.firing_rate_dict[pre_type],
                    dt=self.dt,
                    num_bins=self.n_steps,
                    receiver_dtype=np.float64,
                    rng=self.rng,
                    r_sc=self.correlation_dict[pre_type],
                    tau_c_ms=self.correlation_timescale_ms,
                    t_meas_ms=self.correlation_window_ms,
                    verbose=self.verbose,
                )
                check_stream_statistics(
                    f"{pre_type}->{post_type} ({self.name})",
                    stats["target"],
                    stats["measured"],
                    sample_shape=stats["sample_shape"],
                )
                stats["expected_outer"] = float(expected_outer)
                self.stream_statistics[f"{pre_type}-{post_type}"] = stats

                local_input_memmap_dict[key] = {
                    "R": self.type_counts[post_type],
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
            # the spike counts were drawn at these rates and correlations, so a cache
            # built with different ones is not interchangeable
            "firing_rate_dict": dict(self.firing_rate_dict),
            "correlation_dict": dict(self.correlation_dict),
            # a correlation is only defined together with its measurement window
            # and timescale, so all three are part of what the cache was built at
            "correlation_window_ms": self.correlation_window_ms,
            "correlation_timescale_ms": self.correlation_timescale_ms,
            "source_multiplicity": self.source_multiplicity,
            # what the streams were measured to contain, so a cache can be
            # audited without regenerating it
            "stream_statistics": dict(self.stream_statistics),
            "expected_inner_dict": {
                f"{pre}-{post}": v
                for (pre, post), v in getattr(self, "expected_inner_dict", {}).items()
            },
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

        # The spike counts were drawn at a specific rate and correlation per cell type.
        # Both are absent from caches built before these fields were recorded, and such
        # a cache cannot be shown to match -- refuse it rather than load it silently.
        for field, current in (
            ("firing_rate_dict", self.firing_rate_dict),
            ("correlation_dict", self.correlation_dict),
        ):
            saved = payload.get(field)
            if saved is None:
                raise ValueError(
                    f"Cached missing-input state does not record {field}; it predates "
                    "this check and cannot be verified against the current settings. "
                    "Rebuild missing inputs."
                )
            mismatch = {
                cell_type: (saved.get(cell_type), current[cell_type])
                for cell_type in self.cell_types
                if not np.isclose(
                    saved.get(cell_type, np.nan), current[cell_type], rtol=1e-9, atol=0.0
                )
            }
            if mismatch:
                raise ValueError(
                    f"Cached missing-input {field} does not match current settings "
                    f"(cached vs current: {mismatch}); rebuild missing inputs."
                )

        # A correlation is only defined together with the window it was measured
        # at and the timescale it was realised with, so a cache built at different
        # ones holds different streams even at an identical correlation_dict.
        for field, current in (
            ("correlation_window_ms", self.correlation_window_ms),
            ("correlation_timescale_ms", self.correlation_timescale_ms),
            ("source_multiplicity", self.source_multiplicity),
        ):
            if field not in payload:
                raise ValueError(
                    f"Cached missing-input state does not record {field}; it "
                    "predates this check and cannot be verified. Rebuild missing "
                    "inputs."
                )
            saved = payload[field]
            same = (
                saved is None
                and current is None
                or saved is not None
                and current is not None
                and np.isclose(saved, current, rtol=1e-9, atol=0.0)
            )
            if not same:
                raise ValueError(
                    f"Cached missing-input {field} is {saved!r} but current "
                    f"setting is {current!r}; rebuild missing inputs."
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
            "cortical_correlation": self.cortical_correlation,
            "correlation_window_ms": self.correlation_window_ms,
            "correlation_timescale_ms": self.correlation_timescale_ms,
            "cortical_rate_path": str(self.cortical_rate_path),
            "keys": [f"{pre}-{post}" for (pre, post) in self.cor_input_memmap_dict],
            "stream_statistics": dict(self.stream_statistics),
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

        for field, current in (
            ("cortical_correlation", self.cortical_correlation),
            ("correlation_window_ms", self.correlation_window_ms),
            ("correlation_timescale_ms", self.correlation_timescale_ms),
        ):
            if field not in payload:
                raise ValueError(
                    f"Cached cortical-input state does not record {field}; it "
                    "predates this check and cannot be verified. Rebuild cortical "
                    "inputs."
                )
            saved = payload[field]
            same = (
                saved is None
                and current is None
                or saved is not None
                and current is not None
                and np.isclose(saved, current, rtol=1e-9, atol=0.0)
            )
            if not same:
                raise ValueError(
                    f"Cached cortical-input {field} is {saved!r} but current "
                    f"setting is {current!r}; rebuild cortical inputs."
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

    def _define_expected_input_counts(self):
        """Expected presynaptic counts inside and outside the simulated volume.

        The simulated cube is far smaller than the connection kernel reaches, so
        each receiver is wired only out to ``Rin`` and everything from there to
        ``Rout = 3 sigma`` has to be supplied synthetically. This returns how many
        afferents that is per pair.

        It no longer computes the shared fraction ``f(d)``. That used to be a
        50-point nested quadrature whose result was then imposed on the draw
        through a Gaussian copula; the geometric source pools of
        ``_simulate_distance_dependent_spike_counts`` realise the same overlap
        directly, so ``f(d)`` emerges instead of being imposed and is exact at
        any lattice size or density.

        Returns:
            expected_outer_dict (dict):
                ``(pre, post) -> E_outer``, the synthetic afferent count.

            expected_inner_dict (dict):
                ``(pre, post) -> E_inner``, what the simulated cube supplies. Only
                a sanity check against the connections actually made.

            kernel_dict (dict):
                ``(pre, post) -> (P0, sigma_mm, Rin_mm, Rout_mm, rho_pre)``, what
                the geometric pool builder needs.
        """
        Rin_mm = self.neighborhood_radii_mm
        Rout_mm = self.max_sigma_mm
        rho_pre = {
            pre_type: self.props[pre_type] * self.density
            for pre_type in self.cell_types
        }

        expected_outer_dict = {}
        expected_inner_dict = {}
        kernel_dict = {}

        for post_type in self.cell_types:
            for pre_type in self.cell_types:
                key = (pre_type, post_type)
                if key not in self.conn_params:
                    continue
                P0, sigma_um = self.conn_params[key]
                sigma_mm = sigma_um * 1e-3
                p_func = lambda r_mm, _P0=P0, _s=sigma_mm: self._p_exp(r_mm, _P0, _s)

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
                    print(
                        f"  {pre_type}->{post_type}: Rin={Rin_mm[post_type]:.3f} mm, "
                        f"Rout={Rout_mm[post_type]:.3f} mm, "
                        f"E_inner={expected_inner:.1f}, E_outer={expected_outer:.1f}"
                    )

                expected_outer_dict[key] = expected_outer
                expected_inner_dict[key] = expected_inner
                kernel_dict[key] = (
                    P0,
                    sigma_mm,
                    Rin_mm[post_type],
                    Rout_mm[post_type],
                    rho_pre[pre_type],
                )

        return expected_outer_dict, expected_inner_dict, kernel_dict

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
        # placeholder only -- this demo builds no cortical input, so the mapping is
        # never used. It is NOT the model's proportions; those live in BGM_22's
        # BOLD_optimization/parameters.py.
        cortical_proportions_dict={"dlPFC": 1.0},
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
