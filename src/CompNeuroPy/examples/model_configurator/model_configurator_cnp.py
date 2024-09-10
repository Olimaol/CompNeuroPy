from CompNeuroPy.generate_model import CompNeuroModel
from CompNeuroPy.experiment import CompNeuroExp
from CompNeuroPy.monitors import CompNeuroMonitors
from CompNeuroPy import model_functions as mf
from CompNeuroPy import extra_functions as ef
from CompNeuroPy import system_functions as sf
from ANNarchy import (
    Population,
    Projection,
    Synapse,
    get_population,
    Monitor,
    Network,
    get_projection,
    dt,
    simulate,
    Neuron,
    Binomial,
    CurrentInjection,
    raster_plot,
    set_seed,
)
from ANNarchy.core import ConnectorMethods
import numpy as np
from scipy.signal import argrelmin
import matplotlib.pyplot as plt
import inspect
import sympy as sp
import textwrap
import os

_model_configurator_verbose = False
_model_configurator_figure_folder = "model_configurator_figures"
_model_configurator_do_plot = False

_p_g_after_get_weights = (
    lambda template_weight_dict, template_synaptic_load_dict, template_synaptic_contribution_dict: f"""Now either set the weights of all projections directly or first set the synaptic load of the populations and the synaptic contributions of the afferent projections.
You can set the weights using the function .set_weights() which requires a weight_dict as argument.
Use this template for the weight_dict:

{template_weight_dict}

The values within the template are the maximum weight values.


You can set the synaptic load and contribution using the function .set_syn_load() which requires a synaptic_load_dict for the synaptic load of the populations and a synaptic_contribution_dict for the synaptic contributions to the synaptic load of the afferent projections.
Use this template for the synaptic_load_dict:

{template_synaptic_load_dict}

These values scale the maximum possible weights of the afferent projections of the corresponing target types for each population. The value 1 means that the highest possible weight in the afferent projections is the maximum weight of the population/target type (obtained previously by the ModelConfigurator).

Use this template for the synaptic_contribution_dict:

{template_synaptic_contribution_dict}

These values scale the maximum weights which are set by the synaptic_load_dict. So values between 0 and 1 make sense (1 means, take the maximum possible weight, 0.5 means take half of the maximum possible weight, etc.).
"""
)


class ModelConfigurator:
    """
    The external applied current has to be the variable I_app.
    """

    def __init__(
        self,
        model: CompNeuroModel,
        target_firing_rate_dict: dict,
        max_psp: float = 1.0,
        v_tol: float = 1.0,
        do_not_config_list: list[str] = [],
        cache: bool = False,
        clear_cache: bool = False,
        log_file: str | None = None,
        verbose: bool = False,
        do_plot: bool = False,
    ):
        """
        Args:
            model (CompNeuroModel):
                Model to be configured
            target_firing_rate_dict (dict):
                Dict with the target firing rates for each population
            max_psp (float, optional):
                Maximum inhibitory post-synaptic potential caused by single spikes in mV
                Default is 1.0
            v_tol (float, optional):
                Tolerance for the firing rate in Hz for the optimization of the baseline
                currents. Default is 1.0
            do_not_config_list (list[str], optional):
                List with the names of the populations which should not be configured
                (they should already have the firing rates given in
                target_firing_rate_dict). Default is []
            cache (bool, optional):
                If True, it is tried to load cached variables. If not possible variables
                created during the initialization are cached. Default is False
            clear_cache (bool, optional):
                If True, the cache is cleared and new variables are created. Set this to
                True if you changed the model, max_psp or target_firing_rate_dict.
                Default is False
            log_file (str | None, optional):
                Name of the log file to be created. If None, no log file is created.
                Default is None
            verbose (bool, optional):
                If True, the log is printed to the console. Default is False
            do_plot (bool, optional):
                If True, the plots of simulated data are saved in the folder
                'model_configurator_figures'. Drastically increases the runtime.
                Default is False
        """
        ### update global verbose and do_plot
        global _model_configurator_verbose
        _model_configurator_verbose = verbose
        global _model_configurator_do_plot
        _model_configurator_do_plot = do_plot
        ### store the given variables
        self._model = model
        self._do_not_config_list = do_not_config_list
        self._target_firing_rate_dict = target_firing_rate_dict
        self._base_dict = None
        self.v_tol = v_tol
        ### store the state of the given model
        self._model_was_created = model.created
        self._model_was_compiled = model.compiled
        ### create the figure folder
        sf.create_dir(_model_configurator_figure_folder)
        ### initialize logger
        sf.Logger(log_file=log_file)
        ### analyze the given model, create model before analyzing, then clear ANNarchy
        self._analyze_model = AnalyzeModel(model=self._model)
        ### create the CompNeuroModel object for the reduced model (the model itself is
        ### not created yet)
        self._model_reduced = CreateReducedModel(
            model=self._model,
            analyze_model=self._analyze_model,
            reduced_size=100,
            do_create=False,
            do_compile=False,
            verbose=_model_configurator_verbose,
        )
        ### try to load the cached variables
        if clear_cache:
            sf.clear_dir(".model_config_cache")
        cache_worked = False
        if cache:
            try:
                ### load the cached variables
                cache_loaded = sf.load_variables(
                    name_list=["init_sampler", "max_syn"],
                    path=".model_config_cache",
                )
                cache_worked = True
                sf.Logger().log(
                    "Loaded cached variables.", verbose=_model_configurator_verbose
                )
            except FileNotFoundError:
                pass
        if not cache_worked:
            sf.Logger().log(
                "No cached variables loaded, creating new ones.",
                verbose=_model_configurator_verbose,
            )
        ### create the single neuron networks (networks are compiled and ready to be
        ### simulated), normal model for searching for max conductances, max input
        ### current, resting firing rate; voltage clamp model for preparing the PSP
        ### simulationssearching, i.e., for resting potential and corresponding input
        ### current I_hold (for self-active neurons)
        if not cache_worked:
            self._single_nets = CreateSingleNeuronNetworks(
                model=self._model,
                analyze_model=self._analyze_model,
                do_not_config_list=do_not_config_list,
            )
            ### get the init sampler for the populations
            self._init_sampler = self._single_nets.init_sampler(
                model=self._model, do_not_config_list=do_not_config_list
            )
            ### create simulator with single_nets
            self._simulator = Simulator(
                single_nets=self._single_nets,
                prepare_psp=None,
            )
        else:
            self._init_sampler: CreateSingleNeuronNetworks.AllSampler = cache_loaded[
                "init_sampler"
            ]
        ### get the resting potential and corresponding I_hold for each population using
        ### the voltage clamp networks
        if not cache_worked:
            self._prepare_psp = PreparePSP(
                model=self._model,
                single_nets=self._single_nets,
                do_not_config_list=do_not_config_list,
                simulator=self._simulator,
            )
            self._simulator = Simulator(
                single_nets=self._single_nets,
                prepare_psp=self._prepare_psp,
            )
        ### get the maximum synaptic conductances and input currents for each population
        if not cache_worked:
            self._max_syn = GetMaxSyn(
                model=self._model,
                simulator=self._simulator,
                do_not_config_list=do_not_config_list,
                max_psp=max_psp,
                target_firing_rate_dict=target_firing_rate_dict,
            ).max_syn_getter
        else:
            self._max_syn = cache_loaded["max_syn"]
        ### cache single_nets, prepare_psp, max_syn
        if cache and not cache_worked:
            sf.Logger().log("Caching variables.", verbose=_model_configurator_verbose)
            sf.save_variables(
                variable_list=[
                    self._init_sampler,
                    self._max_syn,
                ],
                name_list=["init_sampler", "max_syn"],
                path=".model_config_cache",
            )
        ### get the weights dictionaries
        self._weight_dicts = GetWeights(
            model=self._model,
            do_not_config_list=do_not_config_list,
            analyze_model=self._analyze_model,
            max_syn=self._max_syn,
        )

        ### recreate the state of the given model
        sf.Logger().log(
            "Clearing ANNarchy and recreating the given model's state.",
            verbose=_model_configurator_verbose,
        )
        mf.cnp_clear(functions=False, constants=False)
        if self._model_was_created:
            self._model.create(do_compile=self._model_was_compiled)

        ### TODO print an explanation for setting the weight dict or the syn load/contribution dicts
        if _model_configurator_verbose:
            self._p_g(
                _p_g_after_get_weights(
                    self._weight_dicts.weight_dict,
                    self._weight_dicts.syn_load_dict,
                    self._weight_dicts.syn_contribution_dict,
                )
            )

    def _p_g(self, txt):
        """
        prints guiding text
        """
        print_width = min([os.get_terminal_size().columns, 80])

        print("\n[model_configurator guide]:")
        for line in txt.splitlines():
            wrapped_text = textwrap.fill(
                line, width=print_width - 5, replace_whitespace=False
            )
            wrapped_text = textwrap.indent(wrapped_text, "    |")
            print(wrapped_text)
        print("")

    def set_weights(self, weight_dict: dict[str, float]):
        """
        Set the weights of the model.

        Args:
            weight_dict (dict[str, float]):
                Dict with the weights for each projection
        """
        self._weight_dicts.weight_dict = weight_dict
        self._check_if_not_config_pops_have_correct_rates()

    def set_syn_load(
        self,
        syn_load_dict: dict[str, float],
        syn_contribution_dict: dict[str, dict[str, float]],
    ):
        """
        Set the synaptic load of the model.

        Args:
            syn_load_dict (dict[str, float]):
                Dict with ampa and gaba synaptic load for each population
            syn_contribution_dict (dict[str, dict[str, float]]):
                Dict with the contribution of the afferent projections to the ampa and
                gaba synaptic load of each population
        """
        self._weight_dicts.syn_load_dict = syn_load_dict
        self._weight_dicts.syn_contribution_dict = syn_contribution_dict
        self._check_if_not_config_pops_have_correct_rates()

    def _check_if_not_config_pops_have_correct_rates(self):
        """
        Check if the populations which should not be configured have the correct firing
        rates.
        """
        ### initialize the normal model + compile the model
        self._init_model_with_fitted_base()

        ### record spikes of the do_not_config populations
        mon = CompNeuroMonitors(
            mon_dict={
                pop_name: ["spike"] for pop_name in self._do_not_config_list
            }  # _model.populations # tmp test
        )
        mon.start()
        ### simulate the model for 5000 ms
        # get_population("stn").I_app = 8  # tmp test
        simulate(5000)

        ### get the firing rates
        recordings = mon.get_recordings()
        for pop_name in self._do_not_config_list:
            spike_dict = recordings[0][f"{pop_name};spike"]
            t, _ = raster_plot(spike_dict)
            spike_count = len(t)
            pop_size = len(get_population(pop_name))
            firing_rate = spike_count / (5 * pop_size)
            if np.abs(firing_rate - self._target_firing_rate_dict[pop_name]) > 1:
                sf.Logger().log(
                    f"Warning: Population {pop_name} has a firing rate of {firing_rate} instead of {self._target_firing_rate_dict[pop_name]}"
                )
                print(
                    f"Warning: Population {pop_name} has a firing rate of {firing_rate} instead of {self._target_firing_rate_dict[pop_name]}"
                )

        # ### tmp plot
        # recording_times = mon.get_recording_times()

        # af.PlotRecordings(
        #     figname="tmp.png",
        #     recordings=recordings,
        #     recording_times=recording_times,
        #     shape=(len(self._model.populations), 1),
        #     plan={
        #         "position": list(range(1, len(self._model.populations) + 1)),
        #         "compartment": self._model.populations,
        #         "variable": ["spike"] * len(self._model.populations),
        #         "format": ["hybrid"] * len(self._model.populations),
        #     },
        # )
        # quit()

    def set_base(self, base_dict: dict[str, float] | None = None):
        """
        Set the baseline currents of the model, found for the current weights to reach
        the target firing rates. The model is compiled after setting the baselines.

        Args:
            base_dict (dict[str, float], optional):
                Dict with the baseline currents for each population. If None, the
                optimized baseline currents are used. Default is None
        """
        ### either use the given base dict or get the base dict (or use the stored one)
        if base_dict is None:
            if self._base_dict is None:
                self.get_base()
            base_dict = self._base_dict

        ### initialize the normal model + set the baselines with the base dict
        self._init_model_with_fitted_base(base_dict=base_dict)

    def get_base(self):
        """
        Get the baseline currents of the model.

        Returns:
            base_dict (dict[str, float]):
                Dict with the baseline currents for each population
        """
        ### get the base dict
        self._base_dict = GetBase(
            model_normal=self._model,
            model_reduced=self._model_reduced,
            target_firing_rate_dict=self._target_firing_rate_dict,
            weight_dicts=self._weight_dicts,
            do_not_config_list=self._do_not_config_list,
            init_sampler=self._init_sampler,
            max_syn=self._max_syn,
            v_tol=self.v_tol,
        ).base_dict
        return self._base_dict

    def _init_model_with_fitted_base(self, base_dict: dict[str, float] | None = None):
        """
        Initialize the neurons of the model using the init_sampler, set the baseline
        currents of the model from the base dict (containing fitted baselines) and the
        weights from the weight dicts and compile the model.
        """
        ### clear ANNarchy and create the normal model
        mf.cnp_clear(functions=False, constants=False)
        self._model.create(do_compile=False)
        ### set the initial variables of the neurons #TODO small problem = init sampler
        ### initializes the neurons in resting-state, but here they get an input current
        for pop_name, init_sampler in self._init_sampler.init_sampler_dict.items():
            init_sampler.set_init_variables(get_population(pop_name))
        ### set the baseline currents
        if base_dict is not None:
            for pop_name, I_app in base_dict.items():
                setattr(get_population(pop_name), "I_app", I_app)
        ### compile the model
        self._model.compile()
        ### set the weights
        for proj_name, weight in self._weight_dicts.weight_dict.items():
            setattr(get_projection(proj_name), "w", weight)


class AnalyzeModel:
    """
    Class to analyze the given model to be able to reproduce it.
    """

    _connector_methods_dict = {
        "One-to-One": ConnectorMethods.connect_one_to_one,
        "All-to-All": ConnectorMethods.connect_all_to_all,
        "Gaussian": ConnectorMethods.connect_gaussian,
        "Difference-of-Gaussian": ConnectorMethods.connect_dog,
        "Random": ConnectorMethods.connect_fixed_probability,
        "Random Convergent": ConnectorMethods.connect_fixed_number_pre,
        "Random Divergent": ConnectorMethods.connect_fixed_number_post,
        "User-defined": ConnectorMethods.connect_with_func,
        "MatrixMarket": ConnectorMethods.connect_from_matrix_market,
        "Connectivity matrix": ConnectorMethods.connect_from_matrix,
        "Sparse connectivity matrix": ConnectorMethods.connect_from_sparse,
        "From File": ConnectorMethods.connect_from_file,
    }

    def __init__(self, model: CompNeuroModel):
        sf.Logger().log(
            "Analyzing model to be able to reproduce it",
            verbose=_model_configurator_verbose,
        )
        ### clear ANNarchy and create the model
        self._clear_model(model=model, do_create=True)

        ### get population info (eq, params etc.)
        self._analyze_populations(model=model)

        ### get projection info
        self._analyze_projections(model=model)

        ### clear ANNarchy
        self._clear_model(model=model, do_create=False)

    def _clear_model(self, model: CompNeuroModel, do_create: bool = True):
        if do_create:
            sf.Logger().log(
                "Clearing ANNarchy and creating model before analyzing",
                verbose=_model_configurator_verbose,
            )
        else:
            sf.Logger().log(
                "Clearing ANNarchy after analyzing",
                verbose=_model_configurator_verbose,
            )
        mf.cnp_clear(functions=False, constants=False)
        if do_create:
            model.create(do_compile=False)

    def _analyze_populations(self, model: CompNeuroModel):
        """
        Get info of each population

        Args:
            model (CompNeuroModel):
                Model to be analyzed
        """
        sf.Logger().log(
            "Analyzing populations to be able to reproduce them",
            verbose=_model_configurator_verbose,
        )
        ### values of the paramters and variables of the population's neurons, keys are
        ### the names of paramters and variables
        self.neuron_model_attr_dict: dict[str, dict] = {}
        ### arguments of the __init__ function of the Neuron class
        self.neuron_model_init_parameter_dict: dict[str, dict] = {}
        ### arguments of the __init__ function of the Population class
        self.pop_init_parameter_dict: dict[str, dict] = {}

        ### for loop over all populations
        for pop_name in model.populations:
            sf.Logger().log(
                f"Analyzing population {pop_name}", verbose=_model_configurator_verbose
            )
            pop: Population = get_population(pop_name)
            ### get the neuron model attributes (parameters/variables)
            ### old: self.neuron_model_parameters_dict
            ### old: self.neuron_model_attributes_dict = keys()
            self.neuron_model_attr_dict[pop.name] = pop.init
            ### get a dict of all arguments of the __init__ function of the Neuron
            ### ignore self
            ### old: self.neuron_model_dict[pop_name]
            init_params = inspect.signature(Neuron.__init__).parameters
            self.neuron_model_init_parameter_dict[pop.name] = {
                param: getattr(pop.neuron_type, param)
                for param in init_params
                if param != "self"
            }
            ### get a dict of all arguments of the __init__ function of the Population
            ### ignore self, storage_order and copied
            init_params = inspect.signature(Population.__init__).parameters
            self.pop_init_parameter_dict[pop.name] = {
                param: getattr(pop, param)
                for param in init_params
                if param != "self" and param != "storage_order" and param != "copied"
            }

    def _analyze_projections(self, model: CompNeuroModel):
        """
        Get info of each projection

        Args:
            model (CompNeuroModel):
                Model to be analyzed
        """
        sf.Logger().log(
            "Analyzing projections to be able to reproduce them",
            verbose=_model_configurator_verbose,
        )
        ### parameters of the __init__ function of the Projection class
        self.proj_init_parameter_dict: dict[str, dict] = {}
        ### parameters of the __init__ function of the Synapse class
        self.synapse_init_parameter_dict: dict[str, dict] = {}
        ### values of the paramters and variables of the synapse, keys are the names of
        ### paramters and variables
        self.synapse_model_attr_dict: dict[str, dict] = {}
        ### connector functions of the projections
        self.connector_function_dict: dict = {}
        ### parameters of the connector functions of the projections
        self.connector_function_parameter_dict: dict = {}
        ### names of pre- and post-synaptic populations of the projections
        ### old: self.post_pop_name_dict and self.pre_pop_name_dict
        self.pre_post_pop_name_dict: dict[str, tuple] = {}
        ### sizes of pre- and post-synaptic populations of the projections
        ### old: self.pre_pop_size_dict
        self.pre_post_pop_size_dict: dict[str, tuple] = {}

        ### loop over all projections
        for proj_name in model.projections:
            sf.Logger().log(
                f"Analyzing projection {proj_name}", verbose=_model_configurator_verbose
            )
            proj: Projection = get_projection(proj_name)
            ### get the synapse model attributes (parameters/variables)
            self.synapse_model_attr_dict[proj.name] = proj.init
            ### get a dict of all paramters of the __init__ function of the Synapse
            init_params = inspect.signature(Synapse.__init__).parameters
            self.synapse_init_parameter_dict[proj.name] = {
                param: getattr(proj.synapse_type, param)
                for param in init_params
                if param != "self"
            }
            ### get a dict of all paramters of the __init__ function of the Projection
            init_params = inspect.signature(Projection.__init__).parameters
            self.proj_init_parameter_dict[proj_name] = {
                param: getattr(proj, param)
                for param in init_params
                if param != "self" and param != "synapse" and param != "copied"
            }

            ### get the connector function of the projection and its parameters
            ### raise errors for not supported connector functions
            if (
                proj.connector_name == "User-defined"
                or proj.connector_name == "MatrixMarket"
                or proj.connector_name == "From File"
            ):
                raise ValueError(
                    f"Connector function '{self._connector_methods_dict[proj.connector_name].__name__}' not supported yet"
                )

            ### get the connector function
            self.connector_function_dict[proj.name] = self._connector_methods_dict[
                proj.connector_name
            ]

            ### get the parameters of the connector function
            self.connector_function_parameter_dict[proj.name] = (
                self._get_connector_parameters(proj)
            )

            ### get the names of the pre- and post-synaptic populations
            self.pre_post_pop_name_dict[proj.name] = (proj.pre.name, proj.post.name)

            ### get the sizes of the pre- and post-synaptic populations
            self.pre_post_pop_size_dict[proj.name] = (
                proj.pre.size,
                proj.post.size,
            )

    def _get_connector_parameters(self, proj: Projection):
        """
        Get the parameters of the given connector function.

        Args:
            proj (Projection):
                Projection for which the connector parameters are needed

        Returns:
            connector_parameters_dict (dict):
                Parameters of the given connector function
        """

        if proj.connector_name == "One-to-One":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "All-to-All":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "allow_self_connections": proj._connection_args[2],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Gaussian":
            return {
                "amp": proj._connection_args[0],
                "sigma": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "limit": proj._connection_args[3],
                "allow_self_connections": proj._connection_args[4],
                "storage_format": proj._storage_format,
            }
        elif proj.connector_name == "Difference-of-Gaussian":
            return {
                "amp_pos": proj._connection_args[0],
                "sigma_pos": proj._connection_args[1],
                "amp_neg": proj._connection_args[2],
                "sigma_neg": proj._connection_args[3],
                "delays": proj._connection_args[4],
                "limit": proj._connection_args[5],
                "allow_self_connections": proj._connection_args[6],
                "storage_format": proj._storage_format,
            }
        elif proj.connector_name == "Random":
            return {
                "probability": proj._connection_args[0],
                "weights": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "allow_self_connections": proj._connection_args[3],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Random Convergent":
            return {
                "number": proj._connection_args[0],
                "weights": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "allow_self_connections": proj._connection_args[3],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Random Divergent":
            return {
                "number": proj._connection_args[0],
                "weights": proj._connection_args[1],
                "delays": proj._connection_args[2],
                "allow_self_connections": proj._connection_args[3],
                "force_multiple_weights": not (proj._single_constant_weight),
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Connectivity matrix":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "pre_post": proj._connection_args[2],
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }
        elif proj.connector_name == "Sparse connectivity matrix":
            return {
                "weights": proj._connection_args[0],
                "delays": proj._connection_args[1],
                "storage_format": proj._storage_format,
                "storage_order": proj._storage_order,
            }


class CreateReducedModel:
    """
    Class to create a reduced model from the original model. It is accessable via the
    attribute model_reduced.

    Attributes:
        model_reduced (CompNeuroModel):
            Reduced model, created but not compiled
    """

    def __init__(
        self,
        model: CompNeuroModel,
        analyze_model: AnalyzeModel,
        reduced_size: int,
        do_create: bool = False,
        do_compile: bool = False,
        verbose: bool = False,
    ) -> None:
        """
        Prepare model for reduction.

        Args:
            model (CompNeuroModel):
                Model to be reduced
            reduced_size (int):
                Size of the reduced populations
        """
        ### set the attributes
        self._model = model
        self._analyze_model = analyze_model
        self._reduced_size = reduced_size
        self._verbose = verbose
        ### recreate model with reduced populations and projections
        sf.Logger().log(
            "Initializing CompNeuroModel with reduced populations and projections.",
            verbose=self._verbose,
        )
        self.model_reduced = CompNeuroModel(
            model_creation_function=self.recreate_model,
            name=f"{model.name}_reduced",
            description=f"{model.description}\nWith reduced populations and projections.",
            do_create=do_create,
            do_compile=do_compile,
            compile_folder_name=f"{model.compile_folder_name}_reduced",
        )

    def set_weights(self, weight_dict: dict[str, float]):
        """
        Set the weights of the reduced model.

        Args:
            weight_dict (dict[str, float]):
                Dict with the weights for each projection of the original model
        """
        ### return if the model was not created yet
        if not self.model_reduced.created:
            print("Warning: Model was not created yet. Cannot set weights.")
            return
        ### loop over all projections with a given weight
        for proj_name, weight in weight_dict.items():
            ### get the name of the post population
            post_pop_name = self._analyze_model.pre_post_pop_name_dict[proj_name][1]
            ### get the target type of the projection
            target_type = self._analyze_model.proj_init_parameter_dict[proj_name][
                "target"
            ]
            ### set the weight in the conductance-calculating input current population
            setattr(
                get_population(f"{post_pop_name}_{target_type}_aux"),
                f"weights_{proj_name}",
                weight,
            )

    def recreate_model(self):
        """
        Recreates the model with reduced populations and projections.
        """
        ### 1st for each population create a reduced population
        for pop_name in self._model.populations:
            self.create_reduced_pop(pop_name)
        ### 2nd for each population which is a presynaptic population, create a spikes collecting aux population
        for pop_name in self._model.populations:
            self.create_spike_collecting_aux_pop(pop_name)
        ## 3rd for each population which has afferents create a population for incoming spikes for each target type
        for pop_name in self._model.populations:
            self.create_conductance_aux_pop(pop_name, target="ampa")
            self.create_conductance_aux_pop(pop_name, target="gaba")

    def create_reduced_pop(self, pop_name: str):
        """
        Create a reduced population from the given population.

        Args:
            pop_name (str):
                Name of the population to be reduced
        """
        sf.Logger().log(
            f"Create reduced population for {pop_name}.", verbose=self._verbose
        )
        ### 1st check how the population is connected
        _, is_postsynaptic, ampa, gaba = self.how_pop_is_connected(pop_name)

        ### 2nd recreate neuron model
        ### get the stored parameters of the __init__ function of the Neuron
        neuron_model_init_parameter_dict = (
            self._analyze_model.neuron_model_init_parameter_dict[pop_name].copy()
        )
        ### if the population is a postsynaptic population adjust the synaptic
        ### conductance equations
        if is_postsynaptic:
            neuron_model_init_parameter_dict = self.adjust_neuron_model(
                neuron_model_init_parameter_dict, ampa=ampa, gaba=gaba
            )
        ### create the new neuron model
        neuron_model_new = Neuron(**neuron_model_init_parameter_dict)

        ### 3rd recreate the population
        ### get the stored parameters of the __init__ function of the Population
        pop_init_parameter_dict = self._analyze_model.pop_init_parameter_dict[
            pop_name
        ].copy()
        ### replace the neuron model with the new neuron model
        pop_init_parameter_dict["neuron"] = neuron_model_new
        ### replace the size with the reduced size (if reduced size is smaller than the
        ### original size)
        ### TODO add model requirements somewhere, here requirements = geometry = int
        pop_init_parameter_dict["geometry"] = min(
            [pop_init_parameter_dict["geometry"][0], self._reduced_size]
        )
        ### append _reduce to the name
        pop_init_parameter_dict["name"] = f"{pop_name}_reduced"
        ### create the new population
        pop_new = Population(**pop_init_parameter_dict)

        ### 4th set the parameters and variables of the population's neurons
        ### get the stored parameters and variables
        neuron_model_attr_dict = self._analyze_model.neuron_model_attr_dict[pop_name]
        ### set the parameters and variables
        for attr_name, attr_val in neuron_model_attr_dict.items():
            setattr(pop_new, attr_name, attr_val)

    def create_spike_collecting_aux_pop(self, pop_name: str):
        """
        Create a spike collecting population for the given population.

        Args:
            pop_name (str):
                Name of the population for which the spike collecting population should be created
        """
        ### get all efferent projections
        efferent_projection_list = [
            proj_name
            for proj_name, pre_post_pop_name_dict in self._analyze_model.pre_post_pop_name_dict.items()
            if pre_post_pop_name_dict[0] == pop_name
        ]
        ### check if pop has efferent projections
        if len(efferent_projection_list) == 0:
            return
        sf.Logger().log(
            f"Create spike collecting aux population for {pop_name}.",
            verbose=self._verbose,
        )
        ### create the spike collecting population
        pop_aux = Population(
            1,
            neuron=self.SpikeProbCalcNeuron(
                reduced_size=get_population(f"{pop_name}_reduced").geometry[0],
            ),
            name=f"{pop_name}_spike_collecting_aux",
        )
        ### create the projection from reduced pop to spike collecting aux pop
        sf.Logger().log(
            f"Create projection from {pop_name}_reduced to {pop_name}_spike_collecting_aux to collect spikes.",
            verbose=self._verbose,
        )
        proj = Projection(
            pre=get_population(f"{pop_name}_reduced"),
            post=pop_aux,
            target="ampa",
            name=f"proj_{pop_name}_spike_collecting_aux",
        )
        proj.connect_all_to_all(weights=1)

    def create_conductance_aux_pop(self, pop_name: str, target: str):
        """
        Create a conductance calculating population for the given population and target.

        Args:
            pop_name (str):
                Name of the population for which the conductance calculating population should be created
            target (str):
                Target type of the conductance calculating population
        """
        ### get all afferent projections
        afferent_projection_list = [
            proj_name
            for proj_name, pre_post_pop_name_dict in self._analyze_model.pre_post_pop_name_dict.items()
            if pre_post_pop_name_dict[1] == pop_name
        ]
        ### check if pop has afferent projections
        if len(afferent_projection_list) == 0:
            return
        ### get all afferent projections with target type
        afferent_target_projection_list = [
            proj_name
            for proj_name in afferent_projection_list
            if self._analyze_model.proj_init_parameter_dict[proj_name]["target"]
            == target
        ]
        ### check if there are afferent projections with target type
        if len(afferent_target_projection_list) == 0:
            return
        sf.Logger().log(
            f"Create conductance calculating aux population for {pop_name} target {target}.",
            verbose=self._verbose,
        )
        ### get projection informations
        ### TODO somewhere add model requirements, here requirements = geometry = int and connection = fixed_probability i.e. random (with weights and probability)
        projection_dict = {
            proj_name: {
                "pre_size": self._analyze_model.pop_init_parameter_dict[
                    self._analyze_model.pre_post_pop_name_dict[proj_name][0]
                ]["geometry"][0],
                "connection_prob": self._analyze_model.connector_function_parameter_dict[
                    proj_name
                ][
                    "probability"
                ],
                "weights": self._analyze_model.connector_function_parameter_dict[
                    proj_name
                ]["weights"],
                "pre_name": self._analyze_model.pre_post_pop_name_dict[proj_name][0],
            }
            for proj_name in afferent_target_projection_list
        }
        ### create the conductance calculating population
        pop_aux = Population(
            get_population(f"{pop_name}_reduced").geometry[0],
            neuron=self.InputCalcNeuron(projection_dict=projection_dict),
            name=f"{pop_name}_{target}_aux",
        )
        ### set number of synapses parameter for each projection
        for proj_name, vals in projection_dict.items():
            number_synapses = Binomial(
                n=vals["pre_size"], p=vals["connection_prob"]
            ).get_values(get_population(f"{pop_name}_reduced").geometry[0])
            setattr(pop_aux, f"number_synapses_{proj_name}", number_synapses)
        ### create the "current injection" projection from conductance calculating
        ### population to the reduced post population
        sf.Logger().log(
            f"Create projection from {pop_name}_{target}_aux to {pop_name}_reduced to forward the calculated conductance to the post population ({pop_name}_reduced)",
            verbose=self._verbose,
        )
        proj = CurrentInjection(
            pre=pop_aux,
            post=get_population(f"{pop_name}_reduced"),
            target=f"incomingaux{target}",
            name=f"proj_{pop_name}_{target}_aux",
        )
        proj.connect_current()
        ### create projection from spike_prob calculating aux neurons of presynaptic
        ### populations to conductance calculating aux population
        sf.Logger().log(
            f"Create projection from spike collecting, i.e. spike probability calculating, aux populations of presynaptic populations of {pop_name}_reduced to the conductance calculating population '{pop_name}_{target}_aux'.",
            verbose=self._verbose,
        )
        for proj_name, vals in projection_dict.items():
            pre_pop_name = vals["pre_name"]
            pre_pop_spike_collecting_aux = get_population(
                f"{pre_pop_name}_spike_collecting_aux"
            )
            proj = Projection(
                pre=pre_pop_spike_collecting_aux,
                post=pop_aux,
                target=f"spikeprob_{pre_pop_name}",
                name=f"{proj_name}_spike_collecting_to_conductance",
            )
            proj.connect_all_to_all(weights=1)

    def how_pop_is_connected(self, pop_name):
        """
        Check how a population is connected. If the population is a postsynaptic and/or
        presynaptic population, check if it gets ampa and/or gaba input.

        Args:
            pop_name (str):
                Name of the population to check

        Returns:
            is_presynaptic (bool):
                True if the population is a presynaptic population, False otherwise
            is_postsynaptic (bool):
                True if the population is a postsynaptic population, False otherwise
            ampa (bool):
                True if the population gets ampa input, False otherwise
            gaba (bool):
                True if the population gets gaba input, False otherwise
        """
        is_presynaptic = False
        is_postsynaptic = False
        ampa = False
        gaba = False
        ### loop over all projections
        for proj_name in self._model.projections:
            ### check if the population is a presynaptic population in any projection
            if self._analyze_model.pre_post_pop_name_dict[proj_name][0] == pop_name:
                is_presynaptic = True
            ### check if the population is a postsynaptic population in any projection
            if self._analyze_model.pre_post_pop_name_dict[proj_name][1] == pop_name:
                is_postsynaptic = True
                ### check if the projection target is ampa or gaba
                if (
                    self._analyze_model.proj_init_parameter_dict[proj_name]["target"]
                    == "ampa"
                ):
                    ampa = True
                elif (
                    self._analyze_model.proj_init_parameter_dict[proj_name]["target"]
                    == "gaba"
                ):
                    gaba = True

        return is_presynaptic, is_postsynaptic, ampa, gaba

    def adjust_neuron_model(
        self, neuron_model_init_parameter_dict, ampa=True, gaba=True
    ):
        """
        Add the new synaptic input coming from the auxillary population to the neuron
        model.

        Args:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron
            ampa (bool):
                True if the population gets ampa input and therefore the ampa conductance
                needs to be adjusted, False otherwise
            gaba (bool):
                True if the population gets gaba input and therefore the gaba conductance
                needs to be adjusted, False otherwise

        Returns:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron
                with DBS mechanisms added
        """
        ### 1st adjust the conductance equations
        ### get the equations of the neuron model as a list of strings
        equations_line_split_list = str(
            neuron_model_init_parameter_dict["equations"]
        ).splitlines()
        ### search for equation with dg_ampa/dt and dg_gaba/dt
        for line_idx, line in enumerate(equations_line_split_list):
            if (
                self.get_line_is_dvardt(line, var_name="g_ampa", tau_name="tau_ampa")
                and ampa
            ):
                ### add " + tau_ampa*g_incomingauxampa/dt"
                ### TODO add model requirements somewhere, here requirements = tau_ampa * dg_ampa/dt = -g_ampa
                equations_line_split_list[line_idx] = (
                    "tau_ampa*dg_ampa/dt = -g_ampa + tau_ampa*g_incomingauxampa/dt"
                )
            if (
                self.get_line_is_dvardt(line, var_name="g_gaba", tau_name="tau_gaba")
                and gaba
            ):
                ### add " + tau_gaba*g_incomingauxgaba/dt"
                ### TODO add model requirements somewhere, here requirements = tau_gaba * dg_gaba/dt = -g_gaba
                equations_line_split_list[line_idx] = (
                    "tau_gaba*dg_gaba/dt = -g_gaba + tau_gaba*g_incomingauxgaba/dt"
                )
        ### join list to a string
        neuron_model_init_parameter_dict["equations"] = "\n".join(
            equations_line_split_list
        )

        ### 2nd extend description
        neuron_model_init_parameter_dict["description"] = (
            f"{neuron_model_init_parameter_dict['description']}\nWith incoming auxillary population input implemented."
        )

        return neuron_model_init_parameter_dict

    def get_line_is_dvardt(self, line: str, var_name: str, tau_name: str):
        """
        Check if a equation string has the form "tau*dvar/dt = -var".

        Args:
            line (str):
                Equation string
            var_name (str):
                Name of the variable
            tau_name (str):
                Name of the time constant

        Returns:
            is_solution_correct (bool):
                True if the equation is as expected, False otherwise
        """
        if var_name not in line:
            return False

        # Define the variables
        var, _, _, _ = sp.symbols(f"{var_name} d{var_name} dt {tau_name}")

        # Given equation as a string
        equation_str = line

        # Parse the equation string
        lhs, rhs = equation_str.split("=")
        lhs = sp.sympify(lhs)
        rhs = sp.sympify(rhs)

        # Form the equation
        equation = sp.Eq(lhs, rhs)

        # Solve the equation for var
        try:
            solution = sp.solve(equation, var)
        except:
            ### equation is not solvable with variables means it is not as expected
            return False

        # Given solution to compare
        expected_solution_str = f"-{tau_name}*d{var_name}/dt"
        expected_solution = sp.sympify(expected_solution_str)

        # Check if the solution is as expected
        is_solution_correct = solution[0] == expected_solution

        return is_solution_correct

    class SpikeProbCalcNeuron(Neuron):
        """
        Neuron model to calculate the spike probabilities of the presynaptic neurons.
        """

        def __init__(self, reduced_size=1):
            """
            Args:
                reduced_size (int):
                    Reduced size of the associated presynaptic population
            """
            parameters = f"""
                reduced_size = {reduced_size} : population
                tau= 1.0 : population
            """
            equations = """
                tau*dr/dt = g_ampa/reduced_size - r
                g_ampa = 0
            """
            super().__init__(parameters=parameters, equations=equations)

    class InputCalcNeuron(Neuron):
        """
        This neurons gets the spike probabilities of the pre neurons and calculates the
        incoming spikes for each projection. It accumulates the incoming spikes of all
        projections (of the same target type) and calculates the conductance increase
        for the post neuron.
        """

        def __init__(self, projection_dict):
            """
            Args:
                projection_dict (dict):
                    keys: names of afferent projections (of the same target type)
                    values: dict with keys "weights", "pre_name"
            """

            ### create parameters
            parameters = [
                f"""
                number_synapses_{proj_name} = 0
                weights_{proj_name} = {vals['weights']}
            """
                for proj_name, vals in projection_dict.items()
            ]
            parameters = "\n".join(parameters)

            ### create equations
            equations = [
                f"""
                incoming_spikes_{proj_name} = round(number_synapses_{proj_name} * sum(spikeprob_{vals['pre_name']}) + Normal(0, 1)*sqrt(number_synapses_{proj_name} * sum(spikeprob_{vals['pre_name']}) * (1 - sum(spikeprob_{vals['pre_name']})))) : min=0, max=number_synapses_{proj_name}
            """
                for proj_name, vals in projection_dict.items()
            ]
            equations = "\n".join(equations)
            sum_of_conductance_increase = (
                "r = "
                + "".join(
                    [
                        f"incoming_spikes_{proj_name} * weights_{proj_name} + "
                        for proj_name in projection_dict.keys()
                    ]
                )[:-3]
            )
            equations = equations + "\n" + sum_of_conductance_increase

            super().__init__(parameters=parameters, equations=equations)


class CreateSingleNeuronNetworks:
    """
    Class to create single neuron networks for normal and voltage clamp mode.

    Attributes:
        single_net_dict (dict):
            Nested dict containing the single neuron networks for normal and voltage
            clamp mode
            keys: mode (str)
                normal or v_clamp
            values: dict
                keys: pop_name (str)
                    population name
                values: dict
                    keys: net, population, monitor, init_sampler
                    values: Network, Population, Monitor, ArrSampler
    """

    def __init__(
        self,
        model: CompNeuroModel,
        analyze_model: AnalyzeModel,
        do_not_config_list: list[str],
    ):
        """
        Args:
            model (CompNeuroModel):
                Model to be analyzed
            analyze_model (AnalyzeModel):
                Analyzed model
            do_not_config_list (list[str]):
                List of population names which should not be configured
        """
        sf.Logger().log(
            "Creating single neuron networks for normal and voltage clamp mode.",
            verbose=_model_configurator_verbose,
        )
        self._single_net_dict = {}
        ### create the single neuron networks for normal and voltage clamp mode
        for mode in ["normal", "v_clamp"]:
            self._single_net_dict[mode] = {}
            self._create_single_neuron_networks(
                model=model,
                analyze_model=analyze_model,
                do_not_config_list=do_not_config_list,
                mode=mode,
            )

    def single_net(self, pop_name: str, mode: str):
        """
        Return the information of the single neuron network for the given population and
        mode.

        Args:
            pop_name (str):
                Name of the population
            mode (str):
                Mode for which the single neuron network should be returned (normal or
                v_clamp)

        Returns:
            ReturnSingleNeuronNetworks:
                Information of the single neuron network with Attributes: net,
                population, monitor, init_sampler
        """
        return self.ReturnSingleNeuronNetworks(self._single_net_dict[mode][pop_name])

    class ReturnSingleNeuronNetworks:
        def __init__(self, single_net_dict):
            self.net: Network = single_net_dict["net"]
            self.population: Population = single_net_dict["population"]
            self.monitor: Monitor = single_net_dict["monitor"]
            self.init_sampler: ArrSampler = single_net_dict["init_sampler"]

    def init_sampler(self, model: CompNeuroModel, do_not_config_list: list[str]):
        """
        Return the init samplers for all populations of the normal mode. All samplers
        are returned in an object with a get method to get the sampler for a specific
        population.

        Args:
            model (CompNeuroModel):
                Model to be analyzed
            do_not_config_list (list[str]):
                List of population names which should not be configured

        Returns:
            AllSampler:
                Object with a get method to get the init sampler for a specific
                population
        """
        init_sampler_dict = {}
        for pop_name in model.populations:
            if pop_name in do_not_config_list:
                continue
            init_sampler_dict[pop_name] = self._single_net_dict["normal"][pop_name][
                "init_sampler"
            ]
        return self.AllSampler(init_sampler_dict)

    class AllSampler:
        def __init__(self, init_sampler_dict: dict[str, "ArrSampler"]):
            self.init_sampler_dict = init_sampler_dict

        def get(self, pop_name: str):
            """
            Get the init sampler for the given population.

            Args:
                pop_name (str):
                    Name of the population

            Returns:
                sampler (ArrSampler):
                    Init sampler for the given population
            """
            sampler: ArrSampler = self.init_sampler_dict[pop_name]
            return sampler

    def _create_single_neuron_networks(
        self,
        model: CompNeuroModel,
        analyze_model: AnalyzeModel,
        do_not_config_list: list[str],
        mode: str,
    ):
        """
        Create the single neuron networks for the given mode. Sets the single_net_dict.

        Args:
            model (CompNeuroModel):
                Model to be analyzed
            analyze_model (AnalyzeModel):
                Analyzed model
            do_not_config_list (list[str]):
                List of population names which should not be configured
            mode (str):
                Mode for which the single neuron networks should be created
        """
        sf.Logger().log(
            f"Creating single neuron networks for {mode} mode.",
            verbose=_model_configurator_verbose,
        )

        ### loop over populations which should be configured
        for pop_name in model.populations:
            ### skip populations which should not be configured
            if pop_name in do_not_config_list:
                continue
            ### store the dict containing the network etc
            self._single_net_dict[mode][pop_name] = self._create_net_single(
                pop_name=pop_name, analyze_model=analyze_model, mode=mode
            )

    def _create_net_single(self, pop_name: str, analyze_model: AnalyzeModel, mode: str):
        """
        Creates a network with the neuron type of the population given by pop_name for
        the given mode. The population size is set to 1.

        Args:
            pop_name (str):
                Name of the population
            analyze_model (AnalyzeModel):
                Analyzed model
            mode (str):
                Mode for which the network should be created

        Returns:
            net_single_dict (dict):
                Dict containing the Network, Population, Monitor and ArrSampler objects
        """
        sf.Logger().log(
            f"[{pop_name}]: Create single neuron network in {mode} mode.",
            verbose=_model_configurator_verbose,
        )
        ### create the adjusted neuron model for the stop condition
        sf.Logger().log(
            f"[{pop_name}]: Create adjusted neuron model.",
            verbose=_model_configurator_verbose,
        )
        neuron_model_new = self._get_single_neuron_neuron_model(
            pop_name=pop_name, analyze_model=analyze_model, mode=mode
        )

        ### create the single neuron population
        sf.Logger().log(
            f"[{pop_name}]: Create single neuron population.",
            verbose=_model_configurator_verbose,
        )
        pop_single_neuron = self._get_single_neuron_population(
            pop_name=pop_name,
            neuron_model_new=neuron_model_new,
            analyze_model=analyze_model,
            mode=mode,
        )

        ### create Monitor for single neuron
        sf.Logger().log(
            f"[{pop_name}]: Create Monitor for single neuron population.",
            verbose=_model_configurator_verbose,
        )
        if mode == "normal":
            mon_single = Monitor(pop_single_neuron, ["spike", "v"])
        elif mode == "v_clamp":
            mon_single = Monitor(pop_single_neuron, ["v_clamp_rec_sign"])

        ### create network with single neuron and compile it
        sf.Logger().log(
            f"[{pop_name}]: Create Network with single neuron population and its monitor and compile it.",
            verbose=_model_configurator_verbose,
        )
        net_single = Network()
        net_single.add([pop_single_neuron, mon_single])
        mf.compile_in_folder(
            folder_name=f"single_net_{mode}_{pop_name}", silent=True, net=net_single
        )

        ### network dict
        net_single_dict = {
            "net": net_single,
            "population": net_single.get(pop_single_neuron),
            "monitor": net_single.get(mon_single),
            "init_sampler": None,
        }

        ### for v_clamp we are done here
        if mode == "v_clamp":
            return net_single_dict

        ### for normal neuron get the init sampler for the variables of the neuron model
        ### (to initialize a population of the neuron model)
        sf.Logger().log(
            f"[{pop_name}]: Create init sampler for single neuron population.",
            verbose=_model_configurator_verbose,
        )
        init_sampler = self._get_neuron_model_init_sampler(
            net=net_single, pop=net_single.get(pop_single_neuron)
        )
        net_single_dict["init_sampler"] = init_sampler

        return net_single_dict

    def _get_single_neuron_neuron_model(
        self, pop_name: str, analyze_model: AnalyzeModel, mode=str
    ):
        """
        Create the adjusted neuron model for the given mode.

        Args:
            pop_name (str):
                Name of the population
            analyze_model (AnalyzeModel):
                Analyzed model
            mode (str):
                Mode for which the neuron model should be created

        Returns:
            neuron_model_new (Neuron):
                Adjusted neuron model
        """
        ### get the stored parameters of the __init__ function of the Neuron
        neuron_model_init_parameter_dict = (
            analyze_model.neuron_model_init_parameter_dict[pop_name].copy()
        )
        ### Define the attributes of the neuron model as sympy symbols
        neuron_model_attributes_name_list = list(
            analyze_model.neuron_model_attr_dict[pop_name].keys()
        )
        ### add v_before_psp and v_psp_thresh to equations/parameters, for the stop
        ### condition below
        self._adjust_neuron_model(
            neuron_model_init_parameter_dict,
            neuron_model_attributes_name_list,
            mode=mode,
        )
        ### create the adjusted neuron model
        neuron_model_new = Neuron(**neuron_model_init_parameter_dict)
        return neuron_model_new

    def _get_single_neuron_population(
        self,
        pop_name: str,
        neuron_model_new: Neuron,
        analyze_model: AnalyzeModel,
        mode: str,
    ):
        """
        Create the single neuron population for the given mode.

        Args:
            pop_name (str):
                Name of the population
            neuron_model_new (Neuron):
                Adjusted neuron model
            analyze_model (AnalyzeModel):
                Analyzed model
            mode (str):
                Mode for which the population should be created

        Returns:
            pop_single_neuron (Population):
                Single neuron population
        """
        if mode == "normal":
            pop_single_neuron = Population(
                1,
                neuron=neuron_model_new,
                name=f"single_neuron_{pop_name}",
                stop_condition="((abs(v-v_psp_thresh)<0.01) and (abs(v_before_psp-v_psp_thresh)>0.01)): any",
            )
        elif mode == "v_clamp":
            ### create the single neuron population
            pop_single_neuron = Population(
                1,
                neuron=neuron_model_new,
                name=f"single_neuron_v_clamp_{pop_name}",
            )

        ### get the stored parameters and variables
        neuron_model_attr_dict = analyze_model.neuron_model_attr_dict[pop_name]
        ### set the parameters and variables
        for attr_name, attr_val in neuron_model_attr_dict.items():
            setattr(pop_single_neuron, attr_name, attr_val)
        return pop_single_neuron

    def _get_neuron_model_init_sampler(self, net: Network, pop: Population):
        """
        Create a sampler for the initial values of the variables of the neuron model by
        simulating the neuron for 10000 ms and afterwards simulating 2000 ms and
        sampling the variables every dt.

        Args:
            net (Network):
                Network with the single neuron population
            pop (Population):
                Single neuron population

        Returns:
            sampler (ArrSampler):
                Sampler for the initial values of the variables of the neuron model
        """

        ### reset network and deactivate input
        net.reset()
        pop.I_app = 0

        ### 10000 ms init duration
        net.simulate(10000)

        ### simulate 2000 ms and check every dt the variables of the neuron
        time_steps = int(2000 / dt())
        var_name_list = list(pop.variables)
        var_arr = np.zeros((time_steps, len(var_name_list)))
        for time_idx in range(time_steps):
            net.simulate(dt())
            get_arr = np.array([getattr(pop, var_name) for var_name in pop.variables])
            var_arr[time_idx, :] = get_arr[:, 0]

        ### reset network after simulation
        net.reset()

        ### create a sampler with the data samples from the21000 ms simulation
        sampler = ArrSampler(arr=var_arr, var_name_list=var_name_list)
        return sampler

    def _adjust_neuron_model(
        self,
        neuron_model_init_parameter_dict: dict,
        neuron_model_attributes_name_list: list[str],
        mode: str,
    ):
        """
        Adjust the parameters and equations of the neuron model for the given mode.

        Args:
            neuron_model_init_parameter_dict (dict):
                Dict with the parameters and equations of the neuron model
            neuron_model_attributes_name_list (list[str]):
                List of the names of the attributes of the neuron model
            mode (str):
                Mode for which the neuron model should be adjusted
        """
        ### get the equations of the neuron model as a list of strings
        equations_line_split_list = str(
            neuron_model_init_parameter_dict["equations"]
        ).splitlines()
        ### get the parameters of the neuron model as a list of strings
        parameters_line_split_list = str(
            neuron_model_init_parameter_dict["parameters"]
        ).splitlines()

        if mode == "normal":
            ### add v_before_psp=v at the beginning of the equations
            equations_line_split_list.insert(0, "v_before_psp = v")
            ### add v_psp_thresh to the parameters
            parameters_line_split_list.append("v_psp_thresh = 0 : population")
        elif mode == "v_clamp":
            ### get new equations for voltage clamp
            equations_new_list = self.CreateVoltageClampEquations(
                equations_line_split_list, neuron_model_attributes_name_list
            ).eq_new
            neuron_model_init_parameter_dict["equations"] = equations_new_list
            ### add v_clamp_rec_thresh to the parameters
            parameters_line_split_list.append("v_clamp_rec_thresh = 0 : population")

        ### join equations and parameters to a string and store them in the dict
        neuron_model_init_parameter_dict["equations"] = "\n".join(
            equations_line_split_list
        )
        neuron_model_init_parameter_dict["parameters"] = "\n".join(
            parameters_line_split_list
        )

    class CreateVoltageClampEquations:
        """
        Class to create voltage clamp equations from the given equations of a neuron model.
        The equations of the neuron model have to contain the voltage change equation in the
        form of ... dv/dt ... = ...

        Attributes:
            eq_new (list[str])
                new equations of the neuron model with the voltage clamp
        """

        def __init__(self, eq: list[str], neuron_model_attributes_name_list: list[str]):
            """
            Args:
                eq (list[str])
                    equations of the neuron model
                neuron_model_attributes_name_list (list[str])
                    list of the names of the attributes of the neuron model
            """
            ### get the dv/dt equation from equations
            eq_v, eq_v_idx = self.get_eq_v(eq=eq)

            ### prepare the equation string for solving
            ### TODO replace random distributions and mathematical expressions which may be on the left side
            eq_v, tags = self.prepare_eq_v(eq_v=eq_v)

            ### solve equation to delta_v (which is dv/dt)
            result = self.solve_delta_v(eq_v, neuron_model_attributes_name_list)

            ### insert the new equation lines for v_clamp and remove the old dv/dt line
            self.eq_new = self.replace_delta_v(
                result=result, eq=eq, eq_v_idx=eq_v_idx, tags=tags
            )

        def replace_delta_v(
            self, result: str, eq: list[str], eq_v_idx: int, tags: str = None
        ):
            """
            Replace the dv/dt line with the voltage clamp lines.

            Args:
                result (str)
                    right side of the dv/dt equation
                eq (list[str])
                    equations of the neuron model
                eq_v_idx (int)
                    index of the dv/dt line
                tags (str)
                    tags of the dv/dt line

            Returns:
                eq (list[str])
                    new equations of the neuron model with the voltage clamp
            """
            ### create the line for recording voltage clamp (right side of dv/dt)
            eq_new_0 = f"v_clamp_rec_sign = {result}"
            ### create the line for the absolute voltage clamp
            eq_new_1 = f"v_clamp_rec = fabs({result})"
            ### create the line for the absolute voltage clamp from the previous time step
            eq_new_2 = "v_clamp_rec_pre = v_clamp_rec"
            ### create the voltage clamp line "dv/dt=0" with tags if they exist
            if not isinstance(tags, type(None)):
                eq_new_3 = f"dv/dt=0 : {tags}"
            else:
                eq_new_3 = "dv/dt=0"
            ### remove old v line and insert new three lines, order is important
            del eq[eq_v_idx]
            eq.insert(eq_v_idx, eq_new_0)
            eq.insert(eq_v_idx, eq_new_1)
            eq.insert(eq_v_idx, eq_new_2)
            eq.insert(eq_v_idx, eq_new_3)
            ### return new neuron equations
            return eq

        def get_line_is_v(self, line: str):
            """
            Check if the line contains the definition of dv/dt.

            Args:
                line (str)
                    line of the equations of the neuron model

            Returns:
                line_is_v (bool)
                    True if the line contains the definition of dv/dt, False otherwise
            """
            if "v" not in line:
                return False

            ### remove whitespaces
            line = line.replace(" ", "")

            ### check for dv/dt
            if "dv/dt" in line:
                return True

            return False

        def get_eq_v(self, eq: list[str]):
            """
            Get the dv/dt equation from the equations of the neuron model.

            Args:
                eq (list[str])
                    equations of the neuron model

            Returns:
                eq_v (str)
                    dv/dt equation
                eq_v_idx (int)
                    index of the dv/dt line
            """
            ### get the dv/dt equation from equations
            ### find the line with dv/dt= or v+= or v=
            line_is_v_list = [False] * len(eq)
            ### check in which lines v is defined
            for line_idx, line in enumerate(eq):
                line_is_v_list[line_idx] = self.get_line_is_v(line)
            ### raise error if no v or multiple times v
            if True not in line_is_v_list or sum(line_is_v_list) > 1:
                raise ValueError(
                    "In the equations of the neurons has to be exactly a single line which defines dv/dt!"
                )
            ### get the index of the line with dv/dt
            eq_v_idx = line_is_v_list.index(True)
            ### set the v equation
            eq_v = eq.copy()[eq_v_idx]
            return eq_v, eq_v_idx

        def prepare_eq_v(self, eq_v: str):
            """
            Prepare the equation string for solving with sympy.

            Args:
                eq_v (str)
                    dv/dt equation

            Returns:
                eq_v (str)
                    dv/dt equation
                tags (str)
                    tags of the dv/dt equation
            """
            ### remove whitespaces
            eq_v = eq_v.replace(" ", "")
            ### replace dv/dt by delta_v
            eq_v = eq_v.replace("dv/dt", "delta_v")
            ### separate equation and tags
            eq_tags_list = eq_v.split(":")
            eq_v = eq_tags_list[0]
            if len(eq_tags_list) > 1:
                tags = eq_tags_list[1]
            else:
                tags = None
            return eq_v, tags

        def solve_delta_v(
            self, eq_v: str, neuron_model_attributes_name_list: list[str]
        ):
            """
            Solve the dv/dt equation for delta_v (which is dv/dt).

            Args:
                eq_v (str)
                    dv/dt equation
                neuron_model_attributes_name_list (list[str])
                    list of the names of the attributes of the neuron model

            Returns:
                solution_str (str)
                    right side of the dv/dt equation
            """
            ### Define the attributes of the neuron model as sympy symbols
            sp.symbols(",".join(neuron_model_attributes_name_list))
            ### Define delta_v and right_side as sympy symbols
            delta_v, _ = sp.symbols("delta_v right_side")

            ### Parse the equation string
            lhs, rhs_string = eq_v.split("=")
            lhs = sp.sympify(lhs)
            rhs = sp.sympify(rhs_string)

            ### Form the equation
            equation = sp.Eq(lhs, rhs)

            ### Solve the equation for delta_v
            try:
                solution = sp.solve(equation, delta_v)[0]
            except:
                raise ValueError("Could not find solution for dv/dt!")

            ### Get the solution as a string
            solution_str = str(solution)

            ### replace right_side by the actual right side string
            solution_str = solution_str.replace("right_side", f"({rhs_string})")

            return solution_str


class ArrSampler:
    """
    Class to store an array and sample from it.
    """

    def __init__(self, arr: np.ndarray, var_name_list: list[str]) -> None:
        """
        Args:
            arr (np.ndarray)
                array with shape (n_samples, n_variables)
            var_name_list (list[str])
                list of variable names
        """
        self.arr_shape = arr.shape
        self.var_name_list = var_name_list
        ### check values of any variable are constant
        self.is_const = np.std(arr, axis=0) <= np.mean(np.absolute(arr), axis=0) / 1000
        ### for the constant variables only the first value is used
        self.constant_arr = arr[0, self.is_const]
        ### array without the constant variables
        self.not_constant_val_arr = arr[:, np.logical_not(self.is_const)]

    def sample(self, n=1, seed=0):
        """
        Sample n samples from the array.

        Args:
            n (int)
                number of samples to be drawn
            seed (int)
                seed for the random number generator

        Returns:
            ret_arr (np.ndarray)
                array with shape (n, n_variables)
        """
        ### get n random indices along the n_samples axis
        rng = np.random.default_rng(seed=seed)
        random_idx_arr = rng.integers(low=0, high=self.arr_shape[0], size=n)
        ### sample with random idx
        sample_arr = self.not_constant_val_arr[random_idx_arr]
        ### create return array
        ret_arr = np.zeros((n,) + self.arr_shape[1:])
        ### add samples to return array
        ret_arr[:, np.logical_not(self.is_const)] = sample_arr
        ### add constant values to return array
        ret_arr[:, self.is_const] = self.constant_arr

        return ret_arr

    def set_init_variables(self, population: Population):
        """
        Set the initial variables of the given population to the given values.
        """
        variable_init_arr = self.sample(len(population), seed=0)
        var_name_list = self.var_name_list
        for var_name in population.variables:
            if var_name in var_name_list:
                set_val = variable_init_arr[:, var_name_list.index(var_name)]
                setattr(population, var_name, set_val)


class Simulator:
    """
    Class with simulations for the single neuron networks.
    """

    def __init__(
        self,
        single_nets: CreateSingleNeuronNetworks,
        prepare_psp: "PreparePSP" = None,
    ):
        """
        Args:
            single_nets (CreateSingleNeuronNetworks):
                Single neuron networks for normal and voltage clamp mode
            prepare_psp (PreparePSP):
                Prepare PSP information
        """
        if prepare_psp is None:
            sf.Logger().log(
                "Initialize Simulator for single neuron networks without prepared calculate-PSP simulations.",
                verbose=_model_configurator_verbose,
            )
        else:
            sf.Logger().log(
                "Initialize Simulator for single neuron networks with prepared calculate-PSP simulations.",
                verbose=_model_configurator_verbose,
            )
        self._single_nets = single_nets
        self._prepare_psp = prepare_psp

    def get_v_clamp_2000(
        self,
        pop_name: str,
        v: float | None = None,
        I_app: float | None = None,
    ) -> float:
        """
        Simulates the v_clamp single neuron network of the given pop_name for 2000 ms
        and returns the v_clamp_rec value of the single neuron after 2000 ms. The
        returned values is "dv/dt". Therefore, to get the hypothetical change of v for a
        single time step multiply it with dt!

        Args:
            pop_name (str):
                Name of the population
            v (float):
                Membrane potential (does not change over time due to voltage clamp)
            I_app (float):
                Applied current

        Returns:
            v_clamp_rec (float):
                v_clamp_rec value of the single neuron after 2000 ms
        """
        ### get the network, population, init_sampler
        net = self._single_nets.single_net(pop_name=pop_name, mode="v_clamp").net
        population = self._single_nets.single_net(
            pop_name=pop_name, mode="v_clamp"
        ).population
        init_sampler = self._single_nets.single_net(
            pop_name=pop_name, mode="v_clamp"
        ).init_sampler
        ### reset network
        net.reset()
        net.set_seed(0)
        ### set the initial variables of the neuron model
        if init_sampler is not None:
            init_sampler.set_init_variables(population)
        ### set v and I_app
        if v is not None:
            population.v = v
        if I_app is not None:
            population.I_app = I_app
        ### simulate 2000 ms
        net.simulate(2000)
        ### return the v_clamp_rec value of the single neuron after 2000 ms
        return population.v_clamp_rec[0]

    def get_v_2000(self, pop_name, initial_variables, I_app=None) -> np.ndarray:
        """
        Simulate normal single neuron 2000 ms and return v for this duration.

        Args:
            pop_name (str):
                Name of the population
            initial_variables (dict):
                Initial variables of the neuron model
            I_app (float):
                Applied current

        Returns:
            v_arr (np.array):
                Membrane potential for the 2000 ms simulation with shape: (time_steps,)
        """
        ### get the network, population, monitor
        net = self._single_nets.single_net(pop_name=pop_name, mode="normal").net
        population = self._single_nets.single_net(
            pop_name=pop_name, mode="normal"
        ).population
        monitor = self._single_nets.single_net(pop_name=pop_name, mode="normal").monitor
        ### reset network
        net.reset()
        net.set_seed(0)
        ### set the initial variables of the neuron model
        for var_name, var_val in initial_variables.items():
            if var_name in population.variables:
                setattr(population, var_name, var_val)
        ### set I_app
        if I_app is not None:
            population.I_app = I_app
        ### simulate
        net.simulate(2000)
        v_arr = monitor.get("v")[:, 0]

        if _model_configurator_do_plot:
            plt.close("all")
            plt.figure()
            plt.title(f"I_app = {population.I_app}")
            plt.plot(v_arr)
            plt.savefig(
                f"{_model_configurator_figure_folder}/v_rest_{pop_name}_{int(I_app*1000)}.png"
            )
            plt.close("all")

        return v_arr

    def get_v_psp(self, v_rest: float, I_app_hold: float, pop_name: str) -> np.ndarray:
        """
        Simulate the single neuron network of the given pop_name for 5000 ms and return
        the variables of the neuron model after 5000 ms.

        Args:
            v_rest (float):
                Resting potential
            I_app_hold (float):
                Applied current to hold the resting potential
            pop_name (str):
                Name of the population

        Returns:
            var_arr (np.array):
                Variables of the neuron model after 5000 ms with shape: (1, n_vars)
        """

        ### get the network, population, monitor
        net = self._single_nets.single_net(pop_name=pop_name, mode="normal").net
        population = self._single_nets.single_net(
            pop_name=pop_name, mode="normal"
        ).population
        ### reset network
        net.reset()
        net.set_seed(0)
        ### set the initial variables of the neuron model
        population.v = v_rest
        population.I_app = I_app_hold
        ### simulate
        net.simulate(5000)
        ### get the variables of the neuron after 5000 ms in the shape (1, n_vars)
        var_name_list = list(population.variables)
        var_arr = np.zeros((1, len(var_name_list)))
        get_arr = np.array(
            [getattr(population, var_name) for var_name in population.variables]
        )
        var_arr[0, :] = get_arr[:, 0]
        return var_arr

    def get_ipsp(
        self,
        pop_name: str,
        g_ampa: float = 0,
        g_gaba: float = 0,
    ):
        """
        Simulate the single neuron network of the given pop_name for max 5000 ms. The
        neuron is hold at the resting potential by setting the applied current to
        I_app_hold. Then the conductances g_ampa and g_gaba are applied (simulating a
        single incoming ampa/gaba spike). The maximum of the (negative) difference of
        the membrane potential and the resting potential is returned as the IPSP.

        Args:
            pop_name (str):
                Name of the population
            g_ampa (float):
                Conductance of the ampa synapse
            g_gaba (float):
                Conductance of the gaba synapse

        Returns:
            psp (float):
                Maximum of the (negative) difference of the membrane potential and the
                resting potential
        """
        ### get the network, population, monitor from single nets
        net = self._single_nets.single_net(pop_name=pop_name, mode="normal").net
        population = self._single_nets.single_net(
            pop_name=pop_name, mode="normal"
        ).population
        monitor = self._single_nets.single_net(pop_name=pop_name, mode="normal").monitor
        ### get init_sampler, I_app_hold from prepare_psp
        init_sampler = self._prepare_psp.get(pop_name=pop_name).psp_init_sampler
        I_app_hold = self._prepare_psp.get(pop_name=pop_name).I_app_hold
        ### reset network
        net.reset()
        net.set_seed(0)
        ### set the initial variables of the neuron model
        if init_sampler is not None:
            init_sampler.set_init_variables(population)
        ### set I_app (I_app_hold) to hold the resting potential
        population.I_app = I_app_hold
        ### simulate 50 ms initial duration
        net.simulate(50)
        ### get the current v and set it as v_psp_thresh for the population's stop
        ### condition
        v_rec_rest = population.v[0]
        population.v_psp_thresh = v_rec_rest
        ### apply given conductances --> changes v, causes psp
        population.g_ampa = g_ampa
        population.g_gaba = g_gaba
        ### simulate until v is near v_rec_rest again or until 5000 ms
        net.simulate_until(max_duration=5000, population=population)
        ### get v and spike dict to calculate psp
        v_rec = monitor.get("v")[:, 0]
        spike_dict = monitor.get("spike")
        ### if neuron spiked only check psps until spike time, otherwise until last
        ### (current) time step
        spike_timestep_list = spike_dict[0] + [net.get_current_step()]
        end_timestep = int(round(min(spike_timestep_list), 0))
        ### find ipsp
        ### 1st calculate difference of v and v_rest
        v_diff = v_rec[:end_timestep] - v_rec_rest
        ### clip diff between None and zero, only take negative values (ipsp)
        v_diff = np.clip(v_diff, None, 0)
        ### add a small value to the clipped values, thus only large enough negative
        ### values considered as ipsp
        v_diff = v_diff + 0.01
        ### get the minimum of the difference as ipsp
        psp = np.min(v_diff)
        ### multiply with -1 to get the positive value of the ipsp
        psp = -1 * psp

        if _model_configurator_do_plot:
            plt.close("all")
            plt.figure()
            plt.title(
                f"g_ampa={g_ampa}\ng_gaba={g_gaba}\nv_rec_rest={v_rec_rest}\npsp={psp}"
            )
            plt.plot(v_rec[:end_timestep])
            plt.plot([0, end_timestep], [v_rec_rest, v_rec_rest], "k--")
            plt.xlim(0, end_timestep)
            plt.tight_layout()
            plt.savefig(
                f"{_model_configurator_figure_folder}/psp_{population.name}_{int(g_ampa*1000)}_{int(g_gaba*1000)}.png"
            )
            plt.close("all")

        return psp

    def get_firing_rate(
        self, pop_name: str, I_app: float = 0, g_ampa: float = 0, g_gaba: float = 0
    ):
        """
        Simulate the single neuron network of the given pop_name for 500 ms initial
        duration and 5000 ms. An input current I_app and the conductances g_ampa and
        g_gaba are applied. The firing rate is calculated from the spikes in the last
        5000 ms.

        Args:
            pop_name (str):
                Name of the population
            I_app (float, optional):
                Applied current
            g_ampa (float, optional):
                Conductance of the ampa synapse
            g_gaba (float, optional):
                Conductance of the gaba synapse

        Returns:
            rate (float):
                Firing rate in Hz
        """

        ### get the network, population, monitor, init_sampler from single nets
        net = self._single_nets.single_net(pop_name=pop_name, mode="normal").net
        population = self._single_nets.single_net(
            pop_name=pop_name, mode="normal"
        ).population
        monitor = self._single_nets.single_net(pop_name=pop_name, mode="normal").monitor
        init_sampler = self._single_nets.single_net(
            pop_name=pop_name, mode="normal"
        ).init_sampler
        ### reset network
        net.reset()
        net.set_seed(0)
        ### set the initial variables of the neuron model
        if init_sampler is not None:
            init_sampler.set_init_variables(population)
        ### slow down conductances (i.e. make them constant)
        population.tau_ampa = 1e20
        population.tau_gaba = 1e20
        ### apply given variables
        population.I_app = I_app
        population.g_ampa = g_ampa
        population.g_gaba = g_gaba
        ### simulate 500 ms initial duration + 5000 ms
        net.simulate(500 + 5000)
        ### get rate for the last 5000 ms
        spike_dict = monitor.get("spike")
        time_list = np.array(spike_dict[0])
        nbr_spks = np.sum((time_list > (500 / dt())).astype(int))
        rate = nbr_spks / (5000 / 1000)

        return rate


class PreparePSP:
    """
    Find v_rest, corresponding I_hold (in case of self-active neurons) and an
    init_sampler to initialize the neuron model for the PSP calculation for each
    population.
    """

    def __init__(
        self,
        model: CompNeuroModel,
        single_nets: CreateSingleNeuronNetworks,
        do_not_config_list: list[str],
        simulator: "Simulator",
    ):
        """
        Args:
            model (CompNeuroModel):
                Model to be prepared
            do_not_config_list (list[str]):
                List of populations which should not be configured
        """
        sf.Logger().log(
            "Prepare PSP calculation simulations for each population, i.e. find v_rest and I_app_hold.",
            verbose=_model_configurator_verbose,
        )
        self._single_nets = single_nets
        self._prepare_psp_dict = {}
        self._simulator = simulator
        ### loop over all populations
        for pop_name in model.populations:
            ### skip populations which should not be configured
            if pop_name in do_not_config_list:
                continue
            ### find initial v_rest using the voltage clamp network
            sf.Logger().log(
                f"[{pop_name}]: search v_rest with y(X) = delta_v_2000(v=X) using grid search",
                verbose=_model_configurator_verbose,
            )
            v_rest, delta_v_v_rest, variables_v_rest = self._find_v_rest_initial(
                pop_name=pop_name,
            )
            sf.Logger().log(
                f"[{pop_name}]: found v_rest={v_rest} with delta_v_2000(v=v_rest)={delta_v_v_rest}",
                verbose=_model_configurator_verbose,
            )
            ### check if v is constant after setting v to v_rest by simulating the normal
            ### single neuron network for 2000 ms
            sf.Logger().log(
                f"[{pop_name}]: check if v stays constant after setting v_rest as membrane potential",
                verbose=_model_configurator_verbose,
            )
            v_rest_is_constant, v_rest_arr = self._get_v_rest_is_const(
                pop_name=pop_name,
                variables_v_rest=variables_v_rest,
            )

            if v_rest_is_constant:
                ### v_rest found (last v value of the previous simulation), no
                ### I_app_hold needed
                sf.Logger().log(
                    f"[{pop_name}]: v_rest stays constant, no I_app_hold needed",
                    verbose=_model_configurator_verbose,
                )
                v_rest = v_rest_arr[-1]
                I_app_hold = 0
            else:
                ### there is no resting_state i.e. neuron is self-active --> find
                ### smallest negative I_app to silence neuron
                sf.Logger().log(
                    f"[{pop_name}]: neuron seems to be self-active --> find smallest I_app to silence the neuron",
                    verbose=_model_configurator_verbose,
                )
                v_rest, I_app_hold = self._find_I_app_hold(
                    pop_name=pop_name,
                    variables_v_rest=variables_v_rest,
                )
            sf.Logger().log(
                f"[{pop_name}]: final values: I_app_hold = {I_app_hold}, v_rest = {v_rest}"
            )

            ### get the sampler for the initial variables
            sf.Logger().log(
                f"[{pop_name}]: create init sampler for single neuron population to be used for the calculate PSP simulations",
                verbose=_model_configurator_verbose,
            )
            psp_init_sampler = self._get_init_neuron_variables_for_psp(
                pop_name=pop_name,
                v_rest=v_rest,
                I_app_hold=I_app_hold,
            )
            ### store the prepare PSP information
            self._prepare_psp_dict[pop_name] = {}
            self._prepare_psp_dict[pop_name]["v_rest"] = v_rest
            self._prepare_psp_dict[pop_name]["I_app_hold"] = I_app_hold
            self._prepare_psp_dict[pop_name]["psp_init_sampler"] = psp_init_sampler

    def get(self, pop_name: str):
        """
        Return the prepare PSP information for the given population.

        Args:
            pop_name (str):
                Name of the population

        Returns:
            ReturnPreparePSP:
                Prepare PSP information for the given population with Attributes: v_rest,
                I_app_hold, psp_init_sampler
        """
        return self.ReturnPreparePSP(
            v_rest=self._prepare_psp_dict[pop_name]["v_rest"],
            I_app_hold=self._prepare_psp_dict[pop_name]["I_app_hold"],
            psp_init_sampler=self._prepare_psp_dict[pop_name]["psp_init_sampler"],
        )

    class ReturnPreparePSP:
        def __init__(
            self, v_rest: float, I_app_hold: float, psp_init_sampler: ArrSampler
        ):
            self.v_rest = v_rest
            self.I_app_hold = I_app_hold
            self.psp_init_sampler = psp_init_sampler

    def _get_init_neuron_variables_for_psp(
        self, pop_name: str, v_rest: float, I_app_hold: float
    ):
        """
        Get the initial variables of the neuron model for the PSP calculation.

        Args:
            pop_name (str):
                Name of the population
            v_rest (float):
                Resting membrane potential
            I_app_hold (float):
                Current which silences the neuron

        Returns:
            sampler (ArrSampler):
                Sampler with the initial variables of the neuron model
        """
        ### get the names of the variables of the neuron model
        var_name_list = self._single_nets.single_net(
            pop_name=pop_name, mode="normal"
        ).population.variables
        ### get the variables of the neuron model after 5000 ms
        var_arr = self._simulator.get_v_psp(
            v_rest=v_rest, I_app_hold=I_app_hold, pop_name=pop_name
        )
        ### create a sampler with this single data sample
        sampler = ArrSampler(arr=var_arr, var_name_list=var_name_list)
        return sampler

    def _find_I_app_hold(
        self,
        pop_name: str,
        variables_v_rest: dict,
    ):
        """
        Find the current which silences the neuron.

        Args:
            pop_name (str):
                Name of the population
            variables_v_rest (dict):
                Stady state variables of the neuron during setting v_rest as membrane
                potential

        Returns:
            v_rest (float):
                Resting membrane potential
            I_app_hold (float):
                Current which silences the neuron
        """
        ### find I_app_hold with find_x_bound
        sf.Logger().log(
            f"[{pop_name}]: search I_app_hold with y(X) = CHANGE_OF_V(I_app=X), y(X_bound) should be 0",
            verbose=_model_configurator_verbose,
        )

        I_app_hold = -ef.find_x_bound(
            ### negative current initially reduces v then v climbs back up -->
            ### get_v_change_after_v_rest checks how much v changes during second half of
            ### 2000 ms simulation
            y=lambda X_val: -self._get_v_change_after_v_rest(
                pop_name=pop_name,
                variables_v_rest=variables_v_rest,
                ### find_x_bound only uses positive values for X and
                ### increases them, expecting to increase y, therefore use -X for I_app
                ### (increasing X will "increase" negative current) and negative sign for
                ### the returned value (for no current input the change is positive, this
                ### should decrease to zero, with negative sign: for no current input the
                ### change is negative, this should increase above zero)
                I_app=-X_val,
            ),
            ### y is initially negative and should increase above 0, therefore search for
            ### y_bound=0 with bound_type="greater"
            x0=0,
            y_bound=0,
            tolerance=0.01,
            bound_type="greater",
        )
        sf.Logger().log(
            f"[{pop_name}]: found I_app_hold={I_app_hold} with y(X_bound)={self._get_v_change_after_v_rest(pop_name=pop_name, variables_v_rest=variables_v_rest, I_app=I_app_hold)}",
            verbose=_model_configurator_verbose,
        )
        ### again simulate the neuron with the obtained I_app_hold to get the new v_rest
        v_rest_arr = self._simulator.get_v_2000(
            pop_name=pop_name,
            initial_variables=variables_v_rest,
            I_app=I_app_hold,
        )
        v_rest = v_rest_arr[-1]
        return v_rest, I_app_hold

    def _find_v_rest_initial(
        self,
        pop_name: str,
    ):
        """
        Find the initial v_rest with the voltage clamp single neuron network for the
        given population. Furthermore, get the change of v durign setting v_rest and the
        stady state variables of the neuron (at the end of the simulation).

        Args:
            pop_name (str):
                Name of the population

        Returns:
            v_rest (float):
                Resting membrane potential
            detla_v_v_rest (float):
                Change of the membrane potential during setting v_rest as membrane
                potential
            variables_v_rest (dict):
                Stady state variables of the neuron during setting v_rest as membrane
                potential
        """
        ### find v where dv/dt is minimal with voltage clamp network (best = 0, it can
        ### only be >= 0)
        v_arr = np.linspace(-90, -20, 200)
        v_clamp_arr = np.array(
            [
                self._simulator.get_v_clamp_2000(pop_name=pop_name, v=v_val)
                for v_val in v_arr
            ]
        )
        v_clamp_min_idx = argrelmin(v_clamp_arr)[0]
        v_rest = np.min(v_arr[v_clamp_min_idx])
        if _model_configurator_do_plot:
            plt.close("all")
            plt.figure()
            plt.plot(v_arr, v_clamp_arr)
            plt.xlabel("v in mV")
            plt.ylabel("v_clamp_rec")
            plt.axvline(v_rest, color="k")
            plt.axhline(0, color="k", ls="dashed")
            plt.savefig(f"{_model_configurator_figure_folder}/v_clamp_{pop_name}.png")
            plt.close("all")

        ### do again the simulation only with the obtained v_rest to get the detla_v for
        ### v_rest
        detla_v_v_rest = (
            self._simulator.get_v_clamp_2000(pop_name=pop_name, v=v_rest) * dt()
        )
        population = self._single_nets.single_net(
            pop_name=pop_name, mode="v_clamp"
        ).population
        ### and the stady state variables of the neuron
        variables_v_rest = {
            var_name: getattr(population, var_name) for var_name in population.variables
        }
        return v_rest, detla_v_v_rest, variables_v_rest

    def _get_v_rest_is_const(self, pop_name: str, variables_v_rest: dict):
        """
        Check if the membrane potential is constant after setting it to v_rest.

        Args:
            pop_name (str):
                Name of the population
            variables_v_rest (dict):
                Stady state variables of the neuron during setting v_rest as membrane
                potential, used as initial variables for the simulation

        Returns:
            v_rest_is_constant (bool):
                True if the membrane potential is constant, False otherwise
            v_rest_arr (np.array):
                Membrane potential for the 2000 ms simulation with shape: (time_steps,)
        """
        ### check if the neuron stays at v_rest with normal neuron
        v_rest_arr = self._simulator.get_v_2000(
            pop_name=pop_name,
            initial_variables=variables_v_rest,
            I_app=0,
        )
        v_rest_arr_is_const = (
            np.std(v_rest_arr) <= np.mean(np.absolute(v_rest_arr)) / 1000
        )
        return v_rest_arr_is_const, v_rest_arr

    def _get_v_change_after_v_rest(
        self, pop_name: str, variables_v_rest: dict, I_app: float
    ):
        """
        Check how much the membrane potential changes after setting it to v_rest.

        Args:
            pop_name (str):
                Name of the population
            variables_v_rest (dict):
                Stady state variables of the neuron during setting v_rest as membrane
                potential, used as initial variables for the simulation

        Returns:
            change_after_v_rest (np.array):
                Change of the membrane potential after setting it to v_rest
        """
        ### simulate 2000 ms after setting v_rest
        v_rest_arr = self._simulator.get_v_2000(
            pop_name=pop_name,
            initial_variables=variables_v_rest,
            I_app=I_app,
        )
        ### check how much v changes during the second half
        ### std(v) - mean(v)/1000 should be close to 0, the larger the value the more v
        ### changes
        change_after_v_rest = (
            np.std(v_rest_arr[len(v_rest_arr) // 2 :], axis=0)
            - np.mean(np.absolute(v_rest_arr[len(v_rest_arr) // 2 :]), axis=0) / 1000
        )
        return change_after_v_rest


class GetMaxSyn:
    """
    Find the maximal synaptic input for each population.
    """

    def __init__(
        self,
        model: CompNeuroModel,
        simulator: Simulator,
        do_not_config_list: list[str],
        max_psp: float,
        target_firing_rate_dict: dict[str, float],
    ):
        """
        Args:
            model (CompNeuroModel):
                Model to be analyzed
            simulator (Simulator):
                Simulator object for simulations with the single neuron networks
            do_not_config_list (list[str]):
                List of populations which should not be configured
            max_psp (float):
                Maximal postsynaptic potential in mV
            target_firing_rate_dict (dict[str, float]):
                Target firing rate for each population
        """
        sf.Logger().log(
            "Find maximal synaptic input for each population.",
            verbose=_model_configurator_verbose,
        )
        self._simulator = simulator
        self._max_syn_dict = {}
        ### loop over all populations
        for pop_name in model.populations:
            ### skip populations which should not be configured
            if pop_name in do_not_config_list:
                continue

            ### get max g_gabe
            g_gaba_max = self._get_max_g_gaba(pop_name=pop_name, max_psp=max_psp)

            ### get max g_ampa
            g_ampa_max = self._get_max_g_ampa(pop_name=pop_name, g_gaba_max=g_gaba_max)

            ### get max I_app
            I_app_max = self._get_max_I_app(
                pop_name=pop_name, target_firing_rate_dict=target_firing_rate_dict
            )

            ### store the maximal synaptic input in dict
            self._max_syn_dict[pop_name] = {}
            self._max_syn_dict[pop_name]["g_gaba"] = g_gaba_max
            self._max_syn_dict[pop_name]["g_ampa"] = g_ampa_max
            self._max_syn_dict[pop_name]["I_app"] = I_app_max

    @property
    def max_syn_getter(self):
        return self.MaxSynGetter(self._max_syn_dict)

    class MaxSynGetter:
        def __init__(self, max_syn_dict: dict) -> None:
            self._max_syn_dict = max_syn_dict

        def get(self, pop_name: str):
            """
            Return the maximal synaptic input for the given population.

            Args:
                pop_name (str):
                    Name of the population

            Returns:
                ReturnMaxSyn:
                    Maximal synaptic input for the given population with Attributes: g_gaba,
                    g_ampa, I_app
            """
            return self.ReturnMaxSyn(
                g_gaba=self._max_syn_dict[pop_name]["g_gaba"],
                g_ampa=self._max_syn_dict[pop_name]["g_ampa"],
                I_app=self._max_syn_dict[pop_name]["I_app"],
            )

        class ReturnMaxSyn:
            def __init__(self, g_gaba: float, g_ampa: float, I_app: float):
                self.g_gaba = g_gaba
                self.g_ampa = g_ampa
                self.I_app = I_app

    def _get_max_g_gaba(self, pop_name: str, max_psp: float):
        """
        Find the maximal g_gaba for the given population. A single spike with maximal
        g_gaba should result in a inhibitory postsynaptic potential of max_psp.

        Args:
            pop_name (str):
                Name of the population
            max_psp (float):
                Maximal postsynaptic potential in mV

        Returns:
            g_gaba_max (float):
                Maximal g_gaba
        """
        ### find g_gaba max using max IPSP
        sf.Logger().log(
            f"[{pop_name}]: search g_gaba_max with y(X) = PSP(g_ampa=0, g_gaba=X), y(X_bound) should be {max_psp}",
            verbose=_model_configurator_verbose,
        )
        x_bound = ef.find_x_bound(
            y=lambda X_val: self._simulator.get_ipsp(
                pop_name=pop_name,
                g_gaba=X_val,
            ),
            x0=0,
            y_bound=max_psp,
            tolerance=0.005,
        )
        sf.Logger().log(
            f"[{pop_name}]: found g_gaba_max={x_bound} with y(X_bound)={self._simulator.get_ipsp(pop_name=pop_name, g_gaba=x_bound)}",
            verbose=_model_configurator_verbose,
        )
        return x_bound

    def _get_max_g_ampa(self, pop_name: str, g_gaba_max: float):
        """
        Find the maximal g_ampa for the given population. The maximal g_ampa should
        override the maximal IPSP of g_gaba.

        Args:
            pop_name (str):
                Name of the population
            g_gaba_max (float):
                Maximal g_gaba

        Returns:
            g_ampa_max (float):
                Maximal g_ampa
        """
        ### find g_ampa max by "overriding" IPSP of g_gaba max
        sf.Logger().log(
            f"[{pop_name}]: search g_ampa_max with y(X) = PSP(g_ampa=X, g_gaba=g_gaba_max={g_gaba_max}), y(X_bound) should be {0}",
            verbose=_model_configurator_verbose,
        )

        def func(x):
            ipsp = self._simulator.get_ipsp(
                pop_name=pop_name,
                g_gaba=g_gaba_max,
                g_ampa=x,
            )
            ### find_x_bound tries to increase x to increase y, therefore, use
            ### -ipsp, since initially the ipsp is maximal, thus, y is negative and by
            ### increasing x it should increase to 0
            y = -ipsp
            ### next problem: find_x_bound expects a function which increases beyond the
            ### bound but -ipsp can maximum reach 0, thus, use -ipsp + x
            if y >= 0:
                return y + x
            else:
                return y

        x_bound = ef.find_x_bound(
            y=func,
            x0=0,
            y_bound=0,
            tolerance=0.005,
        )
        sf.Logger().log(
            f"[{pop_name}]: found g_ampa_max={x_bound} with y(X_bound)={self._simulator.get_ipsp(pop_name=pop_name, g_ampa=x_bound, g_gaba=g_gaba_max)}",
            verbose=_model_configurator_verbose,
        )
        return x_bound

    def _get_max_I_app(self, pop_name: str, target_firing_rate_dict: dict[str, float]):
        """
        Find the maximal current input for the given population. The maximal current
        input should result in "resting" firing rate + target firing rate + 100 Hz.

        Args:
            pop_name (str):
                Name of the population
            target_firing_rate_dict (dict[str, float]):
                Target firing rate for each population

        Returns:
            I_app_max (float):
                Maximal current input
        """
        ### get f_0 and f_max
        f_0 = self._simulator.get_firing_rate(pop_name=pop_name)
        f_max = f_0 + target_firing_rate_dict[pop_name] + 100

        ### find I_max with f_0, and f_max using find_x_bound
        sf.Logger().log(
            f"[{pop_name}]: search I_app_max with y(X) = f(I_app=X, g_ampa=0, g_gaba=0), y(X_bound) should be {f_max}",
            verbose=_model_configurator_verbose,
        )
        x_bound = ef.find_x_bound(
            y=lambda X_val: self._simulator.get_firing_rate(
                pop_name=pop_name,
                I_app=X_val,
            ),
            x0=0,
            y_bound=f_max,
            tolerance=1,
        )
        sf.Logger().log(
            f"[{pop_name}]: found I_app_max={x_bound} with y(X_bound)={self._simulator.get_firing_rate(pop_name=pop_name, I_app=x_bound)}",
            verbose=_model_configurator_verbose,
        )
        return x_bound


class GetWeights:
    def __init__(
        self,
        model: CompNeuroModel,
        do_not_config_list: list[str],
        analyze_model: AnalyzeModel,
        max_syn: GetMaxSyn.MaxSynGetter,
    ):
        sf.Logger().log(
            "Initialize the weight dictionary with the maximal weights.",
            verbose=_model_configurator_verbose,
        )
        self._model = model
        self._do_not_config_list = do_not_config_list
        self._analyze_model = analyze_model
        self._max_syn = max_syn
        ### initialize the weight_dict with the maximal weights
        weight_dict_init = {}
        for proj_name in model.projections:
            post_pop_name = analyze_model.pre_post_pop_name_dict[proj_name][1]
            target_type = analyze_model.proj_init_parameter_dict[proj_name]["target"]
            if target_type == "ampa":
                weight = self._max_syn.get(pop_name=post_pop_name).g_ampa
            elif target_type == "gaba":
                weight = self._max_syn.get(pop_name=post_pop_name).g_gaba
            weight_dict_init[proj_name] = weight
        ### first set the interal weight dict variable, then the property to calculate
        ### syn_load_dict and syn_contribution_dict
        self._weight_dict = weight_dict_init
        self.weight_dict = weight_dict_init

    @property
    def weight_dict(self):
        return self._weight_dict

    @weight_dict.setter
    def weight_dict(self, value: dict[str, float]):
        ### check if the dictionary "value" has the same keys as the internal weight_dict
        if set(value.keys()) != set(self._weight_dict.keys()):
            raise ValueError(
                f"The keys of the weight_dict must be: {set(self._weight_dict.keys())}"
            )
        ### if weight_dict is set, recalculate syn_load_dict and syn_contribution_dict
        self._weight_dict = value
        self._syn_load_dict = self._get_syn_load_dict()
        self._syn_contribution_dict = self._get_syn_contribution_dict()

    @property
    def syn_load_dict(self):
        return self._syn_load_dict

    @syn_load_dict.setter
    def syn_load_dict(self, value: dict[str, dict[str, float]]):
        ### check if the dictionary "value" has the same structure as the internal
        ### nested dict syn_load_dict
        if set(value.keys()) != set(self._syn_load_dict.keys()):
            raise ValueError(
                f"The syn_load_dict must have this structure: {self._syn_load_dict}"
            )
        for pop_name in value.keys():
            if set(value[pop_name].keys()) != set(self._syn_load_dict[pop_name].keys()):
                raise ValueError(
                    f"The syn_load_dict must have this structure: {self._syn_load_dict}"
                )
        ### check if values are between 0 and 1
        for pop_name in value.keys():
            for target in value[pop_name].keys():
                if not 0 <= value[pop_name][target] <= 1:
                    raise ValueError(
                        "The values of the syn_load_dict must be between 0 and 1"
                    )

        ### if syn_load_dict is set, recalculate weight_dict
        self._syn_load_dict = value
        self._weight_dict = self._get_weight_dict()

    @property
    def syn_contribution_dict(self):
        return self._syn_contribution_dict

    @syn_contribution_dict.setter
    def syn_contribution_dict(self, value: dict[str, dict[str, dict[str, float]]]):
        ### check if the dictionary "value" has the same structure as the internal
        ### nested dict syn_contribution_dict
        if set(value.keys()) != set(self._syn_contribution_dict.keys()):
            raise ValueError(
                f"The syn_contribution_dict must have this structure: {self._syn_contribution_dict}"
            )
        for pop_name in value.keys():
            if set(value[pop_name].keys()) != set(
                self._syn_contribution_dict[pop_name].keys()
            ):
                raise ValueError(
                    f"The syn_contribution_dict must have this structure: {self._syn_contribution_dict}"
                )
            for target in value[pop_name].keys():
                if set(value[pop_name][target].keys()) != set(
                    self._syn_contribution_dict[pop_name][target].keys()
                ):
                    raise ValueError(
                        f"The syn_contribution_dict must have this structure: {self._syn_contribution_dict}"
                    )
        ### check if values are between 0 and 1
        for pop_name in value.keys():
            for target in value[pop_name].keys():
                for proj_name in value[pop_name][target].keys():
                    if not 0 <= value[pop_name][target][proj_name] <= 1:
                        raise ValueError(
                            "The values of the syn_contribution_dict must be between 0 and 1"
                        )

        ### if syn_contribution_dict is set, recalculate weight_dict
        self._syn_contribution_dict = value
        self._weight_dict = self._get_weight_dict()

    def _get_weight_dict(self):
        ### set the weights population wise for the afferent projections
        weight_dict = {}
        ### loop over all populations
        for pop_name in self._model.populations:
            ### skip populations which should not be configured
            if pop_name in self._do_not_config_list:
                continue
            synaptic_load = self._syn_load_dict[pop_name]
            ### loop over target types
            for target, load in synaptic_load.items():
                synaptic_contribution = self._syn_contribution_dict[pop_name][target]
                ### loop over afferebt projections with target type
                for proj_name in synaptic_contribution.keys():
                    max_conductance = (
                        self._max_syn.get(pop_name=pop_name).g_ampa
                        if target == "ampa"
                        else self._max_syn.get(pop_name=pop_name).g_gaba
                    )
                    weight_dict[proj_name] = (
                        load * synaptic_contribution[proj_name] * max_conductance
                    )

        return weight_dict

    def _get_syn_load_dict(self):
        syn_load_dict = {}
        ### loop over populations
        for pop_name in self._model.populations:
            ### skip populations which should not be configured
            if pop_name in self._do_not_config_list:
                continue
            syn_load_dict[pop_name] = {}
            ### loop over target types
            for target in ["ampa", "gaba"]:
                ### get all afferent projections with target type
                proj_name_list = self._get_afferent_proj_names(
                    pop_name=pop_name, target=target
                )
                if len(proj_name_list) == 0:
                    continue
                ### get the maximal weight of the afferent projections
                max_weight = max(
                    [self._weight_dict[proj_name] for proj_name in proj_name_list]
                )
                ### get the synaptic load
                if target == "ampa":
                    syn_load_dict[pop_name][target] = (
                        max_weight / self._max_syn.get(pop_name=pop_name).g_ampa
                    )
                elif target == "gaba":
                    syn_load_dict[pop_name][target] = (
                        max_weight / self._max_syn.get(pop_name=pop_name).g_gaba
                    )

        return syn_load_dict

    def _get_syn_contribution_dict(self):
        syn_contribution_dict = {}
        ### loop over populations
        for pop_name in self._model.populations:
            ### skip populations which should not be configured
            if pop_name in self._do_not_config_list:
                continue
            syn_contribution_dict[pop_name] = {}
            ### loop over target types
            for target in ["ampa", "gaba"]:
                ### get all afferent projections with target type
                proj_name_list = self._get_afferent_proj_names(
                    pop_name=pop_name, target=target
                )
                if len(proj_name_list) == 0:
                    continue
                ### get the synaptic contribution
                syn_contribution_dict[pop_name][target] = {}
                for proj_name in proj_name_list:
                    syn_contribution_dict[pop_name][target][
                        proj_name
                    ] = self._weight_dict[proj_name] / max(
                        [self._weight_dict[proj_name] for proj_name in proj_name_list]
                    )

        return syn_contribution_dict

    ### synaptic load for each population between 0 and 1, determines the largest weight of incoming synapses, 1 means maximal conductance
    ### to get synaptic load of a population/target, get all afferent projections of the population/target and take the maximal weight divided by the (global) maximal weight
    ### synaptic contribution for each if a population has for a target type multiple afferent projections --> array with numbers for these projections
    ### divide the array by tjhe max value --> e.g. result is [0.6,1.0] weights of the projections then are 0.6*max_weight and 1.0*max_weight, where max weight is determined by the synaptic load

    def _get_afferent_proj_names(self, pop_name: str, target: str):
        proj_name_list = []
        for proj_name in self._model.projections:
            if (
                self._analyze_model.pre_post_pop_name_dict[proj_name][1] == pop_name
                and self._analyze_model.proj_init_parameter_dict[proj_name]["target"]
                == target
            ):
                proj_name_list.append(proj_name)

        return proj_name_list


class GetBase:
    def __init__(
        self,
        model_normal: CompNeuroModel,
        model_reduced: CreateReducedModel,
        target_firing_rate_dict: dict,
        weight_dicts: "GetWeights",
        do_not_config_list: list[str],
        init_sampler: CreateSingleNeuronNetworks.AllSampler,
        max_syn: "GetMaxSyn.MaxSynGetter",
        v_tol: float,
    ):
        ### TODO if the weights are "too strong" crazy I_app values are needed --> I
        ### have to use lower and upper bounds!
        ### but then TODO: find a better way to get I_app_max ... this f_0+f_target+100 is not so good
        ### maybe something with to voltage clamp
        ### an input current which causes this change of v, maybe I find a good heuristic a la max change of v caused by input current within 1 ms should be 10 mV
        ### TODO another idea: for the optimization not only consider the firing rate but also the membrane potential if the firing rate is zero
        ### the problem with "only firng rate" is that it has no gradient information for below zero firing rates
        sf.Logger().log(
            "Find the base currents for the target firing rates.",
            verbose=_model_configurator_verbose,
        )
        self._model_normal = model_normal
        self._create_reduce_model = model_reduced
        self._weight_dicts = weight_dicts
        self._do_not_config_list = do_not_config_list
        self._init_sampler = init_sampler
        self._max_syn = max_syn
        self._v_tol = v_tol
        ### get the populations names of the configured populations
        self._pop_names_config = [
            pop_name
            for pop_name in model_normal.populations
            if pop_name not in do_not_config_list
        ]
        ### convert the target firing rate dict to an array
        self._target_firing_rate_arr = []
        print(self._pop_names_config)
        for pop_name in self._pop_names_config:
            self._target_firing_rate_arr.append(target_firing_rate_dict[pop_name])
        self._target_firing_rate_arr = np.array(self._target_firing_rate_arr)
        ### get the base currents
        self._prepare_get_base()
        self._base_dict = self._get_base()

    @property
    def base_dict(self):
        return self._base_dict

    def _prepare_get_base(self):
        sf.Logger().log(
            "Prepare the model and experiment for the optimization.",
            verbose=_model_configurator_verbose,
        )
        ### clear ANNarchy
        mf.cnp_clear(functions=False, constants=False)
        ### create and compile the model
        self._create_reduce_model.model_reduced.create()
        ### create monitors for recording the spikes of all populations
        ### for CompNeuroMonitors we need the "_reduced" suffix
        mon = CompNeuroMonitors(
            mon_dict={
                f"{pop_name}_reduced": ["spike"]
                for pop_name in self._model_normal.populations
            }
        )
        ### initialize all populations with the init sampler
        for pop_name in self._pop_names_config:
            ### for get_population we need the "_reduced" suffix
            self._init_sampler.get(pop_name=pop_name).set_init_variables(
                get_population(f"{pop_name}_reduced")
            )
        ### set the model weights
        self._create_reduce_model.set_weights(
            weight_dict=self._weight_dicts.weight_dict
        )
        ### create the experiment
        self._exp = self.MyExperiment(monitors=mon)
        ### store the model state for all populations
        self._exp.store_model_state(
            compartment_list=self._create_reduce_model.model_reduced.populations
        )
        ### set lower and upper bounds and initial guess
        self._lb = []
        self._ub = []
        self._x0 = []
        for pop_name in self._pop_names_config:
            self._lb.append(-self._max_syn.get(pop_name=pop_name).I_app)
            self._ub.append(self._max_syn.get(pop_name=pop_name).I_app)
            self._x0.append(0.0)

    def _get_base(self):
        """
        Perform the optimization to find the base currents for the target firing rates.

        Returns:
            base_dict (dict):
                Dict with the base currents for each population
        """

        ### Perform the optimization using Minimize class
        result = self.Minimize(
            func=self._get_firing_rate,
            yt=self._target_firing_rate_arr,
            x0=np.array(self._x0),
            lb=np.array(self._lb),
            ub=np.array(self._ub),
            tol_error=self._v_tol,
            tol_convergence=0.01,
            max_it=20,
        )

        optimized_inputs = result.x
        if not result.success:
            sf.Logger().log("Optimization failed, target firing rates not reached!")
            print("Optimization failed, target firing rates not reached!")
        base_dict = {
            pop_name: optimized_inputs[idx]
            for idx, pop_name in enumerate(self._pop_names_config)
        }
        return base_dict

    def _get_firing_rate(self, I_app_list: list[float]):
        ### convert the I_app_list to a dict
        I_app_dict = {}
        counter = 0
        for pop_name in self._pop_names_config:
            ### for the I_app_dict we need the "_reduced" suffix
            I_app_dict[f"{pop_name}_reduced"] = I_app_list[counter]
            counter += 1
        ### run the experiment
        results = self._exp.run(I_app_dict)
        ### get the firing rates from the recorded spikes
        rate_list = []
        rate_dict = {}
        for pop_name in self._pop_names_config:
            ### for the spike dict we need the "_reduced" suffix
            spike_dict = results.recordings[0][f"{pop_name}_reduced;spike"]
            t, _ = raster_plot(spike_dict)
            ### only take spikes after the first 500 ms, because the neurons are
            ### initialized in resting-state and with an input current there can be
            ### drastic activity changes at the beginning
            t = t[t > 500]
            nbr_spikes = len(t)
            ### divide number of spikes by the number of neurons and the duration in s
            if nbr_spikes > 0:
                rate = nbr_spikes / (4.5 * get_population(f"{pop_name}_reduced").size)
            else:
                ### for rate 0 return the membrane potential instead, so that <0 still contains gradient information
                rate = np.mean(get_population(f"{pop_name}_reduced").v)
            rate_list.append(rate)
            rate_dict[pop_name] = rate
        # sf.Logger().log(f"I_app_dict: {I_app_dict}")
        # sf.Logger().log(f"Firing rates: {rate_dict}")

        # af.PlotRecordings(
        #     figname="firing_rates.png",
        #     recordings=results.recordings,
        #     recording_times=results.recording_times,
        #     shape=(len(self._model_normal.populations), 1),
        #     plan={
        #         "position": list(range(1, len(self._model_normal.populations) + 1)),
        #         "compartment": [
        #             f"{pop_name}_reduced" for pop_name in self._model_normal.populations
        #         ],
        #         "variable": ["spike"] * len(self._model_normal.populations),
        #         "format": ["hybrid"] * len(self._model_normal.populations),
        #     },
        # )
        return np.array(rate_list)

    class MyExperiment(CompNeuroExp):
        def run(self, I_app_dict: dict[str, float]):
            """
            Simulate the model for 5000 ms with the given input currents.

            Args:
                I_app_dict (dict[str, float]):
                    Dict with the input currents for each population

            Returns:
                results (CompNeuroResults):
                    Results of the simulation
            """
            ### reset to initial state
            self.reset()
            set_seed(0)
            ### activate monitor
            self.monitors.start()
            ### set the input currents
            for pop_name, I_app in I_app_dict.items():
                get_population(pop_name).I_app = I_app
            ### simulate 5000 ms
            simulate(5000)
            ### return results
            return self.results()

    class Minimize:
        def __init__(
            self, func, yt, x0, lb, ub, tol_error, tol_convergence, max_it
        ) -> None:
            """
            Args:
                func (Callable):
                    Function which takes a vector as input and returns a vector as output
                target_values (np.array):
                    Target output vector of the function
                x0 (np.array):
                    Initial input vector
                lb (np.array):
                    Lower bounds of the input vector
                ub (np.array):
                    Upper bounds of the input vector
                tol_error (float):
                    If the error is below this value the optimization stops
                tol_convergence (float):
                    If the change of the error stays below this value the optimization stops
                max_it (int):
                    Maximum number of iterations
            """
            x = x0
            x_old = x0
            y = yt
            error = np.ones(x0.shape) * 20
            error_old = np.ones(x0.shape) * 20
            it = 0
            search_gradient_diff = np.ones(x0.shape)
            alpha = np.ones(x0.shape)
            error_list = []
            dx_list = []
            dy_list = []
            x_list = []
            y_list = []
            it_list = []

            def error_changed(error_list, tol, n=3):
                if len(error_list) < 2:
                    return True
                return (np.max(error_list[-n:]) - np.min(error_list[-n:])) > tol

            ### run until the error does not change anymore or the maximum number of
            ### iterations is reached, also break if the error is small enough
            while it < max_it and error_changed(error_list, tol_convergence):
                sf.Logger().log(f"iteration: {it}", verbose=_model_configurator_verbose)
                y_old = y
                y = func(x)
                dx_list.append(x - x_old)
                dy_list.append(y - y_old)
                ### TODO if x did not change much, use the previous gradient again, but maybe not a good idea, or at least not easy to implement, sicne gradient depends on all inputs
                sf.Logger().log(f"x: {x}", verbose=_model_configurator_verbose)
                sf.Logger().log(f"y: {y}", verbose=_model_configurator_verbose)
                x_list.append(x)
                y_list.append(y)
                it_list.append(it)
                ### here we know the new y(x)
                ### check if the error sign changed
                error_old = error
                error = yt - y
                ### if error is small enough stop the optimization
                if np.all(np.abs(error) < tol_error):
                    break
                error_sign_changed = np.sign(error) != np.sign(error_old)
                sf.Logger().log(
                    f"error_sign_changed: {error_sign_changed}",
                    verbose=_model_configurator_verbose,
                )
                ### get how much the error (in total, not for individual inputs) changed
                error_list.append(np.mean(np.abs(error)))
                sf.Logger().log(f"error_list: {error_list}")
                ### if the error sign changed:
                ### - check if error is larger as before
                ###   - if yes -> check if error is also larger than tolerance
                ###     - if yes -> use again the previous x and compute current y again
                ### - we calculate (as usual) a new gradient
                ### - we reduce alpha, so this time the step is smaller
                error_increased = np.abs(error) > np.abs(error_old)
                error_is_large = np.abs(error) > tol_error
                change_x = error_sign_changed & error_increased & error_is_large
                x[change_x] = x_old[change_x]
                if np.any(change_x):
                    y = func(x)
                    sf.Logger().log(
                        "some errors changed sign, increased, and are larger than tolerance",
                        verbose=_model_configurator_verbose,
                    )
                    sf.Logger().log(f"x: {x}", verbose=_model_configurator_verbose)
                    sf.Logger().log(f"y: {y}", verbose=_model_configurator_verbose)
                    x_list.append(x)
                    y_list.append(y)
                    it_list.append(it)
                ### reduce alpha for the inputs where the error sign changed
                ### for the others alpha reaches 1
                alpha[error_sign_changed] /= 2
                alpha[~error_sign_changed] += (1 - alpha[~error_sign_changed]) / 5
                ### calculate the gradient i.e. change of the output values for each input
                grad = np.zeros((yt.shape[0], x0.shape[0]))
                for i in range(len(x0)):
                    ### search for the gradient of the i-th input, increase the stepwidth
                    ### which is used to calculate the gradient if the gradient for the
                    ### associated output value is below 1
                    while grad[i, i] < 1:
                        x_plus = x.copy()
                        ### change only the currently selected input whose gradient should be
                        ### calculated
                        x_plus[i] += search_gradient_diff[i]
                        y_plus = func(x_plus)
                        sf.Logger().log(f"x_plus: {x_plus}", verbose=False)
                        sf.Logger().log(f"y_plus: {y_plus}", verbose=False)
                        grad[:, i] = y_plus - y
                        ### if gradient is above 10 reduce the search gradient diff
                        if grad[i, i] >= 10:
                            search_gradient_diff[i] /= 1.5
                        ### if gradient is below 1 increase the search gradient diff
                        elif grad[i, i] < 1:
                            search_gradient_diff[i] *= 2
                ### calculate the wanted change of the output values
                delta_y = yt - y
                sf.Logger().log(f"delta_y: {delta_y}", verbose=False)
                sf.Logger().log(f"grad:\n{grad}", verbose=False)

                # Example coefficient matrix A (m x n matrix)
                A = grad

                # Right-hand side vector b (m-dimensional vector)
                b = delta_y

                # Solve the system using least squares method
                solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

                # Output the results
                sf.Logger().log(f"Solution vector x: {solution}", verbose=False)

                # Calculate the matrix-vector product Ax
                Ax = np.dot(A, solution)

                # Output the matrix-vector product and compare with b
                sf.Logger().log(f"delta_y from solution: {Ax}", verbose=False)

                ### solution contains the info how much each input should change (how many
                ### times the change of gradient is needed to reach the target values)
                x_old = x
                x = x + solution * search_gradient_diff * alpha
                it += 1

            self.x = x
            self.success = np.all(np.abs(error) < tol_error)

            x_arr = np.array(x_list)
            y_arr = np.array(y_list)
            it_arr = np.array(it_list)

            ### TODO remove or make this optimal (for debugging), also the prints above
            if _model_configurator_do_plot:
                plt.close("all")
                plt.figure()
                for idx in range(4):
                    ax = plt.subplot(4, 1, idx + 1)
                    ### plot the x values
                    plt.plot(it_arr, x_arr[:, idx])
                    plt.ylabel(f"x{idx}")
                    ### second y axis on the right for the y values
                    ax2 = ax.twinx()
                    ax2.plot(it_arr, y_arr[:, idx], color="red")
                    ax2.set_ylabel(f"y{idx}", color="red")
                plt.xlabel("iteration")
                plt.tight_layout()
                plt.savefig(f"{_model_configurator_figure_folder}/optimization.png")
                plt.close("all")
