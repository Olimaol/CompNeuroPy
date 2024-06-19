from ANNarchy import (
    Neuron,
    Population,
    dt,
    add_function,
    Projection,
    get_population,
    Constant,
    Synapse,
    projections,
    populations,
    get_projection,
    Binomial,
    CurrentInjection,
)
from ANNarchy.core import ConnectorMethods
import numpy as np
from CompNeuroPy import model_functions as mf
from typingchecker import check_types
import inspect
from CompNeuroPy import CompNeuroModel
import sympy as sp

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


class _CreateReducedModel:
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
        self.reduced_size = reduced_size
        self.verbose = verbose
        ### check if model is already created but not compiled, if not clear annarchy
        ### and create it
        if not model.created or model.compiled:
            mf.cnp_clear(functions=False, neurons=True, synapses=True, constants=False)
            model.create(do_compile=False)

        ### analyze model to be able to recreate it
        self.analyze_model()

        ### clear model
        mf.cnp_clear(functions=False, neurons=True, synapses=True, constants=False)

        ### recreate model with reduced populations and projections
        self.model_reduced = CompNeuroModel(
            model_creation_function=self.recreate_model,
            name=f"{model.name}_reduced",
            description=f"{model.description}\nWith reduced populations and projections.",
            do_create=do_create,
            do_compile=do_compile,
            compile_folder_name=f"{model.compile_folder_name}_reduced",
        )

    def analyze_model(
        self,
    ):
        """
        Analyze the model to be able to recreate it.
        """
        ### get all population and projection names
        (
            self.population_name_list,
            self.projection_name_list,
        ) = self.get_all_population_and_projection_names()

        ### get population info (eq, params etc.)
        (
            self.neuron_model_attr_dict,
            self.neuron_model_init_parameter_dict,
            self.pop_init_parameter_dict,
        ) = self.analyze_populations()

        ### get projection info
        (
            self.proj_init_parameter_dict,
            self.synapse_init_parameter_dict,
            self.synapse_model_attr_dict,
            self.connector_function_dict,
            self.connector_function_parameter_dict,
            self.pre_post_pop_name_dict,
        ) = self.analyze_projections()

    def get_all_population_and_projection_names(self):
        """
        Get all population and projection names.

        Returns:
            population_name_list (list):
                List of all population names
            projection_name_list (list):
                List of all projection names
        """
        population_name_list: list[str] = [pop.name for pop in populations()]
        projection_name_list: list[str] = [proj.name for proj in projections()]

        return population_name_list, projection_name_list

    def analyze_populations(self):
        """
        Get info of each population
        """
        ### values of the paramters and variables of the population's neurons, keys are the names of paramters and variables
        neuron_model_attr_dict: dict[str, dict] = {}
        ### parameters of the __init__ function of the Neuron class
        neuron_model_init_parameter_dict: dict[str, dict] = {}
        ### parameters of the __init__ function of the Population class
        pop_init_parameter_dict: dict[str, dict] = {}

        ### for loop over all populations
        for pop_name in self.population_name_list:
            pop: Population = get_population(pop_name)
            ### get the neuron model attributes (parameters/variables)
            neuron_model_attr_dict[pop.name] = pop.init
            ### get a dict of all paramters of the __init__ function of the Neuron
            init_params = inspect.signature(Neuron.__init__).parameters
            neuron_model_init_parameter_dict[pop.name] = {
                param: getattr(pop.neuron_type, param)
                for param in init_params
                if param != "self"
            }
            ### get a dict of all paramters of the __init__ function of the Population
            init_params = inspect.signature(Population.__init__).parameters
            pop_init_parameter_dict[pop.name] = {
                param: getattr(pop, param)
                for param in init_params
                if param != "self" and param != "storage_order" and param != "copied"
            }

        return (
            neuron_model_attr_dict,
            neuron_model_init_parameter_dict,
            pop_init_parameter_dict,
        )

    def analyze_projections(self):
        """
        Get info of each projection
        """
        ### parameters of the __init__ function of the Projection class
        proj_init_parameter_dict: dict[str, dict] = {}
        ### parameters of the __init__ function of the Synapse class
        synapse_init_parameter_dict: dict[str, dict] = {}
        ### values of the paramters and variables of the synapse, keys are the names of paramters and variables
        synapse_model_attr_dict: dict[str, dict] = {}
        ### connector functions of the projections
        connector_function_dict: dict = {}
        ### parameters of the connector functions of the projections
        connector_function_parameter_dict: dict = {}
        ### names of pre- and post-synaptic populations of the projections
        pre_post_pop_name_dict: dict[str, tuple] = {}

        ### loop over all projections
        for proj_name in self.projection_name_list:
            proj: Projection = get_projection(proj_name)
            ### get the synapse model attributes (parameters/variables)
            synapse_model_attr_dict[proj.name] = proj.init
            ### get a dict of all paramters of the __init__ function of the Synapse
            init_params = inspect.signature(Synapse.__init__).parameters
            synapse_init_parameter_dict[proj.name] = {
                param: getattr(proj.synapse_type, param)
                for param in init_params
                if param != "self"
            }
            ### get a dict of all paramters of the __init__ function of the Projection
            init_params = inspect.signature(Projection.__init__).parameters
            proj_init_parameter_dict[proj_name] = {
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
                    f"Connector function '{_connector_methods_dict[proj.connector_name].__name__}' not supported yet"
                )

            ### get the connector function
            connector_function_dict[proj.name] = _connector_methods_dict[
                proj.connector_name
            ]

            ### get the parameters of the connector function
            connector_function_parameter_dict[proj.name] = (
                self.get_connector_parameters(proj)
            )

            ### get the names of the pre- and post-synaptic populations
            pre_post_pop_name_dict[proj.name] = (proj.pre.name, proj.post.name)

        return (
            proj_init_parameter_dict,
            synapse_init_parameter_dict,
            synapse_model_attr_dict,
            connector_function_dict,
            connector_function_parameter_dict,
            pre_post_pop_name_dict,
        )

    def get_connector_parameters(self, proj: Projection):
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

    def recreate_model(self):
        """
        Recreates the model with reduced populations and projections.
        """
        ### 1st for each population create a reduced population
        for pop_name in self.population_name_list:
            self.create_reduced_pop(pop_name)
        ### 2nd for each population which is a presynaptic population, create a spikes collecting aux population
        for pop_name in self.population_name_list:
            self.create_spike_collecting_aux_pop(pop_name)
        ## 3rd for each population which has afferents create a population for incoming spikes for each target type
        for pop_name in self.population_name_list:
            self.create_conductance_aux_pop(pop_name, target="ampa")
            self.create_conductance_aux_pop(pop_name, target="gaba")

    def create_reduced_pop(self, pop_name: str):
        """ """
        if self.verbose:
            print(f"create_reduced_pop for {pop_name}")
        ### 1st check how the population is connected
        is_presynaptic, is_postsynaptic, ampa, gaba = self.how_pop_is_connected(
            pop_name
        )

        ### 2nd recreate neuron model
        ### get the stored parameters of the __init__ function of the Neuron
        neuron_model_init_parameter_dict = self.neuron_model_init_parameter_dict[
            pop_name
        ].copy()
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
        pop_init_parameter_dict = self.pop_init_parameter_dict[pop_name].copy()
        ### replace the neuron model with the new neuron model
        pop_init_parameter_dict["neuron"] = neuron_model_new
        ### replace the size with the reduced size (if reduced size is smaller than the
        ### original size)
        ### TODO add model requirements somewhere, here requirements = geometry = int
        pop_init_parameter_dict["geometry"] = min(
            [pop_init_parameter_dict["geometry"][0], self.reduced_size]
        )
        ### append _reduce to the name
        pop_init_parameter_dict["name"] = f"{pop_name}_reduced"
        ### create the new population
        pop_new = Population(**pop_init_parameter_dict)

        ### 4th set the parameters and variables of the population's neurons
        ### get the stored parameters and variables
        neuron_model_attr_dict = self.neuron_model_attr_dict[pop_name]
        ### set the parameters and variables
        for attr_name, attr_val in neuron_model_attr_dict.items():
            setattr(pop_new, attr_name, attr_val)

    def create_spike_collecting_aux_pop(self, pop_name: str):
        """ """
        ### get all efferent projections
        efferent_projection_list = [
            proj_name
            for proj_name, pre_post_pop_name_dict in self.pre_post_pop_name_dict.items()
            if pre_post_pop_name_dict[0] == pop_name
        ]
        ### check if pop has efferent projections
        if len(efferent_projection_list) == 0:
            return
        if self.verbose:
            print(f"create_spike_collecting_aux_pop for {pop_name}")
        ### create the spike collecting population
        pop_aux = Population(
            1,
            neuron=self.SpikeProbCalcNeuron(
                reduced_size=min(
                    [
                        self.pop_init_parameter_dict[pop_name]["geometry"][0],
                        self.reduced_size,
                    ]
                )
            ),
            name=f"{pop_name}_spike_collecting_aux",
        )
        ### create the projection from reduced pop to spike collecting aux pop
        proj = Projection(
            pre=get_population(pop_name + "_reduced"),
            post=pop_aux,
            target="ampa",
            name=f"proj_{pop_name}_spike_collecting_aux",
        )
        proj.connect_all_to_all(weights=1)

    def create_conductance_aux_pop(self, pop_name: str, target: str):
        """ """
        ### get all afferent projections
        afferent_projection_list = [
            proj_name
            for proj_name, pre_post_pop_name_dict in self.pre_post_pop_name_dict.items()
            if pre_post_pop_name_dict[1] == pop_name
        ]
        ### check if pop has afferent projections
        if len(afferent_projection_list) == 0:
            return
        ### get all afferent projections with target type
        afferent_target_projection_list = [
            proj_name
            for proj_name in afferent_projection_list
            if self.proj_init_parameter_dict[proj_name]["target"] == target
        ]
        ### check if there are afferent projections with target type
        if len(afferent_target_projection_list) == 0:
            return
        if self.verbose:
            print(f"create_conductance_aux_pop for {pop_name} target {target}")
        ### get projection informations TODO in ReduceModel class weights and probs not global constants
        ### TODO somewhere add model requirements, here requirements = geometry = int and connection = fixed_probability i.e. random (with weights and probability)
        projection_dict = {
            proj_name: {
                "pre_size": self.pop_init_parameter_dict[
                    self.pre_post_pop_name_dict[proj_name][0]
                ]["geometry"][0],
                "connection_prob": self.connector_function_parameter_dict[proj_name][
                    "probability"
                ],
                "weights": self.connector_function_parameter_dict[proj_name]["weights"],
                "pre_name": self.pre_post_pop_name_dict[proj_name][0],
            }
            for proj_name in afferent_target_projection_list
        }
        ### create the conductance calculating population
        pop_aux = Population(
            self.pop_init_parameter_dict[pop_name]["geometry"][0],
            neuron=self.InputCalcNeuron(projection_dict=projection_dict),
            name=f"{pop_name}_{target}_aux",
        )
        ### set number of synapses parameter for each projection
        for proj_name, vals in projection_dict.items():
            number_synapses = Binomial(
                n=vals["pre_size"], p=vals["connection_prob"]
            ).get_values(self.pop_init_parameter_dict[pop_name]["geometry"][0])
            setattr(pop_aux, f"number_synapses_{proj_name}", number_synapses)
        ### create the "current injection" projection from conductance calculating
        ### population to the reduced post population
        proj = CurrentInjection(
            pre=pop_aux,
            post=get_population(f"{pop_name}_reduced"),
            target=f"incomingaux{target}",
            name=f"proj_{pop_name}_{target}_aux",
        )
        proj.connect_current()
        ### create projection from spike_prob calculating aux neurons of presynaptic
        ### populations to conductance calculating aux population
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
        for proj_name in self.projection_name_list:
            ### check if the population is a presynaptic population in any projection
            if self.pre_post_pop_name_dict[proj_name][0] == pop_name:
                is_presynaptic = True
            ### check if the population is a postsynaptic population in any projection
            if self.pre_post_pop_name_dict[proj_name][1] == pop_name:
                is_postsynaptic = True
                ### check if the projection target is ampa or gaba
                if self.proj_init_parameter_dict[proj_name]["target"] == "ampa":
                    ampa = True
                elif self.proj_init_parameter_dict[proj_name]["target"] == "gaba":
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
        def __init__(self, reduced_size=1):
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
        def __init__(self, projection_dict):
            """
            This neurons get the spike probabilities of the pre neurons and calculates the
            incoming spikes for each projection. It accumulates the incoming spikes of all
            projections (of the same target type) and calculates the conductance increase
            for the post neuron.

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
