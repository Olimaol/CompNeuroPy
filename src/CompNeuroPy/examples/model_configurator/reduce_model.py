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
)
from ANNarchy.core import ConnectorMethods
import numpy as np
from CompNeuroPy import model_functions as mf
from CompNeuroPy.generate_model import generate_model
from typingchecker import check_types
import inspect
from CompNeuroPy import CompNeuroModel

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
    Class to create a reduced model from the original model.
    """

    def __init__(self, model: CompNeuroModel, reduced_size: int) -> None:
        """
        Prepare model for DBS stimulation

        Args:
            model (CompNeuroModel):
                Model to be reduced
            reduced_size (int):
                Size of the reduced populations
        """
        self.reduced_size = reduced_size
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
        self.recreate_model()

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
        ### recreate populations
        for pop_name in self.population_name_list:
            self.recreate_population(pop_name)
        ### recreate projections
        for proj_name in self.projection_name_list:
            self.recreate_projection(proj_name)

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

    def recreate_population(self, pop_name):
        """
        Recreate a population with the same neuron model and parameters.

        Args:
            pop_name (str):
                Name of the population to recreate
        """
        ### 1st check how the population is connected
        is_presynaptic, is_postsynaptic, ampa, gaba = self.how_pop_is_connected(
            pop_name
        )

        ### 2nd recreate neuron model
        ### get the stored parameters of the __init__ function of the Neuron
        neuron_model_init_parameter_dict = self.neuron_model_init_parameter_dict[
            pop_name
        ]
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
        pop_init_parameter_dict = self.pop_init_parameter_dict[pop_name]
        ### replace the neuron model with the new neuron model
        pop_init_parameter_dict["neuron"] = neuron_model_new
        ### replace the size with the reduced size (if reduced size is smaller than the
        ### original size)
        ### TODO add model requirements somewhere, here requirements = geometry = int
        pop_init_parameter_dict["geometry"] = min(
            [pop_init_parameter_dict["geometry"], self.reduced_size]
        )
        ### create the new population
        pop_new = Population(**pop_init_parameter_dict)

        ### 4th set the parameters and variables of the population's neurons
        ### get the stored parameters and variables
        neuron_model_attr_dict = self.neuron_model_attr_dict[pop_name]
        ### set the parameters and variables
        for attr_name, attr_val in neuron_model_attr_dict.items():
            setattr(pop_new, attr_name, attr_val)

        ### 5th if the population is a presynaptic population create an auxiliary
        ### population to calculate the spike probability
        if is_presynaptic:
            Population(
                1,
                neuron=self.SpikeProbCalcNeuron(
                    pre_size=pop_init_parameter_dict["geometry"]
                ),
                name=f"{pop_name}_auxspikeprob",
            )

        ### 6th if the population is a postsynaptic population create an auxiliary
        ### population to calculate the incoming auxillary population input
        ### for the ampa and gaba conductance
        if ampa:
            Population(
                pop_init_parameter_dict["geometry"],
                neuron=self.SpikeProbCalcNeuron(
                    pre_size=pop_init_parameter_dict["geometry"]
                ),
                name=f"{pop_name}_auxinputexc",
            )

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
            if self.get_line_is_dvardt(line, "g_ampa") and ampa:
                ### add " + tau_ampa*g_incomingauxexc/dt"
                ### TODO add model requirements somewhere, here requirements = tau_ampa * dg_ampa/dt = -g_ampa
                equations_line_split_list[line_idx] = self.add_term_to_eq_line(
                    line=equations_line_split_list[line_idx],
                    term=" + tau_ampa*g_incomingauxexc/dt",
                )
            if self.get_line_is_dvardt(line, "g_gaba") and gaba:
                ### add " + tau_gaba*g_incomingauxinh/dt"
                ### TODO add model requirements somewhere, here requirements = tau_gaba * dg_gaba/dt = -g_gaba
                equations_line_split_list[line_idx] = self.add_term_to_eq_line(
                    line=equations_line_split_list[line_idx],
                    term=" + tau_gaba*g_incomingauxinh/dt",
                )
        ### join list to a string
        neuron_model_init_parameter_dict["equations"] = "\n".join(
            equations_line_split_list
        )

        ### 3rd extend description
        neuron_model_init_parameter_dict["description"] = (
            f"{neuron_model_init_parameter_dict['description']}\nWith incoming auxillary population input implemented."
        )

        return neuron_model_init_parameter_dict

    def add_term_to_eq_line(self, line: str, term: str):
        """
        Add a term to an equation string.

        Args:
            line (str):
                Equation string
            term (str):
                Term to add

        Returns:
            line_new (str):
                Equation string with added term
        """
        ### check if colon is in line
        if ":" not in line:
            ### add term
            line_new = line + term
        else:
            ### split line at colon
            line_split = line.split(":")
            ### add term
            line_split[0] = line_split[0] + term
            ### join line again
            line_new = ":".join(line_split)
        ### return new line
        return line_new

    def get_line_is_dvardt(self, line: str, var_name: str):
        """
        Check if a equation string contains dvar/dt.

        Args:
            line (str):
                Equation string
        """
        if "var_name" not in line:
            return False

        ### remove whitespaces
        line = line.replace(" ", "")

        ### check if dvar/dt is in line and before "="
        if f"d{var_name}/dt" in line and line.find(f"d{var_name}/dt") < line.find("="):
            return True

        return False

    def add_DBS_to_rate_coded_neuron_model(self, neuron_model_init_parameter_dict):
        """
        Add DBS mechanisms to the rate-coded neuron model

        Args:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron

        Returns:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron
                with DBS mechanisms added
        """

        ### 1st add new DBS parameters
        ### get the parameters as a list of strings
        parameters_line_split_list = str(
            neuron_model_init_parameter_dict["parameters"]
        ).splitlines()
        ### append list with new parameters
        parameters_line_split_list.append("dbs_depolarization = 0 : population")
        parameters_line_split_list.append("dbs_on = 0")
        parameters_line_split_list.append(
            "axon_rate_amp = 1.0 : population # equivalent to prob_axon_spike in spiking model"
        )
        ### join list to a string
        neuron_model_init_parameter_dict["parameters"] = "\n".join(
            parameters_line_split_list
        )

        ### 2nd add new equations
        ### get the equations of the neuron model as a list of strings
        equations_line_split_list = str(
            neuron_model_init_parameter_dict["equations"]
        ).splitlines()
        ### append axon_rate
        equations_line_split_list.append(
            "axon_rate = axon_rate_amp*dbs_on # equivalent to axon_spike in spiking model"
        )
        ### search for equation with dmp/dt
        lines_with_mp_count = 0
        for line_idx, line in enumerate(equations_line_split_list):
            if self.get_line_is_dmpdt(line):
                ### add depolarization term
                equations_line_split_list[line_idx] = self.add_term_to_eq_line(
                    line=equations_line_split_list[line_idx],
                    term=" + pulse(t)*dbs_on*dbs_depolarization*neg(-1 - mp)",
                )
                lines_with_mp_count += 1
        if lines_with_mp_count == 0:
            raise ValueError(
                "No line with dmp/dt found, only rate-coded models with mp as 'membrane potential' supported yet"
            )
        ### join list to a string
        neuron_model_init_parameter_dict["equations"] = "\n".join(
            equations_line_split_list
        )

        ### 3rd extend description
        neuron_model_init_parameter_dict["description"] = (
            f"{neuron_model_init_parameter_dict['description']}\nWith DBS mechanisms implemented."
        )

        return neuron_model_init_parameter_dict

    def recreate_projection(self, proj_name):
        """
        Recreate a projection with the same synapse model and parameters and connector
        function.

        Args:
            proj_name (str):
                Name of the projection to recreate
        """

        ### 1st recreate synapse model
        ### get the stored parameters of the __init__ function of the Synapse
        synapse_init_parameter_dict = self.synapse_init_parameter_dict[proj_name]
        ### get the stored parameters of the __init__ function of the Projection
        proj_init_parameter_dict = self.proj_init_parameter_dict[proj_name]
        ### adjust the equations and paramters of the synapse model to implement DBS
        synapse_init_parameter_dict = self.add_DBS_to_synapse_model(
            synapse_init_parameter_dict,
        )
        ### create the new synapse model
        synapse_new = Synapse(**synapse_init_parameter_dict)

        ### 2nd recreate projection
        ### replace the synapse model with the new synapse model
        proj_init_parameter_dict["synapse"] = synapse_new
        ### replace pre and post to new populations
        proj_init_parameter_dict["pre"] = get_population(
            self.pre_post_pop_name_dict[proj_name][0]
        )
        proj_init_parameter_dict["post"] = get_population(
            self.pre_post_pop_name_dict[proj_name][1]
        )
        ### create the new projection
        proj_new = Projection(**proj_init_parameter_dict)

        ### 3rd connect the projection with the connector function
        ### get the connector function
        connector_function = self.connector_function_dict[proj_name]
        ### get the parameters of the connector function
        connector_function_parameter_dict = self.connector_function_parameter_dict[
            proj_name
        ]
        ### connect the projection
        connector_function(proj_new, **connector_function_parameter_dict)

        ### 4th set the parameters and variables of the synapse
        ### get the stored parameters and variables
        synapse_model_attr_dict = self.synapse_model_attr_dict[proj_name]
        ### set the parameters and variables
        for attr_name, attr_val in synapse_model_attr_dict.items():
            setattr(proj_new, attr_name, attr_val)

    def add_DBS_to_synapse_model(self, synapse_init_parameter_dict):
        """
        Add DBS mechanisms to the synapse model.

        Args:
            synapse_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Synapse

        Returns:
            synapse_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Synapse
                with DBS mechanisms added
        """

        ### check if projection is spiking
        spiking = not (isinstance(synapse_init_parameter_dict["pre_spike"], type(None)))

        ### add DBS mechanisms
        if spiking:
            return self.add_DBS_to_spiking_synapse_model(synapse_init_parameter_dict)
        else:
            return self.add_DBS_to_rate_coded_synapse_model(synapse_init_parameter_dict)

    def add_DBS_to_spiking_synapse_model(self, synapse_init_parameter_dict):
        """
        Add DBS mechanisms to the spiking synapse model.

        Args:
            synapse_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Synapse

        Returns:
            synapse_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Synapse
                with DBS mechanisms added
        """

        ### 1st add new DBS parameters
        ### get the parameters as a list of strings
        parameters_line_split_list = str(
            synapse_init_parameter_dict["parameters"]
        ).splitlines()
        ### append list with new parameters
        parameters_line_split_list.append("p_axon_spike_trans=0 : projection")
        ### join list to a string
        synapse_init_parameter_dict["parameters"] = "\n".join(
            parameters_line_split_list
        )

        ### 2nd add new equation for uniform variable
        ### get the equations of the synapse model as a list of strings
        equations_line_split_list = str(
            synapse_init_parameter_dict["equations"]
        ).splitlines()
        ### prepend uniform variable
        equations_line_split_list.insert(0, "unif_var_dbs = Uniform(0., 1.)")
        ### join list to a string
        synapse_init_parameter_dict["equations"] = "\n".join(equations_line_split_list)

        ### 3rd add pre_axon_spike
        synapse_init_parameter_dict["pre_axon_spike"] = (
            "g_target+=ite(unif_var_dbs<p_axon_spike_trans,w*post.dbs_on,0)"
        )

        ### 4th extend description
        synapse_init_parameter_dict["description"] = (
            f"{synapse_init_parameter_dict['description']}\nWith DBS mechanisms implemented."
        )

        return synapse_init_parameter_dict

    def add_DBS_to_rate_coded_synapse_model(self, synapse_init_parameter_dict):
        """
        Add DBS mechanisms to the rate-coded synapse model.

        Args:
            synapse_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Synapse

        Returns:
            synapse_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Synapse
                with DBS mechanisms added
        """

        ### 1st add new DBS parameters
        ### get the parameters as a list of strings
        parameters_line_split_list = str(
            synapse_init_parameter_dict["parameters"]
        ).splitlines()
        ### append list with new parameters
        parameters_line_split_list.append("p_axon_spike_trans=0 : projection")
        ### join list to a string
        synapse_init_parameter_dict["parameters"] = "\n".join(
            parameters_line_split_list
        )

        ### 2nd add new equations and replace pre.r
        ### get the equations of the synapse model as a list of strings
        equations_line_split_list = str(
            synapse_init_parameter_dict["equations"]
        ).splitlines()
        ### replace pre.r with pre_rate everywhere
        for key, val in synapse_init_parameter_dict.items():
            if isinstance(val, str):
                synapse_init_parameter_dict[key] = val.replace("pre.r", "pre_rate")
        ### prepend pre_rate definition
        equations_line_split_list.insert(
            0, "pre_rate = pre.r + p_axon_spike_trans*pre.axon_rate*post.dbs_on"
        )
        ### join list to a string
        synapse_init_parameter_dict["equations"] = "\n".join(equations_line_split_list)

        ### 3rd extend description
        synapse_init_parameter_dict["description"] = (
            f"{synapse_init_parameter_dict['description']}\nWith DBS mechanisms implemented."
        )

        return synapse_init_parameter_dict

    class SpikeProbCalcNeuron(Neuron):
        def __init__(self, pre_size=1):
            parameters = f"""
                pre_size = {pre_size} : population
                tau= 1.0 : population
            """
            equations = """
                tau*dr/dt = g_ampa/pre_size - r
                g_ampa = 0
            """
            super().__init__(parameters=parameters, equations=equations)
