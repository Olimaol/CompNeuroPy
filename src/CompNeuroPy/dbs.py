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


class _CreateDBSmodel:
    """
    Class to create a model for testing DBS. The attributes are the same as the
    parameters of the __init__ function but hte Populations and Projections have been
    replaced by the Populations and Projections of the new model with DBS mechanisms.

    Attributes:
        stimulated_population (Population):
            Population which is stimulated by DBS
        excluded_populations_list (list):
            List of populations which are excluded from DBS effects
        passing_fibres_list (list):
            List of projections which are passing fibres
        axon_rate_amp (float | dict):
            Amplitude of the axon rate, either a float or a dict with keys being
            populations (or the string "default") and values being floats.
    """

    def __init__(
        self,
        stimulated_population: Population,
        excluded_populations_list: list[Population],
        passing_fibres_list: list[Projection],
        axon_rate_amp: float | dict[Population | str, float],
    ) -> None:
        """
        Prepare model for DBS stimulation

        Args:
            stimulated_population (Population):
                Population which is stimulated by DBS
            excluded_populations_list (list):
                List of populations which are excluded from DBS effects
            passing_fibres_list (list):
                List of projections which are passing fibres
            axon_rate_amp (float | dict):
                Amplitude of the axon rate, either a float or a dict with keys being
                populations (or the string "default") and values being floats.
        """

        ### analyze model to be able to recreate it
        self.analyze_model(
            stimulated_population,
            excluded_populations_list,
            passing_fibres_list,
            axon_rate_amp,
        )

        ### clear model
        mf.cnp_clear(functions=False, neurons=True, synapses=True, constants=False)

        ### recreate model with DBS mechanisms
        self.recreate_model()

        ### get variables containing Populations and Projections
        self.stimulated_population: Population = get_population(
            self.stimulated_population_name
        )
        self.excluded_populations_list: list[Population] = [
            get_population(pop_name) for pop_name in self.excluded_populations_name_list
        ]
        self.passing_fibres_list: list[Projection] = [
            get_projection(proj_name) for proj_name in self.passing_fibres_name_list
        ]
        if isinstance(self.axon_rate_amp, type(None)):
            ### self.axon_rate_amp is None --> use the axon_rate_amp_pop_name_list and axon_rate_amp_value_list to create the dict
            self.axon_rate_amp: dict[Population | str, float] = {
                ### key is either a Populaiton or the string "default"
                get_population(pop_name[4:])
                if pop_name.startswith("pop;")
                else pop_name: axon_rate_amp_val
                for pop_name, axon_rate_amp_val in zip(
                    self.axon_rate_amp_pop_name_list, self.axon_rate_amp_value_list
                )
            }

    def analyze_model(
        self,
        stimulated_population: Population,
        excluded_populations_list: list[Population],
        passing_fibres_list: list[Projection],
        axon_rate_amp: float | dict[Population | str, float],
    ):
        """
        Analyze the model to be able to recreate it.

        Args:
            stimulated_population (Population):
                Population which is stimulated by DBS
            excluded_populations_list (list):
                List of populations which are excluded from DBS effects
            passing_fibres_list (list):
                List of projections which are passing fibres
            axon_rate_amp (float | dict):
                Amplitude of the axon rate, either a float or a dict with keys being
                populations (or the string "default") and values being floats.
        """

        ### get names of Populations and Projections used by variables to retrieve them later
        self.stimulated_population_name = stimulated_population.name
        self.excluded_populations_name_list = [
            pop.name for pop in excluded_populations_list
        ]
        self.passing_fibres_name_list = [proj.name for proj in passing_fibres_list]
        if isinstance(axon_rate_amp, float):
            self.axon_rate_amp = axon_rate_amp
            self.axon_rate_amp_pop_name_list = []
            self.axon_rate_amp_value_list = []
        elif isinstance(axon_rate_amp, dict):
            self.axon_rate_amp = None
            ### get the axon_rate_amp_pop_name_list
            ### if key is a Population, use the name of the Population and prepend pop;
            ### if key is the string "default", use the string
            self.axon_rate_amp_pop_name_list = [
                f"pop;{axon_rate_amp_key.name}"
                if isinstance(axon_rate_amp_key, Population)
                else axon_rate_amp_key
                for axon_rate_amp_key in axon_rate_amp.keys()
            ]
            self.axon_rate_amp_value_list = list(axon_rate_amp.values())

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
            connector_function_parameter_dict[
                proj.name
            ] = self.get_connector_parameters(proj)

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
        Recreates the model with DBS mechanisms.
        """
        ### recreate populations
        for pop_name in self.population_name_list:
            self.recreate_population(pop_name)
        ### recreate projections
        for proj_name in self.projection_name_list:
            self.recreate_projection(proj_name)

    def recreate_population(self, pop_name):
        """
        Recreate a population with the same neuron model and parameters.

        Args:
            pop_name (str):
                Name of the population to recreate
        """

        ### 1st recreate neuron model
        ### get the stored parameters of the __init__ function of the Neuron
        neuron_model_init_parameter_dict = self.neuron_model_init_parameter_dict[
            pop_name
        ]
        ### adjust these parameters to implement DBS
        neuron_model_init_parameter_dict = self.add_DBS_to_neuron_model(
            neuron_model_init_parameter_dict
        )
        ### create the new neuron model
        neuron_model_new = Neuron(**neuron_model_init_parameter_dict)

        ### 2nd recreate the population
        ### get the stored parameters of the __init__ function of the Population
        pop_init_parameter_dict = self.pop_init_parameter_dict[pop_name]
        ### replace the neuron model with the new neuron model
        pop_init_parameter_dict["neuron"] = neuron_model_new
        ### create the new population
        pop_new = Population(**pop_init_parameter_dict)

        ### 3rd set the parameters and variables of the population's neurons
        ### get the stored parameters and variables
        neuron_model_attr_dict = self.neuron_model_attr_dict[pop_name]
        ### set the parameters and variables
        for attr_name, attr_val in neuron_model_attr_dict.items():
            setattr(pop_new, attr_name, attr_val)

    def add_DBS_to_neuron_model(self, neuron_model_init_parameter_dict):
        """
        Add DBS mechanisms to the neuron model.

        Args:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron

        Returns:
            neuron_model_init_parameter_dict (dict):
                Dictionary with the parameters of the __init__ function of the Neuron
                with DBS mechanisms added
        """

        ### get the type of the neuron --> DBS mechanisms different for spiking and rate-coded neurons
        spiking = not (
            isinstance(neuron_model_init_parameter_dict["spike"], type(None))
        )

        ### add DBS mechanisms
        if spiking:
            return self.add_DBS_to_spiking_neuron_model(
                neuron_model_init_parameter_dict
            )
        else:
            return self.add_DBS_to_rate_coded_neuron_model(
                neuron_model_init_parameter_dict
            )

    def add_DBS_to_spiking_neuron_model(self, neuron_model_init_parameter_dict):
        """
        Add DBS mechanisms to the spiking neuron model

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
        parameters_line_split_list.append("antidromic = 0 : population")
        parameters_line_split_list.append("antidromic_prob = 0 : population")
        parameters_line_split_list.append("prob_axon_spike = 0 : population")
        ### join list to a string
        neuron_model_init_parameter_dict["parameters"] = "\n".join(
            parameters_line_split_list
        )

        ### 2nd add new equations
        ### get the equations of the neuron model as a list of strings
        equations_line_split_list = str(
            neuron_model_init_parameter_dict["equations"]
        ).splitlines()
        ### prepend uniform variables
        equations_line_split_list.insert(0, "unif_var_dbs1 = Uniform(0.0, 1.0)")
        equations_line_split_list.insert(0, "unif_var_dbs2 = Uniform(0.0, 1.0)")
        ### search for equation with dv/dt
        lines_with_v_count = 0
        for line_idx, line in enumerate(equations_line_split_list):
            if self.get_line_is_dvdt(line):
                ### add depolarization term
                equations_line_split_list[line_idx] = self.add_term_to_eq_line(
                    line=equations_line_split_list[line_idx],
                    term=" + pulse(t)*dbs_on*dbs_depolarization*neg(-90 - v)",
                )
                ### increase counter
                lines_with_v_count += 1
        if lines_with_v_count == 0:
            raise ValueError(
                "No line with dv/dt found, only Izhikevich spiking models supported yet"
            )
        ### join list to a string
        neuron_model_init_parameter_dict["equations"] = "\n".join(
            equations_line_split_list
        )

        ### 3rd add axon spike term
        neuron_model_init_parameter_dict[
            "axon_spike"
        ] = "pulse(t)*dbs_on*unif_var_dbs1 > 1-prob_axon_spike"

        ### 4th add axon reset term
        neuron_model_init_parameter_dict[
            "axon_reset"
        ] = """
            v += ite(unif_var_dbs2 < antidromic_prob, dbs_on*antidromic*(-v + c), 0)
            u += ite(unif_var_dbs2 < antidromic_prob, dbs_on*antidromic*d, 0)
        """

        ### 5th extend description
        neuron_model_init_parameter_dict[
            "description"
        ] = f"{neuron_model_init_parameter_dict['description']}\nWith DBS mechanisms implemented."

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

    def get_line_is_dvdt(self, line: str):
        """
        Check if a equation string contains dv/dt.

        Args:
            line (str):
                Equation string
        """
        if "v" not in line:
            return False

        ### remove whitespaces
        line = line.replace(" ", "")

        ### check if dv/dt is in line and check if dv/dt is left of =
        if "dv/dt" in line and line.find("dv/dt") < line.find("="):
            return True

        return False

    def get_line_is_dmpdt(self, line: str):
        """
        Check if a equation string contains dmp/dt.

        Args:
            line (str):
                Equation string
        """
        if "mp" not in line:
            return False

        ### remove whitespaces
        line = line.replace(" ", "")

        ### check if dv/dt is in line and check if dv/dt is left of =
        if "dmp/dt" in line and line.find("dmp/dt") < line.find("="):
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
        neuron_model_init_parameter_dict[
            "description"
        ] = f"{neuron_model_init_parameter_dict['description']}\nWith DBS mechanisms implemented."

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
        synapse_init_parameter_dict[
            "pre_axon_spike"
        ] = "g_target+=ite(unif_var_dbs<p_axon_spike_trans,w*post.dbs_on,0)"

        ### 4th extend description
        synapse_init_parameter_dict[
            "description"
        ] = f"{synapse_init_parameter_dict['description']}\nWith DBS mechanisms implemented."

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
        synapse_init_parameter_dict[
            "description"
        ] = f"{synapse_init_parameter_dict['description']}\nWith DBS mechanisms implemented."

        return synapse_init_parameter_dict


class _CreateDBSmodelcnp(_CreateDBSmodel):
    """
    Class to create a model for testing DBS. Using only populations and projections from
    the given CompNeuroPy-model. The attributes are the same as the parameters of the
    __init__ function but hte Populations and Projections have been replaced by the
    Populations and Projections of the new model with DBS mechanisms.

    Attributes:
        model (CompNeuroPy Model):
            DBS model
        stimulated_population (Population):
            Population which is stimulated by DBS
        excluded_populations_list (list):
            List of populations which are excluded from DBS effects
        passing_fibres_list (list):
            List of projections which are passing fibres
        axon_rate_amp (float | dict):
            Amplitude of the axon rate, either a float or a dict with keys being
            populations (or the string "default") and values being floats.
    """

    def __init__(
        self,
        model: generate_model,
        stimulated_population: Population,
        excluded_populations_list: list[Population],
        passing_fibres_list: list[Projection],
        axon_rate_amp: float | dict[Population | str, float],
    ) -> None:
        """
        Prepare model for DBS stimulation.

        Args:
            model (generate_model):
                CompNeuroPy model
            stimulated_population (Population):
                Population which is stimulated by DBS
            excluded_populations_list (list):
                List of populations which are excluded from DBS effects, they are not
                affected and their axons do not generate axon spikes
            passing_fibres_list (list):
                List of projections which are passing fibres
            axon_rate_amp (float | dict):
                Amplitude of the axon rate, either a float or a dict with keys being
                populations (or the string "default") and values being floats.
        """
        self.model = model
        super().__init__(
            stimulated_population,
            excluded_populations_list,
            passing_fibres_list,
            axon_rate_amp,
        )

    def recreate_model(self):
        """
        Recreates the model with DBS mechanisms.
        Creates a new CompNeuroPy model by using the recreate_model function of the
        parent class as model_creation_function. The new model can be accessed via the
        model attribute.
        """
        self.model = generate_model(
            model_creation_function=super().recreate_model,
            name=f"{self.model.name}_dbs",
            description=f"{self.model.description}\nWith DBS mechanisms implemented.",
            do_compile=False,
            compile_folder_name=f"{self.model.compile_folder_name}_dbs",
        )

    def get_all_population_and_projection_names(self):
        """
        Get all population and projection names from the CompNeuroPy model.

        Returns:
            population_name_list (list):
                List of all population names
            projection_name_list (list):
                List of all projection names
        """
        population_name_list: list[str] = self.model.populations
        projection_name_list: list[str] = self.model.projections

        return population_name_list, projection_name_list


class DBSstimulator:
    """
    Class for stimulating a population with DBS.

    !!! warning
        If you use auto_implement, pointers to the populations and projections of
        the model are not valid anymore (new populations and projections are
        created)! Use a CompNeuroPy model working with names of populations and
        projections anyway (recommended) or use the update_pointers method.

    Examples:
        ```python
        from ANNarchy import Population, Izhikevich, compile, simulate, setup
        from CompNeuroPy import DBSstimulator

        # setup ANNarchy
        setup(dt=0.1)

        # create populations
        population1 = Population(10, neuron=Izhikevich, name="my_pop1")
        population2 = Population(10, neuron=Izhikevich, name="my_pop2")
        >>>
        # create DBS stimulator
        dbs = DBSstimulator(
            stimulated_population=population1,
            population_proportion=0.5,
            dbs_depolarization=30,
            auto_implement=True,
        )

        # update pointers to correct populations
        population1, population2 = dbs.update_pointers(
            pointer_list=[population1, population2]
        )

        # compile network
        compile()

        # run simulation
        # 1000 ms without dbs
        simulate(1000)
        # 1000 ms with dbs
        dbs.on()
        simulate(1000)
        # 1000 ms without dbs
        dbs.off()
        simulate(1000)
        ```
    """

    @check_types()
    def __init__(
        self,
        stimulated_population: Population,
        population_proportion: float = 1.0,
        excluded_populations_list: list[Population] = [],
        dbs_depolarization: float = 0.0,
        orthodromic: bool = False,
        antidromic: bool = False,
        efferents: bool = False,
        afferents: bool = False,
        passing_fibres: bool = False,
        passing_fibres_list: list[Projection] = [],
        passing_fibres_strength: float | list[float] = 1.0,
        sum_branches: bool = True,
        dbs_pulse_frequency_Hz: float = 130.0,
        dbs_pulse_width_us: float = 300.0,
        axon_spikes_per_pulse: float = 1.0,
        axon_rate_amp: float | dict[Population | str, float] = 1.0,
        seed: int | None = None,
        auto_implement: bool = False,
        model: generate_model | None = None,
    ) -> None:
        """
        Initialize DBS stimulator.

        !!! warning
            Do this before compiling the model!

        Args:
            stimulated_population (Population):
                Population which is stimulated by DBS
            population_proportion (float, optional):
                Proportion of the stimulated population which is affected by DBS,
                neurons are distributed randomly. Default: 1.0.
            excluded_populations_list (list, optional):
                List of populations which are excluded from DBS effects, they are not
                affected and their axons do not generate axon spikes. Default: [].
            dbs_depolarization (float, optional):
                Depolarization effect of the DBS pulse to local soma. Default: 0.0.
            orthodromic (bool, optional):
                If True, DBS causes axonal spikes which are forwarded orthodromically.
                Default: False.
            antidromic (bool, optional):
                If True, DBS causes axonal spikes which are forwarded antidromically,
                only available in spiking networks. Default: False.
            efferents (bool, optional):
                If True, DBS affects the efferents of the stimulated population
                (orthodromic and/or antidromic have to be True too). Default: False.
            afferents (bool, optional):
                If True, DBS affects the afferents of the stimulated population
                (orthodromic and/or antidromic have to be True too). Default: False.
            passing_fibres (bool, optional):
                If True, DBS affects the passing fibres of the stimulated region defined
                in passing_fibres_list (orthodromic and/or antidromic have to be True
                too). Default: False.
            passing_fibres_list (list of Projections, optional):
                List of projections which pass the DBS stimulated region and therefore
                are activated by DBS. Default: [], also set passing_fibres True!
            passing_fibres_strength (float or list of float, optional):
                Single value or list of float values between 0 and 1 defining how strong
                the passing fibres are activated by DBS (0: not activated, 1: fully
                activated like the projections in the DBS stimulated region).
                Default: 1.
            sum_branches (bool, optional):
                If True, the antidromic_prob of a presynaptic population (defining how
                many axon spikes affect the pop antidromically) of passing fibres is
                the sum of the passing_fibres_strengths of the single axon branches.
                Default: True.
            dbs_pulse_frequency_Hz (float, optional):
                Frequency of the DBS pulse. Default: 130 Hz.
            dbs_pulse_width_us (float, optional):
                Width of the DBS pulse. Default: 300 us.
            axon_spikes_per_pulse (float, optional):
                Number of average axon spikes per DBS pulse. Default: 1.
            axon_rate_amp (float or dict of float, optional):
                Similar to prob_axon_spike in spiking model. Which rate is forwarded on
                axons caused by DBS. You can specify this for each population
                individually by using a dictionary (keys = Population instances)
                axon_rate_amp = {pop: 1.5} --> the efferent axons of pop forward a rate
                of 1.5 during DBS (all other affected projections forward the default
                value)
                You can specify the default value by using the key "default", e.g.
                {pop: 1.5, "default": 1.0} -> efferent axons of all populations except
                pop forward a rate of 1.0 during DBS. Default: 1.0.
            seed (int, optional):
                Seed for the random distribution of affected neurons based on
                population_proportion. Default: None.
            auto_implement (bool, optional):
                If True, automatically implement DBS mechanisms to the model. Only
                supported for Izhikevich spiking models and rate-coded models.
                Default: False.
                TODO test what happens with mixed models
            model (generate_model, optional):
                CompNeuroPy model which is used to automatically implement DBS
                mechanisms, should not be compiled!. Default: None, i.e., use all
                populations and projections of the current magic model
        """

        if auto_implement:
            ### recreate model with DBS mechanisms
            ### give all variables containing Populations and Projections
            ### and also recreate them during recreating the model
            ### variables are:
            ### - stimulated_population
            ### - excluded_populations_list
            ### - passing_fibres_list
            ### - axon_rate_amp
            if not isinstance(model, type(None)):
                ### CompNeuroPy model given
                ### recreate model with DBS mechanisms
                create_dbs_model_obj = _CreateDBSmodelcnp(
                    model,
                    stimulated_population,
                    excluded_populations_list,
                    passing_fibres_list,
                    axon_rate_amp,
                )
                ### get the new CompNeuroPy model
                model = create_dbs_model_obj.model
            else:
                ### no CompNeuroPy model given --> use all populations and projections of the current magic model
                ### recreate model with DBS mechanisms
                create_dbs_model_obj = _CreateDBSmodel(
                    stimulated_population,
                    excluded_populations_list,
                    passing_fibres_list,
                    axon_rate_amp,
                )
            ### get the new variables containing Populations and Projections
            stimulated_population = create_dbs_model_obj.stimulated_population
            excluded_populations_list = create_dbs_model_obj.excluded_populations_list
            passing_fibres_list = create_dbs_model_obj.passing_fibres_list
            axon_rate_amp = create_dbs_model_obj.axon_rate_amp

        ### set parameters
        self.stimulated_population = stimulated_population
        self.population_proportion = population_proportion
        self.excluded_populations_list = excluded_populations_list
        self.dbs_depolarization = dbs_depolarization
        self.orthodromic = orthodromic
        self.antidromic = antidromic
        self.efferents = efferents
        self.afferents = afferents
        self.passing_fibres = passing_fibres
        self.passing_fibres_list = passing_fibres_list
        self.passing_fibres_strength = passing_fibres_strength
        self.sum_branches = sum_branches
        self.dbs_pulse_width_us = dbs_pulse_width_us
        self.axon_spikes_per_pulse = axon_spikes_per_pulse
        self.axon_rate_amp = axon_rate_amp
        self.seed = seed
        self.model = model

        ### ANNarchy constants for DBS
        self._set_constants(dbs_pulse_frequency_Hz)

        ### randomly select affected neurons i.e. create dbs_on_array
        self.dbs_on_array = self._create_dbs_on_array(population_proportion, seed)

    def _create_dbs_on_array(self, population_proportion: float, seed: int):
        """
        Create an array with the shape of the stimulated population with ones and zeros
        randomly distributed with the specified population_proportion.

        Args:
            population_proportion (float):
                Proportion of the stimulated population which is affected by DBS,
                neurons are distributed randomly
            seed (int):
                Seed for the random distribution of affected neurons based on
                population_proportion

        Returns:
            dbs_on_array (np.array):
                Array with the shape of the stimulated population with ones and zeros
                randomly distributed with the specified population_proportion
        """
        ### create random number generator
        rng = np.random.default_rng(seed)
        ### create an array of zeros with the shape of the population, then flatten it
        dbs_on_array = np.zeros(self.stimulated_population.geometry).flatten()
        ### get the number of affected neurons based on the population_proportion
        number_of_affected_neurons = population_proportion * dbs_on_array.size
        ### randomly ceil or floor the number of affected neurons
        number_of_affected_neurons = int(
            rng.choice(
                [
                    np.ceil(number_of_affected_neurons),
                    np.floor(number_of_affected_neurons),
                ]
            )
        )
        ### insert ones to the dbs_on_array
        dbs_on_array[:number_of_affected_neurons] = 1
        ### shuffle array
        rng.shuffle(dbs_on_array)
        ### reshape array to the shape of the population
        dbs_on_array = dbs_on_array.reshape(self.stimulated_population.geometry)
        ### return array
        return dbs_on_array

    def _set_constants(self, dbs_pulse_frequency_Hz: float):
        """
        Set constants for DBS.

        Args:
            dbs_pulse_frequency_Hz (float):
                Frequency of the DBS pulse in Hz
        """
        # pulse frequency:
        Constant("dbs_pulse_frequency_Hz", dbs_pulse_frequency_Hz)
        # pulse width:
        # Neumant et al.. 2023: 60us but Meier et al. 2022: 300us... 60us = 0.06ms is very small for ANNarchy simulations
        Constant("dbs_pulse_width_us", self.dbs_pulse_width_us)

        ### add global function for DBS pulse
        add_function(
            "pulse(time_ms) = ite(modulo(time_ms*1000, 1000000./dbs_pulse_frequency_Hz) < dbs_pulse_width_us, 1., 0.)"
        )

    def _axon_spikes_per_pulse_to_prob(self, axon_spikes_per_pulse: float):
        """
        Convert number of axon spikes per pulse to probability of axon spikes per
        timestep during DBS pulse

        Args:
            axon_spikes_per_pulse (float):
                Number of average axon spikes per DBS pulse

        Returns:
            prob_axon_spike_time_step (float):
                Probability of axon spikes per timestep during DBS pulse
        """
        return np.clip(
            (axon_spikes_per_pulse * 1000 * dt() / self.dbs_pulse_width_us), 0, 1
        )

    def _set_depolarization(self, dbs_depolarization: float | None = None):
        """
        Set depolarization of population.

        Args:
            dbs_depolarization (float, optional):
                Depolarization effect of the DBS pulse to local soma. Default: None,
                i.e., use value from initialization
        """
        ### either use given depolarization or use default value
        if isinstance(dbs_depolarization, type(None)):
            dbs_depolarization = self.dbs_depolarization

        ### set depolarization of population
        for pop in populations():
            if pop == self.stimulated_population:
                pop.dbs_depolarization = dbs_depolarization
            else:
                pop.dbs_depolarization = 0

    def _set_axon_spikes(
        self,
        orthodromic: bool | None = None,
        antidromic: bool | None = None,
        efferents: bool | None = None,
        afferents: bool | None = None,
        passing_fibres: bool | None = None,
        passing_fibres_strength: float | list[float] | None = None,
        sum_branches: bool | None = None,
        axon_spikes_per_pulse: float | None = None,
        axon_rate_amp: float | dict[Population | str, float] | None = None,
    ):
        """
        Set axon spikes forwarding orthodromic

        Args:
            orthodromic (bool, optional):
                If True, DBS causes axonal spikes which are forwarded orthodromically,
                Default: None, i.e., use value from initialization
            antidromic (bool, optional):
                If True, DBS causes axonal spikes which are forwarded antidromically,
                only available in spiking networks. Default: None, i.e., use value from
                initialization
            efferents (bool, optional):
                If True, DBS affects the efferents of the stimulated population
                (orthodromic and/or antidromic have to be True too). Default: None,
                i.e., use value from initialization
            afferents (bool, optional):
                If True, DBS affects the afferents of the stimulated population
                (orthodromic and/or antidromic have to be True too). Default: None,
                i.e., use value from initialization
            passing_fibres (bool, optional):
                If True, DBS affects the passing fibres of the stimulated region defined
                in passing_fibres_list (orthodromic and/or antidromic have to be True
                too). Default: None, i.e., use value from initialization
            passing_fibres_strength (float | list[float], optional):
                Single value or list of float values between 0 and 1 defining how strong
                the passing fibres are activated by DBS (0: not activated, 1: fully
                activated like the projections in the DBS stimulated region).
                Default: None, i.e., use value from initialization
            sum_branches (bool, optional):
                If True, the antidromic_prob of a presynaptic population (defining how
                many axon spikes affect the pop antidromically) of passing fibres is
                the sum of the passing_fibres_strengths of the single axon branches.
                Default: None, i.e., use value from initialization
            axon_spikes_per_pulse (float, optional):
                Number of average axon spikes per DBS pulse. Default: None, i.e., use
                value from initialization
            axon_rate_amp (float | dict[Population | str, float], optional):
                Similar to prob_axon_spike in spiking model. Which rate is forwarded on
                axons caused by DBS. You can specify this for each population
                individually by using a dictionary (keys = Population instances)
                axon_rate_amp = {pop: 1.5} --> the efferent axons of pop forward a rate
                of 1.5 during DBS (all other affected projections forward the default
                value)
                You can specify the default value by using the key "default", e.g.
                {pop: 1.5, "default": 1.0} -> efferent axons of all populations except
                pop forward a rate of 1.0 during DBS. Default: None, i.e., use value
                from initialization
        """

        ### either use given orthodromic or use default value
        if isinstance(orthodromic, type(None)):
            orthodromic = self.orthodromic
        ### either use given antidromic or use default value
        if isinstance(antidromic, type(None)):
            antidromic = self.antidromic
        ### either use given efferents or use default value
        if isinstance(efferents, type(None)):
            efferents = self.efferents
        ### either use given afferents or use default value
        if isinstance(afferents, type(None)):
            afferents = self.afferents
        ### either use given afferents or use default value
        if isinstance(passing_fibres, type(None)):
            passing_fibres = self.passing_fibres
        ### either use given passing_fibres_strength or use default value
        if isinstance(passing_fibres_strength, type(None)):
            passing_fibres_strength = self.passing_fibres_strength
        ### either use given sum_branches or use default value
        if isinstance(sum_branches, type(None)):
            sum_branches = self.sum_branches
        ### either use given axon_spikes_per_pulse or use default value
        if isinstance(axon_spikes_per_pulse, type(None)):
            axon_spikes_per_pulse = self.axon_spikes_per_pulse
        ### either use given axon_rate_amp or use default value
        if isinstance(axon_rate_amp, type(None)):
            axon_rate_amp = self.axon_rate_amp

        ### check if passing_fibres_strength is a list
        if not isinstance(passing_fibres_strength, list):
            passing_fibres_strength = [passing_fibres_strength] * len(
                self.passing_fibres_list
            )
        ### check if axon_rate_amp is a dict or float
        if isinstance(axon_rate_amp, dict):
            ### check if default key is missing
            if "default" not in axon_rate_amp.keys():
                ### add the key "default" with the value 1.0 to the dict
                axon_rate_amp["default"] = 1.0
        else:
            ### create dict with default value
            axon_rate_amp = {"default": axon_rate_amp}

        ### deactivate DBS axon transmission
        self._deactivate_axon_DBS()

        ### activate orthodromic transmission for all projections
        if orthodromic:
            self._set_orthodromic(
                efferents,
                afferents,
                passing_fibres,
                passing_fibres_strength,
                axon_spikes_per_pulse,
                axon_rate_amp,
            )

        ### activate antidromic transmission for all populations
        if antidromic:
            self._set_antidromic(
                efferents,
                afferents,
                passing_fibres,
                passing_fibres_strength,
                sum_branches,
                axon_spikes_per_pulse,
            )

    def _deactivate_axon_DBS(self):
        """
        Deactivate axon spikes forwarding for both orthodromic and antidromic.
        """
        for pop in populations():
            ### deactivate axon spike genearation for all populations
            pop.prob_axon_spike = 0
            pop.axon_rate_amp = 0
            ### deactivate antidromic transmission for all populations
            pop.antidromic = 0
            pop.antidromic_prob = 0

        ### deactivate orthodromic transmission for all projections
        for proj in projections():
            proj.axon_transmission = 0
            proj.p_axon_spike_trans = 0

    def _set_orthodromic(
        self,
        efferents: bool,
        afferents: bool,
        passing_fibres: bool,
        passing_fibres_strength: list[float],
        axon_spikes_per_pulse: float,
        axon_rate_amp: dict[Population | str, float],
    ):
        """
        Set orthodromic axon spikes forwarding.

        Args:
            efferents (bool):
                If True, DBS affects the efferents of the stimulated population
                (orthodromic and/or antidromic have to be True too)
            afferents (bool):
                If True, DBS affects the afferents of the stimulated population
                (orthodromic and/or antidromic have to be True too)
            passing_fibres (bool):
                If True, DBS affects the passing fibres of the stimulated population
                (orthodromic and/or antidromic have to be True too and there have to
                be passing fibres in the passing_fibres_list)
            passing_fibres_strength (list[float]):
                List of float values between 0 and 1 defining how strong the passing
                fibres are activated by DBS (0: not activated, 1: fully activated
                like projections in DBS stimulated region)
            axon_spikes_per_pulse (float):
                Number of average axon spikes per DBS pulse
            axon_rate_amp (dict[Population | str, float]):
                Similar to prob_axon_spike in spiking model. Which rate is forwarded
                on axons caused by DBS. The dictionary has to contain the key
                "default" with the default value for all projections and can contain
                keys for each population with a different value for the axon_rate of
                the efferent axons of this population.
        """
        if efferents:
            ### activate all efferent projections
            projection_list = projections(pre=self.stimulated_population)
            for proj in projection_list:
                ### skip excluded populations
                if proj.post in self.excluded_populations_list:
                    continue
                ### activate axon transmission
                proj.axon_transmission = 1
                proj.p_axon_spike_trans = 1
                ### set prob_axon_spike for spiking model
                proj.pre.prob_axon_spike = self._axon_spikes_per_pulse_to_prob(
                    axon_spikes_per_pulse
                )
                ### set axon_rate_amp for rate-coded model
                if proj.pre in axon_rate_amp.keys():
                    ### axon_rate_amp is specified for this population
                    proj.pre.axon_rate_amp = axon_rate_amp[proj.pre]
                else:
                    ### axon_rate_amp is not specified for this population, use default value
                    proj.pre.axon_rate_amp = axon_rate_amp["default"]

        if afferents:
            ### activate all afferent projections
            projection_list = projections(post=self.stimulated_population)
            for proj in projection_list:
                ### skip excluded populations
                if proj.pre in self.excluded_populations_list:
                    continue
                ### activate axon transmission
                proj.axon_transmission = 1
                proj.p_axon_spike_trans = 1
                ### set prob_axon_spike for spiking model
                proj.pre.prob_axon_spike = self._axon_spikes_per_pulse_to_prob(
                    axon_spikes_per_pulse
                )
                ### set axon_rate_amp for rate-coded model
                if proj.pre in axon_rate_amp.keys():
                    ### axon_rate_amp is specified for this population
                    proj.pre.axon_rate_amp = axon_rate_amp[proj.pre]
                else:
                    ### axon_rate_amp is not specified for this population, use default value
                    proj.pre.axon_rate_amp = axon_rate_amp["default"]

        if passing_fibres:
            ### activate all passing projections
            for proj_idx, proj in enumerate(self.passing_fibres_list):
                proj.axon_transmission = 1
                proj.p_axon_spike_trans = passing_fibres_strength[proj_idx]
                ### set prob_axon_spike for spiking model
                proj.pre.prob_axon_spike = self._axon_spikes_per_pulse_to_prob(
                    axon_spikes_per_pulse
                )
                ### set axon_rate_amp for rate-coded model
                if proj.pre in axon_rate_amp.keys():
                    ### axon_rate_amp is specified for this population
                    proj.pre.axon_rate_amp = axon_rate_amp[proj.pre]
                else:
                    ### axon_rate_amp is not specified for this population, use default value
                    proj.pre.axon_rate_amp = axon_rate_amp["default"]

    def _set_antidromic(
        self,
        efferents: bool,
        afferents: bool,
        passing_fibres: bool,
        passing_fibres_strength: list[float],
        sum_branches: bool,
        axon_spikes_per_pulse: float,
    ):
        """
        Set antidromic axon spikes forwarding.

        Args:
            efferents (bool):
                If True, DBS affects the efferents of the stimulated population
                (orthodromic and/or antidromic have to be True too)
            afferents (bool):
                If True, DBS affects the afferents of the stimulated population
                (orthodromic and/or antidromic have to be True too)
            passing_fibres (bool):
                If True, DBS affects the passing fibres of the stimulated population
                (orthodromic and/or antidromic have to be True too and there have to
                be passing fibres in the passing_fibres_list)
            passing_fibres_strength (list[float]):
                List of float values between 0 and 1 defining how strong the passing
                fibres are activated by DBS (0: not activated, 1: fully activated
                like projections in DBS stimulated region)
            sum_branches (bool):
                If True, the antidromic_prob of a presynaptic population (defining how
                many axon spikes affect the pop antidromically) of passing fibres is
                the sum of the passing_fibres_strengths of the single axon branches.
            axon_spikes_per_pulse (float):
                Number of average axon spikes per DBS pulse
        """

        if efferents:
            ### activate all efferent projections, i.e. antodromic activation of stimulated population
            pop = self.stimulated_population
            pop.antidromic = 1
            pop.antidromic_prob = 1
            pop.prob_axon_spike = self._axon_spikes_per_pulse_to_prob(
                axon_spikes_per_pulse
            )
        if afferents:
            ### activate all afferent projections, i.e. all presynaptic populations of stimulated population
            ### get presynaptic projections
            projection_list = projections(post=self.stimulated_population)
            ### get presynaptic populations from projections
            presyn_pop_list = []
            presyn_pop_name_list = []
            for proj in projection_list:
                ### skip excluded populations
                if proj.pre in self.excluded_populations_list:
                    continue
                ### check if presynaptic population is already in list
                if proj.pre.name not in presyn_pop_name_list:
                    presyn_pop_name_list.append(proj.pre.name)
                    presyn_pop_list.append(proj.pre)
            ### set antidromic for all presynaptic populations
            for pop in presyn_pop_list:
                pop.antidromic = 1
                pop.antidromic_prob = np.mean(self.stimulated_population.dbs_on)
                pop.prob_axon_spike = self._axon_spikes_per_pulse_to_prob(
                    axon_spikes_per_pulse
                )
        if passing_fibres:
            ### get presynaptic populations from passing fibres projections
            presyn_pop_list = []
            presyn_pop_name_list = []
            for proj in self.passing_fibres_list:
                ### check if presynaptic population is already in list
                if proj.pre.name not in presyn_pop_name_list:
                    presyn_pop_name_list.append(proj.pre.name)
                    presyn_pop_list.append(proj.pre)
            ### get antidomic_prob for each presynatic population with the passing_fibres_strength
            antidromic_prob_list = [0] * len(presyn_pop_list)
            for pop_idx, pop in enumerate(presyn_pop_list):
                ### get all passing fibres and their strength of a presynaptic pop
                passing_fibres_strength_of_pop_list = []
                for proj_idx, proj in enumerate(self.passing_fibres_list):
                    if proj.pre.name == pop.name:
                        passing_fibres_strength_of_pop_list.append(
                            passing_fibres_strength[proj_idx]
                        )
                ### check if the probs of the single axon branches should be summed up
                ### if for example a presynaptic pop contributes to two passing fibres, the axons of the presynaptic pop split up into two branches
                ### thus, if these two branches are both stimulated, they both forward APs antidromic
                ### thus, sum up the antidromic_prob of the single branches to obtain the antidromic spikes which affect the presynaptic pop
                ### if sum_branches is False, then this would represent that the stimulation at the axon is before it splits up into multiple branches and there should not be different passing_fibres_strengths for the same presynaptic pop
                if sum_branches:
                    antidromic_prob_list[pop_idx] = sum(
                        passing_fibres_strength_of_pop_list
                    )
                else:
                    if len(set(passing_fibres_strength_of_pop_list)) != 1:
                        ### list contains different values
                        raise ValueError(
                            "Different passing fibres strengths for the same presynaptic population detected. This is not possible if sum_branches is False."
                        )
                    ### all values are the same, thus take the first one
                    antidromic_prob_list[pop_idx] = passing_fibres_strength_of_pop_list[
                        0
                    ]

                ### TODO
                ### if summing axon branches leads to a prob > 1, then
                ### the prob should be set to 1
                ### the axon spike generation in this pop should be increased
                ### and all axon spike transmissions from this pop should be decreased by the same factor
                ### this is not implemented yet... maybe in the future
                if antidromic_prob_list[pop_idx] > 1:
                    raise ValueError(
                        "Summing the passing fibres strengths of a presynaptic population leads to a antidromic spike probability > 1. This is not possible yet."
                    )

            ### set antidromic for all presynaptic populations
            for pop_idx, pop in enumerate(presyn_pop_list):
                pop.antidromic = 1
                pop.antidromic_prob = antidromic_prob_list[pop_idx]
                pop.prob_axon_spike = self._axon_spikes_per_pulse_to_prob(
                    axon_spikes_per_pulse
                )

    @check_types()
    def on(
        self,
        population_proportion: float | None = None,
        dbs_depolarization: float | None = None,
        orthodromic: bool | None = None,
        antidromic: bool | None = None,
        efferents: bool | None = None,
        afferents: bool | None = None,
        passing_fibres: bool | None = None,
        passing_fibres_strength: float | list[float] | None = None,
        sum_branches: bool | None = None,
        axon_spikes_per_pulse: float | None = None,
        axon_rate_amp: float | dict[Population | str, float] | None = None,
        seed: int | None = None,
    ):
        """
        Activate DBS.

        Args:
            population_proportion (float, optional):
                Proportion of the stimulated population which is affected by DBS,
                neurons are distributed randomly. Default: None, i.e., use value from
                initialization
            dbs_depolarization (float, optional):
                Depolarization effect of the DBS pulse to local soma. Default: None,
                i.e., use value from initialization
            orthodromic (bool, optional):
                If True, DBS causes axonal spikes which are forwarded orthodromically.
                Default: None, i.e., use value from initialization
            antidromic (bool, optional):
                If True, DBS causes axonal spikes which are forwarded antidromically,
                only available in spiking networks. Default: None, i.e., use value from
                initialization
            efferents (bool, optional):
                If True, DBS affects the efferents of the stimulated population
                (orthodromic and/or antidromic have to be True too). Default: None,
                i.e., use value from initialization
            afferents (bool, optional):
                If True, DBS affects the afferents of the stimulated population
                (orthodromic and/or antidromic have to be True too). Default: None,
                i.e., use value from initialization
            passing_fibres (bool, optional):
                If True, DBS affects the passing fibres of the stimulated region defined
                in passing_fibres_list (orthodromic and/or antidromic have to be True
                too). Default: None, i.e., use value from initialization
            passing_fibres_strength (float | list[float], optional):
                Single value or list of float values between 0 and 1 defining how strong
                the passing fibres are activated by DBS (0: not activated, 1: fully
                activated like the projections in the DBS stimulated region).
                Default: None, i.e., use value from initialization
            sum_branches (bool, optional):
                If True, the antidromic_prob of a presynaptic population (defining how
                many axon spikes affect the pop antidromically) of passing fibres is
                the sum of the passing_fibres_strengths of the single axon branches.
                Default: None, i.e., use value from initialization
            axon_spikes_per_pulse (float, optional):
                Number of average axon spikes per DBS pulse. Default: None, i.e., use
                value from initialization
            axon_rate_amp (float | dict[Population | str, float], optional):
                Similar to prob_axon_spike in spiking model. Which rate is forwarded on
                axons caused by DBS. You can specify this for each population
                individually by using a dictionary (keys = Population instances)
                axon_rate_amp = {pop: 1.5} --> the efferent axons of pop forward a rate
                of 1.5 during DBS (all other affected projections forward the default
                value). You can specify the default value by using the key "default",
                e.g. {pop: 1.5, "default": 1.0} -> efferent axons of all populations
                except pop forward a rate of 1.0 during DBS. Default: None, i.e., use
                value from initialization
            seed (int, optional):
                Seed for the random number generator. Default: None, i.e., use value
                from initialization
        """

        ### set DBS on for all populations
        ### also sets the proportion of affected neurons, call this before set_depolarization and set_axon_spikes!
        self._set_dbs_on(population_proportion, seed)

        ### set depolarization of population
        self._set_depolarization(dbs_depolarization)

        ### set axon spikes forwarding
        self._set_axon_spikes(
            orthodromic,
            antidromic,
            efferents,
            afferents,
            passing_fibres,
            passing_fibres_strength,
            sum_branches,
            axon_spikes_per_pulse,
            axon_rate_amp,
        )

    def _set_dbs_on(self, population_proportion: float | None, seed: int | None):
        """
        Set DBS on for all populations, for the stimulated population only the specified
        proportion is affected by DBS.

        Args:
            population_proportion (float, optional):
                Proportion of the stimulated population which is affected by DBS,
                neurons are distributed randomly. Default: None, i.e., use value from
                initialization
            seed (int, optional):
                Seed for the random number generator. Default: None, i.e., use value
                from initialization
        """
        ### set parameters for the creation of the DBS on array
        ### either use given population_proportion or use default value
        if isinstance(population_proportion, type(None)):
            population_proportion = self.population_proportion
        ### either use given seed or use default value
        if isinstance(seed, type(None)):
            seed = self.seed

        ### if seed and population_propotion are the same as in the initialization, use the same dbs_on_array
        if seed == self.seed and population_proportion == self.population_proportion:
            ### use the same dbs_on_array as in the initialization
            dbs_on_array = self.dbs_on_array
        else:
            ### create new dbs_on_array
            dbs_on_array = self._create_dbs_on_array(population_proportion, seed)

        ### set DBS on for all populations
        for pop in populations():
            ### of the stimulated population only the specified proportion is affected by DBS
            if pop == self.stimulated_population:
                pop.dbs_on = dbs_on_array
            else:
                pop.dbs_on = 1

    def off(self):
        """
        Deactivate DBS.
        """
        ### set DBS off for all populations
        for pop in populations():
            pop.dbs_on = 0
            pop.prob_axon_spike = 0
            pop.axon_rate_amp = 0

        ### deactivate DBS axon transmission
        self._deactivate_axon_DBS()

    def update_pointers(self, pointer_list):
        """
        Update pointers to populations and projections after recreating the model.

        Args:
            pointer_list (list):
                List of pointers to populations and projections

        Returns:
            pointer_list_new (list):
                List of pointers to populations and projections of the new model
        """
        ### update pointers
        pointer_list_new: list[Population | Projection] = []
        for pointer in pointer_list:
            compartment_name = pointer.name
            if isinstance(pointer, Population):
                pointer_list_new.append(get_population(compartment_name))
            elif isinstance(pointer, Projection):
                pointer_list_new.append(get_projection(compartment_name))
            else:
                raise TypeError(
                    f"Pointer {pointer} is neither a Population nor a Projection"
                )
        return pointer_list_new
