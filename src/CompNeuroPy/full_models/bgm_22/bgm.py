import numpy as np
from ANNarchy import get_population, get_projection
from ANNarchy.core.Random import (
    Uniform,
    DiscreteUniform,
    Normal,
    LogNormal,
    Exponential,
    Gamma,
)
from CompNeuroPy import CompNeuroModel
import csv
import os
import importlib
import inspect
import traceback
import sys
from typingchecker import check_types


class BGM(CompNeuroModel):
    """
    The basal ganglia model based on the model from [Goenner et al. (2021)](https://doi.org/10.1111/ejn.15082).

    Attributes:
        name (str):
            name of the model
        description (str):
            description of the model
        model_creation_function (function):
            function which creates the model
        compile_folder_name (str):
            name of the folder in which the model is compiled
        model_kwargs (dict):
            keyword arguments for model_creation_function
        populations (list):
            list of names of all populations of the model
        projections (list):
            list of names of all projections of the model
        created (bool):
            True if the model is created
        compiled (bool):
            True if the model is compiled
        attribute_df (pandas dataframe):
            dataframe containing all attributes of the model compartments
        params (dict):
            dictionary containing all parameters of the model
        name_appendix (str):
            string which is appended to all model compartments and parameters
    """

    @check_types()
    def __init__(
        self,
        name: str = "BGM_v01_p01",
        do_create: bool = True,
        do_compile: bool = True,
        compile_folder_name: str | None = None,
        seed: int | None = None,
        name_appendix: str = "",
    ):
        """
        Args:
            name (str, optional):
                name of the model, syntax: "BGM_v<model_version>_p<parameters_version>"
                replace <model_version> and <parameters_version> with the versions you
                want to use, see CompNeuroPy.full_models.BGM_22.parameters for available
                versions. Default: "BGM_v01_p01"
            do_create (bool, optional):
                if True, the model is created after initialization. Default: True
            do_compile (bool, optional):
                if True, the model is compiled after creation. Default: True
            compile_folder_name (str, optional):
                name of the folder in which the compiled model is saved. Default: None,
                i.e. "annarchy_BGM_v<model_version>" is used
            seed (int, optional):
                the seed for the random number generator used during model creation.
                Default: None, i.e. random seed is used
            name_appendix (str, optional):
                string which is appended to all model compartments and parameters.
                Allows to create multiple models with the same name and keep names of
                compartments and parameters unique. Default: ""
        """
        ### check if name is correct, otherwise raise ValueError
        if not (
            len(name.split("_")) == 3
            and name.split("_")[0] == "BGM"
            and name.split("_")[1][0] == "v"
            and name.split("_")[2][0] == "p"
        ):
            raise ValueError(
                "name has to be of the form 'BGM_v<model_version>_p<parameters_version>'"
            )

        ### set attributes (except the ones which are set in the super().__init__())
        self.name_appendix = name_appendix
        self.seed = seed
        if len(self.name_appendix) > 0:
            self._name_appendix_to_add = ":" + name_appendix
        else:
            self._name_appendix_to_add = ""

        ### set model_version_name
        self._model_version_name = "_".join(name.split("_")[:2])

        ### update name with name_appendix
        name = name + self._name_appendix_to_add

        ### init default compile_folder_name
        if compile_folder_name == None:
            compile_folder_name = "annarchy_" + self._model_version_name

        ### set description
        description = (
            "The basal ganglia model based on the model from Goenner et al. (2021)"
        )

        ### init random number generator
        self._rng = np.random.default_rng(seed)

        ### get model parameters before init, ignore name_appendix
        self.params = self._get_params(name.split(":")[0])

        ### init
        super().__init__(
            model_creation_function=self._model_creation_function,
            name=name,
            description=description,
            do_create=do_create,
            do_compile=do_compile,
            compile_folder_name=compile_folder_name,
        )

    def _add_name_appendix(self):
        """
        Rename all model compartments, keys (except general) in params dict and
        names in attribute_df by appending the name_appendix to the original name.
        """

        ### update the attribute_df of the model object (it still contains the original
        ### names of the model creation)
        self.attribute_df["compartment_name"] = (
            self.attribute_df["compartment_name"] + self._name_appendix_to_add
        )
        ### rename populations and projections
        populations_new = []
        for pop_name in self.populations:
            populations_new.append(pop_name + self._name_appendix_to_add)
            get_population(pop_name).name = pop_name + self._name_appendix_to_add
        self.populations = populations_new
        projections_new = []
        for proj_name in self.projections:
            projections_new.append(proj_name + self._name_appendix_to_add)
            get_projection(proj_name).name = proj_name + self._name_appendix_to_add
        self.projections = projections_new
        ### rename parameter keys except general
        params_new = {}
        for key, param_val in self.params.items():
            param_object = key.split(".")[0]
            param_name = key.split(".")[1]

            if param_object == "general":
                params_new[key] = param_val
                continue

            param_object = param_object + self._name_appendix_to_add
            key_new = param_object + "." + param_name
            params_new[key_new] = param_val
        self.params = params_new

    def _model_creation_function(self):
        """
        Creates the model using the model_creation_function from the
        model_creation_functions.py file. The function is defined by the
        model_version_name.
        """
        model_creation_function = eval(
            "importlib.import_module('CompNeuroPy.full_models.bgm_22.model_creation_functions')."
            + self._model_version_name
        )
        model_creation_function(self)

    def create(self, do_compile=True, compile_folder_name=None):
        """
        Creates the model and optionally compiles it directly.

        Args:
            do_compile (bool, optional):
                If True the model is compiled directly. Default: True.
            compile_folder_name (str, optional):
                Name of the folder in which the model is compiled. Default: value from
                initialization.
        """
        ### create the model, but do not compile to set parameters before compilation
        super().create(do_compile=False, compile_folder_name=compile_folder_name)

        ### update names of compartments and parameters
        self._add_name_appendix()

        ### set parameters and connectivity of projections
        ### for each projection the connectivity has to be defined in the params
        self._set_params()
        self._set_noise_values()
        self._set_connections()

        ### compile the model, after setting all parameters (included in compile state)
        if do_compile:
            self.compile(compile_folder_name)

    def _set_params(self):
        """
        sets params of all populations
        """

        ### loop over all params
        for key, param_val in self.params.items():
            ### split key in param object and param name
            param_object = key.split(".")[0]
            param_name = key.split(".")[1]

            ### if param is a noise param --> skip (separate function)
            if param_name.split("_")[-1] == "noise":
                continue

            ### if param name ends with init --> actual param_name (in pop) is without init
            if param_name.split("_")[-1] == "init":
                param_name = "_".join(param_name.split("_")[:-1])

            ### if param_object is a pop in network
            if param_object in self.populations:
                ### and the param_name is an attribute of the pop --> set param of pop
                if param_name in vars(get_population(param_object))["attributes"]:
                    ### if parameter values are given as distribution --> get numpy array
                    if isinstance(param_val, str):
                        if (
                            "Uniform" in param_val
                            or "DiscreteUniform" in param_val
                            or "Normal" in param_val
                            or "LogNormal" in param_val
                            or "Exponential" in param_val
                            or "Gamma" in param_val
                        ):
                            distribution = eval(param_val)
                            param_val = distribution.get_values(
                                shape=get_population(param_object).geometry
                            )
                    self.set_param(
                        compartment=param_object,
                        parameter_name=param_name,
                        parameter_value=param_val,
                    )
                    ### if parameter base_mean --> also set I_base
                    if param_name == "base_mean":
                        self.set_param(
                            compartment=param_object,
                            parameter_name="I_base",
                            parameter_value=param_val,
                        )

    def _set_noise_values(self):
        """
        sets noise params of all populations
        """

        ### loop over all params
        for key, param_val in self.params.items():
            ### split key in param object and param name
            param_object = key.split(".")[0]
            param_name = key.split(".")[1]

            ### if param_object is a pop in network and param_name ends with noise --> set noise param of pop
            if (
                param_object in self.populations
                and param_name.split("_")[-1] == "noise"
            ):
                if param_name == "mean_rate_noise":
                    ### for mean and sd the actual parameter of the pop has to be calculated
                    mean = param_val
                    try:
                        ### noise values defined by mean and sd
                        sd = self.params[param_object + ".rate_sd_noise"]
                    except:
                        ### if only mean is available, only set mean
                        sd = 0
                    if sd != 0:
                        self.set_param(
                            compartment=param_object,
                            parameter_name="rates_noise",
                            parameter_value=self._rng.normal(
                                mean, sd, get_population(param_object).size
                            ),
                        )
                    else:
                        self.set_param(
                            compartment=param_object,
                            parameter_name="rates_noise",
                            parameter_value=mean,
                        )
                elif param_name in vars(get_population(param_object))["attributes"]:
                    ### noise parameters which are actual attributes of the pop are simply set
                    self.set_param(
                        compartment=param_object,
                        parameter_name=param_name,
                        parameter_value=param_val,
                    )
                else:
                    continue

    def _set_connections(self):
        """
        sets the connectivity and parameters of all projections
        """

        ### dict for each projection, which params were already set during connectivity definition
        already_set_params = {}

        ### set connectivity
        ### loop over all projections
        set_con_failed = False
        error_message_list = []
        for proj_name in self.projections:
            ### get the type of connectivity for projection
            try:
                connectivity = self.params[proj_name + ".connectivity"]
            except:
                print(
                    "\nERROR: missing connectivity parameter for",
                    proj_name,
                    "\n",
                    proj_name + ".connectivity",
                    "needed!\n",
                    "parameters id:",
                    self.params["general.id"],
                    "\n",
                )
                quit()

            possible_con_list = [
                "connect_fixed_number_pre",
                "connect_all_to_all",
                "connect_one_to_one",
                "connect_fixed_probability",
            ]
            if connectivity in possible_con_list:
                try:
                    # get all possible parameters of the connectivity function
                    con_func = eval(f"get_projection(proj_name).{connectivity}")
                    possible_con_params_list = list(
                        inspect.signature(con_func).parameters.keys()
                    )
                    # check if paramters are given in the params dict and create the kwargs for the connectivity function
                    con_kwargs = {}
                    for con_param_key in possible_con_params_list:
                        if proj_name + "." + con_param_key in self.params:
                            con_kwargs[con_param_key] = eval(
                                str(self.params[proj_name + "." + con_param_key])
                            )
                    # call the connectivity function with the obtained kwargs
                    con_func(**con_kwargs)
                    # store which parameters have been set
                    already_set_params[proj_name] = list(con_kwargs.keys())
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    error_message = traceback.format_exception_only(exc_type, exc_value)
                    error_message_list.append([f"ERROR: {proj_name}"] + error_message)
                    set_con_failed = True
            else:
                print(
                    "\nERROR: wrong connectivity parameter for",
                    proj_name + ".connectivity!\n",
                    "parameters id:",
                    self.params["general.id"],
                    "possible:",
                    possible_con_list,
                    "\n",
                )
                quit()
        if set_con_failed:
            print("\n")
            for error_message in error_message_list:
                print(" ".join(error_message))
            raise TypeError("Setting connectivities failed")

        ### set parameters
        ### loop over all params
        for key, param_val in self.params.items():
            ### split key in param object and param name
            param_object = key.split(".")[0]
            param_name = key.split(".")[1]

            if param_object == "general":
                continue

            ### if param_object is proj in network and param not already used and param is an attribute of proj --> set param of proj
            if (
                param_object in self.projections
                and not (param_name in already_set_params[param_object])
                and param_name in vars(get_projection(param_object))["attributes"]
            ):
                self.set_param(
                    compartment=param_object,
                    parameter_name=param_name,
                    parameter_value=param_val,
                )

    def _get_params(self, name):
        """
        read all parameters for specified model name

        Args:
            name (str):
                name of the model, specifies which column in the csv file is used
        """

        csvPath = os.path.dirname(os.path.realpath(__file__)) + "/parameters.csv"
        csvfile = open(csvPath, newline="")

        params = {}
        reader = csv.reader(csvfile, delimiter=",")
        fileRows = []
        idx = -1
        ### check if name is in the .csv file
        for row in reader:
            if row[0] == "":
                continue
            fileRows.append(row)
            if "general.id" == row[0] and True in [
                name == row[i] for i in range(1, len(row))
            ]:
                idx = [name == row[i] for i in range(1, len(row))].index(True) + 1
            elif "general.id" == row[0]:
                print(
                    "No Parameters available for given model name "
                    + name
                    + "! (file "
                    + csvPath
                    + ")"
                )
                quit()
        if idx == -1:
            print("No general.id in parameter csv file!")
            quit()
        ### read the column corresponding to name
        for row in fileRows:
            if "###" in row[0]:
                continue
            if row[idx] == "":
                continue

            value = row[idx]
            try:
                ### if float(value) works value is a number --> check if it is int
                if float(value) - int(float(value)) == 0:
                    params[row[0]] = int(float(value))
                else:
                    params[row[0]] = float(value)
            except:
                ### value is a string
                if value[0] == "$" and value[-1] == "$":
                    ### value is a formula
                    params[row[0]] = float(eval(value[1:-1]))
                else:
                    ### value is some other string
                    params[row[0]] = value
        csvfile.close()

        return params

    def _needed_imports(self):
        for import_val in [
            Uniform,
            DiscreteUniform,
            Normal,
            LogNormal,
            Exponential,
            Gamma,
            importlib,
        ]:
            print(import_val)
