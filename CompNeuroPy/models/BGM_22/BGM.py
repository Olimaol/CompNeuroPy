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
from CompNeuroPy import generate_model
import csv
import os
import importlib
import inspect
import traceback
import sys


class BGM(generate_model):
    """
    The basal ganglia model based on the model from Goenner et al. (2021)
    """

    def __init__(
        self,
        name="BGM_v01_p01",
        do_create=True,
        do_compile=True,
        compile_folder_name=None,
        seed=None,
        name_appendix=None,
    ):
        """
        runs the standard init but with already predefined model_creation_function and description
        one can still adjust name, do_compile and compile_folder_name

        seed: int, default=None, the seed for the random number generator used during model creation
        """
        ### check if name is correct
        self._name_split_ = name.split("_")
        assert (
            len(self._name_split_) == 3
            and self._name_split_[0] == "BGM"
            and self._name_split_[1][0] == "v"
            and self._name_split_[2][0] == "p"
        ), 'ERROR generate_model BGM: "name" must have form "BGM_vXX_pXX"'
        self._model_version_name_ = "_".join(self._name_split_[:2])

        ### set name_appendix
        if isinstance(name_appendix, type(None)):
            self.name_appendix = ""
            name = name + self.name_appendix
        elif isinstance(name_appendix, type("txt")):
            name_appendix = ":" + name_appendix
            self.name_appendix = name_appendix
            name = name + self.name_appendix
        else:
            raise TypeError("name_appendix has to be string")
            quit()

        ### init default compile_folder_name
        if compile_folder_name == None:
            compile_folder_name = "annarchy_" + self._model_version_name_

        ### set description
        description = (
            "The basal ganglia model based on the model from Goenner et al. (2021)"
        )

        ### init random number generator
        self.rng = np.random.default_rng(seed)

        ### get model parameters before init, ignore name_appendix
        self.params = self.__get_params__(name.split(":")[0])

        ### init
        super().__init__(
            model_creation_function=self.__model_creation_function__,
            name=name,
            description=description,
            do_create=do_create,
            do_compile=do_compile,
            compile_folder_name=compile_folder_name,
        )

    def __add_name_appendix__(self):
        """
        all model compartments names end with the name_appendix --> for setting the parameters you need to change the names of the model compartments in the paramter keys
        """
        ### update the attribute_df of the model object (it still contains the original names of the model creation)
        self.attribute_df["compartment_name"] = (
            self.attribute_df["compartment_name"] + self.name_appendix
        )
        ### rename populations and projections
        populations_new = []
        for pop_name in self.populations:
            populations_new.append(pop_name + self.name_appendix)
            get_population(pop_name).name = pop_name + self.name_appendix
        self.populations = populations_new
        projections_new = []
        for proj_name in self.projections:
            projections_new.append(proj_name + self.name_appendix)
            get_projection(proj_name).name = proj_name + self.name_appendix
        self.projections = projections_new
        ### rename parameters
        params_new = {}
        for key, param_val in self.params.items():
            param_object = key.split(".")[0]
            param_name = key.split(".")[1]

            if param_object == "general":
                params_new[key] = param_val
                continue

            param_object = param_object + self.name_appendix
            key_new = param_object + "." + param_name
            params_new[key_new] = param_val
        self.params = params_new

    def __model_creation_function__(self):
        model_creation_function = eval(
            "importlib.import_module('CompNeuroPy.models.BGM_22.model_creation_functions')."
            + self._model_version_name_
        )
        model_creation_function(self)

    def create(self, do_compile=True, compile_folder_name=None):
        super().create(do_compile=False, compile_folder_name=compile_folder_name)
        ### after creating the model, self.populations and self.projections are available --> now set the parameters
        ### do not compile during create but after setting parameters --> parameter values are included in compilation state
        ### before setting parameters add name_appendix to model compartments and parameters
        self.__add_name_appendix__()
        self.__set_params__()
        self.__set_noise_values__()
        self.__set_connections__()

        if do_compile:
            self.compile(compile_folder_name)

    def compile(self, compile_folder_name=None):
        ### just run the standard compile
        super().compile(compile_folder_name)

        ### and update the weights in the attribute_df
        ### for each projection there has to be a connectivity parameter (otherwise error occurs)
        ### and for each projections the weights are modified
        ### thus just update weights of all projections
        ### loop over all projections
        for proj_name in self.projections:
            ### update the model attribute_df
            values = get_projection(proj_name).w
            self.__update_attribute_df__(
                compartment=proj_name, parameter_name="w", parameter_value=values
            )

    def __set_params__(self):
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
                    self.set_param(
                        compartment=param_object,
                        parameter_name=param_name,
                        parameter_value=param_val,
                    )

    def __set_noise_values__(self):
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
                            parameter_value=self.rng.normal(
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

    def __set_connections__(self):
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

    def __get_params__(self, name):
        """
        read all parameters for specified model name

        name : str
            specifies which column in the csv file is used
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

    def needed_imports(self):
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
