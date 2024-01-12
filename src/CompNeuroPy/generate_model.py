from typing import Callable
from CompNeuroPy import model_functions as mf
from CompNeuroPy import analysis_functions as af
from ANNarchy import get_population, get_projection
import numpy as np
import pandas as pd
from typingchecker import check_types


class CompNeuroModel:
    """
    Class for creating and compiling a model.

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
    """

    _initialized_models = {}
    _compiled_models = {}
    _compiled_models_updated = False

    @check_types()
    def __init__(
        self,
        model_creation_function: Callable,
        model_kwargs: dict | None = None,
        name: str = "model",
        description: str = "",
        do_create: bool = True,
        do_compile: bool = True,
        compile_folder_name: str = "annarchy",
    ):
        """
        Initializes the CompNeuroModel class.

        Args:
            model_creation_function (function):
                Function which creates the model.
            model_kwargs (dict):
                Keyword arguments for model_creation_function. Default: None.
            name (str):
                Name of the model. Default: "model".
            description (str):
                Description of the model. Default: "".
            do_create (bool):
                If True the model is created directly. Default: True.
            do_compile (bool):
                If True the model is compiled directly. Default: True.
            compile_folder_name (str):
                Name of the folder in which the model is compiled. Default: "annarchy".
        """
        self.name = name
        if name == "model":
            self.name = name + str(self._nr_models())
        self.description = description
        self.model_creation_function = model_creation_function
        self.compile_folder_name = compile_folder_name
        self.model_kwargs = model_kwargs
        self.populations = []
        self.projections = []
        self.created = False
        self.compiled = False
        self._attribute_df = None
        self._attribute_df_compiled = False
        if do_create:
            self.create(do_compile=do_compile, compile_folder_name=compile_folder_name)

    @property
    def compiled(self):
        """
        True if the model is compiled.
        """
        ### check if ANNarchy was compiled and _compiled_models is not updated yet
        if mf.annarchy_compiled() and not self._compiled_models_updated:
            self._update_compiled_models()
        return self._compiled_models[self.name]

    @compiled.setter
    def compiled(self, value):
        """
        Setter for compiled property.
        """
        self._compiled_models[self.name] = value

    @property
    def created(self):
        """
        True if the model is created.
        """
        return self._initialized_models[self.name]

    @created.setter
    def created(self, value):
        """
        Setter for created property.
        """
        self._initialized_models[self.name] = value

    @property
    def attribute_df(self):
        """
        Dataframe containing all attributes of the model compartments.
        """
        ### check if ANNarchy was compiled and _attribute_df is not updated yet
        if mf.annarchy_compiled() and not self._attribute_df_compiled:
            self._update_attribute_df_weights()
        return self._attribute_df

    def _update_compiled_models(self):
        """
        Updates _compiled_models to True for all models.
        """
        ### update _compiled_models
        for key in self._compiled_models.keys():
            self._compiled_models[key] = True
        self._compiled_models_updated = True

    def _update_attribute_df_weights(self):
        """
        Updates _attribute_df for the weights of all projections.
        """
        for proj_name in self.projections:
            values = get_projection(proj_name).w
            self._update_attribute_df(
                compartment=proj_name, parameter_name="w", parameter_value=values
            )
        self._attribute_df_compiled = True

    def compile(self, compile_folder_name=None):
        """
        Compiles a created model.

        Args:
            compile_folder_name (str, optional):
                Name of the folder in which the model is compiled. Default: value from
                initialization.
        """
        ### check if this model is created
        if self.created:
            if compile_folder_name == None:
                compile_folder_name = self.compile_folder_name

            ### check if other models were initialized but not created --> warn that they are not compiled
            not_created_model_list = self._check_if_models_created()
            if len(not_created_model_list) > 0:
                print(
                    "\nWARNING during compile of model "
                    + self.name
                    + ": There are initialized models which are not created, thus not compiled! models:\n"
                    + "\n".join(not_created_model_list)
                    + "\n"
                )
            mf.compile_in_folder(compile_folder_name)
            self.compiled = True

            ### update attribute_df to compiled state, since weights are only available
            ### after compilation
            self._update_attribute_df_weights()
        else:
            print("\n")
            assert False, (
                "ERROR during compile of model "
                + self.name
                + ": Only compile the model after it has been created!"
            )

    def create(self, do_compile=True, compile_folder_name=None):
        """
        Creates a model and optionally compiles it directly.

        Args:
            do_compile (bool, optional):
                If True the model is compiled directly. Default: True.
            compile_folder_name (str, optional):
                Name of the folder in which the model is compiled. Default: value from
                initialization.
        """
        if self.created:
            print("model", self.name, "already created!")
        else:
            initial_existing_model = mf.get_full_model()
            ### create model populations and projections
            if self.model_kwargs != None:
                self.model_creation_function(**self.model_kwargs)
            else:
                self.model_creation_function()
            self.created = True

            ### check which populations and projections have been added
            post_existing_model = mf.get_full_model()
            ### save only added not all projections/populations
            for initial_pop in initial_existing_model["populations"]:
                post_existing_model["populations"].remove(initial_pop)
            for initial_proj in initial_existing_model["projections"]:
                post_existing_model["projections"].remove(initial_proj)
            self.populations = post_existing_model["populations"]
            self.projections = post_existing_model["projections"]

            ### check if names of populations and projections are unique
            self._check_double_compartments()

            ### create parameter dictionary
            self._attribute_df = self._get_attribute_df()

            if do_compile:
                self.compile(compile_folder_name)

    def _check_if_models_created(self):
        """
        Checks which CompNeuroPy models are created

        Returns:
            not_created_model_list (list):
                list of names of all initialized CompNeuroPy models which are not
                created yet
        """
        not_created_model_list = []
        for key in self._initialized_models.keys():
            if self._initialized_models[key] == False:
                not_created_model_list.append(key)

        return not_created_model_list

    def _nr_models(self):
        """
        Returns:
            nr_models (int):
                The current number of initialized (not considering "created")
                CompNeuroPy models
        """
        return len(list(self._initialized_models.keys()))

    def set_param(self, compartment, parameter_name, parameter_value):
        """
        Sets the specified parameter of the specified compartment.

        Args:
            compartment (str):
                name of model compartment
            parameter_name (str):
                name of parameter of the compartment
            parameter_value (number or array-like with shape of compartment geometry):
                the value or values of the parameter

        Raises:
            AssertionError: if model is not created
            AssertionError: if compartment is neither a population nor a projection of
                the model
        """
        ### catch if model is not created
        assert (
            self.created == True
        ), f"ERROR set_param: model {self.name} has to be created before setting parameters!"

        ### check if compartment is in populations or projections
        comp_in_pop = compartment in self.populations
        comp_in_proj = compartment in self.projections

        if comp_in_pop:
            comp_obj = get_population(compartment)
        elif comp_in_proj:
            comp_obj = get_projection(compartment)
        else:
            assert (
                comp_in_pop or comp_in_proj
            ), f"ERROR set_param: setting parameter {parameter_name} of compartment {compartment}. The compartment is neither a population nor a projection of the model {self.name}!"

        ### set the parameter value
        setattr(comp_obj, parameter_name, parameter_value)

        ### update the model attribute_df
        self._update_attribute_df(compartment, parameter_name, parameter_value)

    def _update_attribute_df(self, compartment, parameter_name, parameter_value):
        """
        updates the attribute df for a specific paramter

        Args:
            compartment (str):
                name of model compartment
            parameter_name (str):
                name of parameter of the compartment
            parameter_value (number or array-like with shape of compartment geometry):
                the value or values of the parameter
        """
        paramter_mask = (
            (self._attribute_df["compartment_name"] == compartment).astype(int)
            * (self._attribute_df["attribute_name"] == parameter_name).astype(int)
        ).astype(bool)
        parameter_idx = np.arange(paramter_mask.size).astype(int)[paramter_mask][0]
        min_val = af.get_minimum(parameter_value)
        max_val = af.get_maximum(parameter_value)
        if min_val != max_val:
            self._attribute_df.at[parameter_idx, "value"] = f"[{min_val}, {max_val}]"
        else:
            self._attribute_df.at[parameter_idx, "value"] = str(min_val)
        self._attribute_df.at[parameter_idx, "definition"] = "modified"

    def _check_double_compartments(self):
        """
        Goes over all compartments of the model and checks if compartment is only a
        population or a projection and not both.

        Raises:
            AssertionError: if model is not created
            AssertionError: if compartment is both a population and a projection
        """
        ### cach if model is not created, only if created populations and projections are available
        assert (
            self.created == True
        ), f"ERROR model {self.name}: model has to be created before checking for double compartments!"
        ### only have to go over populations and check if they are also projections (go over projections not neccessary)
        pop_in_projections_list = []
        pop_in_projections = False
        for pop_name in self.populations:
            if pop_name in self.projections:
                pop_in_projections_list.append(pop_name)
                pop_in_projections = True

        assert (
            pop_in_projections == False
        ), f"ERROR model {self.name}: One or multiple compartments are both population and projection ({pop_in_projections_list}). Rename them!"

    def _get_attribute_df(self):
        """
        Creates a dataframe containing the attributes of all model compartments.

        Returns:
            attribute_df (pandas dataframe):
                dataframe containing all attributes of the model compartments

        Raises:
            AssertionError: if model is not created
        """
        ### cach if model is not created, only if created populations and projections are available
        assert (
            self.created == True
        ), f"ERROR model {self.name}: model has to be created before creating paramteer dictionary!"

        ### create empty paramteter dict
        attribute_dict = {
            "compartment_type": [],
            "compartment_name": [],
            "attribute_name": [],
            "value": [],
            "definition": [],
        }

        ### fill paramter dict with population attributes
        for pop in self.populations:
            for attribute in vars(get_population(pop))["attributes"]:
                ### store min and max of attribute
                ### create numpy array with getattr to use numpy min max function
                values = np.array(
                    [getattr(get_population(pop), attribute)]
                    + [getattr(get_population(pop), attribute)]
                )
                attribute_dict["compartment_type"].append("population")
                attribute_dict["compartment_name"].append(pop)
                attribute_dict["attribute_name"].append(attribute)
                if values.min() != values.max():
                    attribute_dict["value"].append(f"[{values.min()}, {values.max()}]")
                else:
                    attribute_dict["value"].append(str(values.min()))
                attribute_dict["definition"].append("init")

        ### fill paramter dict with projection attributes
        for proj in self.projections:
            for attribute in vars(get_projection(proj))["attributes"]:
                ### store min and max of attribute
                ### create numpy array with getattr to use numpy min max function
                values = np.array(
                    [getattr(get_projection(proj), attribute)]
                    + [getattr(get_projection(proj), attribute)]
                )
                attribute_dict["compartment_type"].append("projection")
                attribute_dict["compartment_name"].append(proj)
                attribute_dict["attribute_name"].append(attribute)
                if values.min() != values.max():
                    attribute_dict["value"].append(f"[{values.min()}, {values.max()}]")
                else:
                    attribute_dict["value"].append(values.min())
                attribute_dict["definition"].append("init")

        ### return dataframe
        return pd.DataFrame(attribute_dict)


### old name for compatibility, TODO: remove
generate_model = CompNeuroModel
