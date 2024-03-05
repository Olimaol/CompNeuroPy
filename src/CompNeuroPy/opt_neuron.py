from ANNarchy import setup, Population, get_population, Neuron
import numpy as np
import traceback
from CompNeuroPy import system_functions as sf
from CompNeuroPy import extra_functions as ef
from CompNeuroPy import model_functions as mf
from CompNeuroPy import analysis_functions as af
from CompNeuroPy.monitors import CompNeuroMonitors
from CompNeuroPy.generate_model import CompNeuroModel
from CompNeuroPy.experiment import CompNeuroExp
import matplotlib.pyplot as plt
import sys
from typing import Callable, Any, Type
from typingchecker import check_types
from copy import deepcopy
import pandas as pd
from multiprocessing import Process
import multiprocessing
import json
from time import time

try:
    # hyperopt
    from hyperopt import fmin, tpe, hp, STATUS_OK

    # torch
    import torch

    # sbi
    from sbi import analysis as analysis
    from sbi import utils as utils
    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

    # pybads
    from pybads.bads import BADS

except:
    print(
        "OptNeuron: Error: You need to install hyperopt, torch, pybads, and sbi to use OptNeuron (e.g. use pip install hyperopt torch pybads sbi))"
    )
    sys.exit()


class OptNeuron:
    """
    This class is used to optimize neuron models with ANNarchy.
    """

    opt_created = []

    @check_types(warnings=False)
    def __init__(
        self,
        experiment: Type[CompNeuroExp],
        get_loss_function: Callable[[Any, Any], float | list[float]],
        variables_bounds: dict[str, float | str | list[float | str]],
        neuron_model: Neuron,
        results_soll: Any | None = None,
        target_neuron_model: Neuron | None = None,
        time_step: float = 1.0,
        recording_period: float | None = None,
        compile_folder_name: str = "annarchy_OptNeuron",
        num_rep_loss: int = 1,
        method: str = "deap",
        prior=None,
        fv_space: list = None,
        record: list[str] = [],
        cma_params_dict: dict = {},
        bads_params_dict: dict = {},
        source_solutions: list[tuple[np.ndarray, float]] = [],
        variables_bounds_guess: None | dict[str, list[float]] = None,
        verbose=False,
    ):
        """
        This prepares the optimization. To run the optimization call the run function.

        Args:
            experiment (CompNeuroExp class):
                CompNeuroExp class containing a 'run' function which defines the
                simulations and recordings
            get_loss_function (function):
                function which takes results_ist and results_soll as arguments and
                calculates/returns the loss
            variables_bounds (dict):
                Dictionary with parameter names (keys) and their bounds (values). If
                single values are given as values, the parameter is constant, i.e., not
                optimized. If a list is given as value, the parameter is optimized and
                the list contains the lower and upper bound of the parameter (order is
                not important). If strings instead of numbers are given, the string is
                interpreted as an mathematical expression which is evaluated with the
                other parameter values (i.e. {"x":[0,1],"dxy":[-1,1],"y":"x+dxy","z":5}).
            neuron_model (ANNarchy Neuron):
                The neuron model whose parameters should be optimized.
            results_soll (Any, optional):
                Some variable which contains the target data and can be used by the
                get_loss_function (second argument of get_loss_function)
                !!! warning
                    Either provide results_soll or a target_neuron_model not both!
                Default: None.
            target_neuron_model (ANNarchy Neuron, optional):
                The neuron model which produces the target data by running the
                experiment.
                !!! warning
                    Either provide results_soll or a target_neuron_model not both!
                Default: None.
            time_step (float, optional):
                The time step for the simulation in ms. Default: 1.
            recording_period (float, optional):
                The recording period for the simulation in ms. Default: None, i.e., the
                time_step is used.
            compile_folder_name (string, optional):
                The name of the annarchy compilation folder within annarchy_folders/.
                Default: 'annarchy_OptNeuron'.
            num_rep_loss (int, optional):
                Only interesting for noisy simulations/models. How often should the
                simulaiton be run to calculate the loss (the defined number of losses
                is obtained and averaged). Default: 1.
            method (str, optional):
                Either 'deap', 'sbi', or 'hyperopt'. Defines the tool which is used for
                optimization. Default: 'deap'.
            prior (distribution, optional):
                The prior distribution used by sbi. Default: None, i.e., uniform
                distributions between the variable bounds are assumed.
            fv_space (list, optional):
                The search space for hyperopt. Default: None, i.e., uniform
                distributions between the variable bounds are assumed.
            record (list, optional):
                List of strings which define what variables of the tuned neuron should
                be recorded. Default: [].
            cma_params_dict (dict, optional):
                Dictionary with parameters for the deap.cma.Strategy. Default: {}.
                See [here](https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.Strategy) for more information.
            bads_params_dict (dict, optional):
                Dictionary with parameters for the bads optimization. Default: {}.
                See [here](https://acerbilab.github.io/pybads/api/options/bads_options.html) for more information.
            source_solutions (list, optional):
                List of tuples with the source solutions. Each tuple contains a numpy
                array with the parameter values and the loss. Used for initialization of
                cma optimization with deap. Default: [].
            variables_bounds_guess (dict, optional):
                Dictionary with parameter names (keys) and their bounds (values) as
                list. These bounds define the region there the minimum is expected. Used
                for the BADS optimization. Default: None.
            verbose (bool, optional):
                If True, print additional information. Default: False.
        """

        if len(self.opt_created) > 0:
            print(
                "OptNeuron: Error: Already another OptNeuron created. Only create one per python session!"
            )
            quit()
        else:
            print(
                "OptNeuron: Initialize OptNeuron... do not create anything with ANNarchy before!"
            )

            ### set object variables
            self.verbose = verbose
            self.verbose_run = False
            self.opt_created.append(1)
            self.record = record
            self.results_soll = results_soll
            self.variables_bounds = variables_bounds
            self.fitting_variables_name_list = self._get_fitting_variables_name_list()
            self.method = method
            if method == "hyperopt":
                if fv_space is None:
                    self.fv_space = self._get_hyperopt_space()
                else:
                    self.fv_space = fv_space
            self.const_params = self._get_const_params()
            self.num_rep_loss = num_rep_loss
            self.neuron_model = neuron_model
            if method == "sbi":
                self.prior = self._get_prior(prior)
            self.target_neuron = target_neuron_model
            self.compile_folder_name = compile_folder_name
            self._get_loss = get_loss_function
            self.cma_params_dict = cma_params_dict
            self.source_solutions = source_solutions
            self.variables_bounds_guess = variables_bounds_guess
            self.bads_params_dict = bads_params_dict
            self.loss_history = []
            self.start_time = time()
            self.recording_period = recording_period

            ### if using deap pop size is the number of individuals for the optimization
            if method == "deap":
                self._deap_cma = self._prepare_deap_cma()
                self.popsize = self._deap_cma.deap_dict["strategy"].lambda_
            else:
                self.popsize = 1
            if self.verbose:
                print("OptNeuron: popsize:", self.popsize)

            ### check target_neuron/results_soll
            self._check_target()
            ### check neuron models
            self._check_neuron_models()

            ### setup ANNarchy
            setup(dt=time_step)

            ### create and compile model
            ### if neuron models and target neuron model --> create both models then
            ### test, then clear and create only model for neuron model
            model, target_model, monitors = self._generate_models(self.popsize)

            self.pop = model.populations[0]
            if target_model is not None:
                self.pop_target = target_model.populations[0]
            else:
                self.pop_target = None
            ### create experiment with current monitors
            self.experiment = experiment(monitors=monitors)

            ### check variables of model
            self._test_variables()

            ### check neuron models, experiment, get_loss
            ### if results_soll is None -_> generate results_soll
            self._check_get_loss_function()

            ### after checking neuron models, experiment, get_loss
            ### clear ANNarchy and create/compile again only
            ### standard model, thus recreate also monitors and experiment
            mf.cnp_clear()
            model, _, monitors = self._generate_models(self.popsize)
            self.monitors = monitors
            self.experiment = experiment(monitors=monitors)

    def _get_lower_upper_p0(self):
        """
        Returns the lower and upper bounds and the initial values for the cma
        optimization with deap.

        Returns:
            lower (np.array):
                The lower bounds for the optimization.
            upper (np.array):
                The upper bounds for the optimization.
            p0 (np.array):
                The initial values for the optimization.
        """

        lower = np.array(
            [
                min(self.variables_bounds[name])
                for name in self.fitting_variables_name_list
            ]
        )
        upper = np.array(
            [
                max(self.variables_bounds[name])
                for name in self.fitting_variables_name_list
            ]
        )
        p0 = np.array(
            [
                np.random.uniform(
                    min(self.variables_bounds[key]),
                    max(self.variables_bounds[key]),
                )
                for key in self.fitting_variables_name_list
            ]
        )
        return lower, upper, p0

    def _get_lower_upper_x0(self):
        """
        Returns the lower and upper bounds and the initial values for the optimization
        with bads.

        Returns:
            lower (np.array):
                The lower bounds for the optimization.
            upper (np.array):
                The upper bounds for the optimization.
            x0 (np.array):
                The initial values for the optimization.
            lower_guess (np.array):
                The lower bounds for the optimization where the minimum is expected.
            upper_guess (np.array):
                The upper bounds for the optimization where the minimum is expected.
        """
        lower, upper, x0 = self._get_lower_upper_p0()

        if not isinstance(self.variables_bounds_guess, type(None)):
            lower_guess = np.array(
                [
                    min(self.variables_bounds_guess[name])
                    for name in self.fitting_variables_name_list
                ]
            )
            upper_guess = np.array(
                [
                    max(self.variables_bounds_guess[name])
                    for name in self.fitting_variables_name_list
                ]
            )
        else:
            lower_guess = deepcopy(lower)
            upper_guess = deepcopy(upper)

        return lower, upper, x0, lower_guess, upper_guess

    def _prepare_deap_cma(self):
        """
        Initializes the DeapCma class.

        Returns:
            deap_cma (DeapCma):
                The initialized DeapCma object.
        """

        LOWER, UPPER, p0 = self._get_lower_upper_p0()

        deap_cma = ef.DeapCma(
            max_evals=0,
            lower=LOWER,
            upper=UPPER,
            evaluate_function=self._deap_simulation_wrapper,
            p0=p0,
            param_names=self.fitting_variables_name_list,
            learn_rate_factor=1,
            damping_factor=1,
            verbose=False,
            plot_file=None,
            cma_params_dict=self.cma_params_dict,
            source_solutions=self.source_solutions,
        )
        return deap_cma

    class _NullContextManager:
        def __enter__(self):
            # This method is called when entering the context
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            # This method is called when exiting the context
            pass

    def _generate_models(self, popsize=1):
        """
        Generates the tuned model and the target_model (only if results_soll is None).

        Args:
            popsize (int, optional):
                The number of neurons in the population(s). Default: 1.

        Returns:
            model (CompNeuroModel):
                The model which is used for the optimization.

            target_model (CompNeuroModel):
                The model which is used to generate the target data. If results_soll is
                provided, target_model is None.

            monitors (CompNeuroMonitors):
                The monitors which are used to record the data. If no variables are
                recorded, monitors is None.
        """
        with self._NullContextManager() if self.verbose else ef.suppress_stdout():
            model = None
            target_model = None
            monitors = None
            if self.results_soll is None:
                if self.verbose:
                    print(
                        "OptNeuron: Create two models (optimized and target for obtaining results_soll)"
                    )
                    print("optimized neuron model:", self.neuron_model)
                    print("target neuron model:", self.target_neuron)
                ### create two models
                model = CompNeuroModel(
                    model_creation_function=self._raw_neuron,
                    model_kwargs={
                        "neuron": self.neuron_model,
                        "name": "model_neuron",
                        "size": popsize,
                    },
                    name="standard_model",
                    do_create=True,
                    do_compile=False,
                    compile_folder_name=self.compile_folder_name,
                )

                target_model = CompNeuroModel(
                    model_creation_function=self._raw_neuron,
                    model_kwargs={
                        "neuron": self.target_neuron,
                        "name": "target_model_neuron",
                        "size": 1,
                    },
                    name="target_model",
                    do_create=True,
                    do_compile=True,
                    compile_folder_name=self.compile_folder_name,
                )

                ### create monitors
                if len(self.record) > 0:
                    recording_period_str = (
                        f";{self.recording_period}"
                        if self.recording_period is not None
                        and ("spike" not in self.record or len(self.record) > 1)
                        else ""
                    )
                    monitors = CompNeuroMonitors(
                        {
                            f"{pop_name}{recording_period_str}": self.record
                            for pop_name in [
                                model.populations[0],
                                target_model.populations[0],
                            ]
                        }
                    )

            else:
                if self.verbose:
                    print(
                        "OptNeuron: Create one model (optimized, results_soll is available)"
                    )
                    print("optimized neuron model:", self.neuron_model)
                ### create one model
                model = CompNeuroModel(
                    model_creation_function=self._raw_neuron,
                    model_kwargs={
                        "neuron": self.neuron_model,
                        "name": "model_neuron",
                        "size": popsize,
                    },
                    name="single_model",
                    do_create=True,
                    do_compile=True,
                    compile_folder_name=self.compile_folder_name,
                )
                ### create monitors
                if len(self.record) > 0:
                    recording_period_str = (
                        f";{self.recording_period}"
                        if self.recording_period is not None
                        and ("spike" not in self.record or len(self.record) > 1)
                        else ""
                    )
                    monitors = CompNeuroMonitors(
                        {f"{model.populations[0]}{recording_period_str}": self.record}
                    )

        return model, target_model, monitors

    def _check_neuron_models(self):
        """
        Checks if the neuron models are ANNarchy neuron models.
        """
        if not (isinstance(self.neuron_model, type(Neuron()))) or (
            self.target_neuron is not None
            and not (isinstance(self.target_neuron, type(Neuron())))
        ):
            print(
                "OptNeuron: Error: neuron_model and/or target_neuron_model have to be ANNarchy neuron models"
            )
            quit()

    def _check_target(self):
        """
        Check if either results_soll or target_neuron are provided and not both.
        """
        if self.target_neuron is None and self.results_soll is None:
            print(
                "OptNeuron: Error: Either provide results_soll or target_neuron_model"
            )
            quit()
        elif self.target_neuron is not None and self.results_soll is not None:
            print(
                "OptNeuron: Error: Either provide results_soll or target_neuron_model, not both"
            )
            quit()

    def _get_prior(self, prior):
        """
        Get the prior distribution used by sbi. If no prior is given, uniform
        distributions between the variable bounds are assumed. If a prior is given,
        this prior is used.

        Args:
            prior (distribution, optional):
                The prior distribution used by sbi. Default: None, i.e., uniform
                distributions between the variable bounds are assumed.

        Returns:
            prior (distribution):
                The prior distribution used by sbi.
        """
        if prior is None:
            prior_min = []
            prior_max = []
            for _, param_bounds in self.variables_bounds.items():
                if isinstance(param_bounds, list):
                    prior_min.append(param_bounds[0])
                    prior_max.append(param_bounds[1])

            return utils.BoxUniform(
                low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
            )
        else:
            return prior

    def _get_fitting_variables_name_list(self):
        """
        Returns a list with the names of the fitting variables.

        Returns:
            fitting_variables_name_list (list):
                list with names of fitting variables
        """
        name_list = []
        for param_name, param_bounds in self.variables_bounds.items():
            if isinstance(param_bounds, list):
                name_list.append(param_name)

        if self.verbose:
            print("OptNeuron: Fitting variables:", name_list)
        return name_list

    def _get_hyperopt_space(self):
        """
        Generates the hyperopt variable space from the fitting variable bounds. The
        variable space is a uniform distribution between the bounds.

        Returns:
            fitting_variables_space (list):
                list with hyperopt variables
        """
        fitting_variables_space = []
        for param_name, param_bounds in self.variables_bounds.items():
            if isinstance(param_bounds, list):
                fitting_variables_space.append(
                    hp.uniform(param_name, min(param_bounds), max(param_bounds))
                )
        return fitting_variables_space

    def _get_const_params(self):
        """
        Returns:
            const_params (dict):
                Dictionary with constant variables. The keys are the parameter names
                and the values are the parameter values.
        """
        const_params = {}
        for param_name, param_bounds in self.variables_bounds.items():
            if not (isinstance(param_bounds, list)):
                const_params[param_name] = param_bounds

        if self.verbose:
            print("OptNeuron: Constant parameters:", const_params)
        return const_params

    def _check_get_loss_function(self):
        """
        Checks if the get_loss_function is compatible to the experiment and the neuron
        model(s). To test, the experiment is run once with the tuned neuron model
        (generating results_ist) and once with the target neuron model (if provided,
        generating results_soll). Then, the get_loss_function is called with the
        results_ist and results_soll.
        """
        print("checking neuron_models, experiment, get_loss...", end="")

        fitparams = []
        for bounds in self.variables_bounds.values():
            if isinstance(bounds, list):
                fitparams.append(bounds[0])
        if self.verbose:
            print(
                f"fitparams selected for checking: {fitparams} ({self.fitting_variables_name_list})"
            )

        if self.results_soll is not None:
            ### only generate results_ist with standard neuron model
            results_ist = self._run_simulator_with_results(fitparams)["results"]
        else:
            ### run simulator with both populations (standard neuron model and target
            ### neuron model) and generatate results_ist and results_soll
            results_ist = self._run_simulator_with_results(fitparams)["results"]
            self.results_soll = self._run_simulator_with_results(
                fitparams, pop=self.pop_target
            )["results"]

        try:
            self._wrapper_get_loss(results_ist, self.results_soll)
        except:
            print(
                "\nThe get_loss_function, experiment and neuron model(s) are not compatible:\n"
            )
            traceback.print_exc()
            quit()
        print("Done\n")

    def _wrapper_get_loss(self, results_ist, results_soll):
        """
        Makes it possible to use the get_loss_function with multiple neurons. The
        get_loss_function should always calculate the loss for neuron rank 0!

        Args:
            results_ist (object):
                the results object returned by the run function of experiment (see above)
                it can contain recordings of multiple neurons
            results_soll (any):
                the target data directly provided to OptNeuron during initialization
                it always contains only the recordings of a single neuron

        Returns:
            all_loss_list (list):
                list of lists containing the 'all_loss_list' for each neuron
        """
        ### loop over neurons and calculate all_loss_list for each neuron
        all_loss_list = []
        for neuron_idx in range(self.popsize):
            results_ist_neuron = self._get_results_of_single_neuron(
                results_ist, neuron_idx
            )
            all_loss_list.append(self._get_loss(results_ist_neuron, results_soll))

        return all_loss_list

    def _get_results_of_single_neuron(self, results, neuron_idx):
        """
        Returns a results object which contains only the recordings of the given neuron
        index. The defined neuron will be neuron rank 0 in the returned results object.

        Args:
            results (object):
                the results object returned by the run function of experiment (see above)
                it can contain recordings of multiple neurons
            neuron_idx (int):
                index of the neuron whose recordings should be returned

        Returns:
            results_neuron (object):
                the results object as returned by the run function of experiment for a
                single neuron
        """
        ### if only one neuron, simply return results
        if self.popsize == 1:
            return results

        ### if multiple neurons, return results for single neuron, do not change
        ### original results!
        results_neuron = deepcopy(results)

        ### loop over chunks and recordings and select only the recordings of the
        ### defined neuron
        for chunk in range(len(results_neuron.recordings)):
            for rec_key in results_neuron.recordings[chunk].keys():
                ### adjust spike dictionary
                if "spike" in rec_key and not ("target" in rec_key):
                    results_neuron.recordings[chunk][rec_key] = {
                        0: results_neuron.recordings[chunk][rec_key][neuron_idx]
                    }
                ### adjust all recorded arrays
                elif not (
                    "period" in rec_key
                    or "parameter_dict" in rec_key
                    or "dt" in rec_key
                    or "target" in rec_key
                ):
                    results_neuron.recordings[chunk][rec_key] = (
                        results_neuron.recordings[chunk][rec_key][
                            :, neuron_idx
                        ].reshape(-1, 1)
                    )
                ### adjust parameter_dict
                elif "parameter_dict" in rec_key and not ("target" in rec_key):
                    results_neuron.recordings[chunk][rec_key] = {
                        parameter_dict_key: np.array(
                            [
                                results_neuron.recordings[chunk][rec_key][
                                    parameter_dict_key
                                ][neuron_idx]
                            ]
                        )
                        for parameter_dict_key in results_neuron.recordings[chunk][
                            rec_key
                        ].keys()
                    }

        return results_neuron

    def _raw_neuron(self, neuron, name, size):
        """
        Generates a population with one neuron of the given neuron model.

        Args:
            neuron (ANNarchy Neuron):
                The neuron model.
            name (str):
                The name of the population.
            size (int):
                The number of neurons in the population.
        """
        Population(size, neuron=neuron, name=name)

    def _test_variables(self):
        """
        Check if the tuned neuron model contains all parameters which are defined in
        variables_bounds or even more.
        """
        ### collect all names
        all_vars_names = np.concatenate(
            [
                np.array(list(self.const_params.keys())),
                np.array(self.fitting_variables_name_list),
            ]
        ).tolist()
        ### check if pop has these parameters
        pop_parameter_names = get_population(self.pop).attributes.copy()
        for name in pop_parameter_names.copy():
            if name in all_vars_names:
                all_vars_names.remove(name)
                pop_parameter_names.remove(name)
        if len(pop_parameter_names) > 0:
            print(
                "OptNeuron: WARNING: attributes",
                pop_parameter_names,
                "are not used/initialized.",
            )
        if len(all_vars_names) > 0:
            print(
                "OptNeuron: WARNING: The neuron_model does not contain parameters",
                all_vars_names,
                "!",
            )

    def _run_simulator(self, fitparams):
        """
        Runs the function simulator with the multiprocessing manager (if function is
        called multiple times this saves memory, otherwise same as calling simulator
        directly).

        Args:
            fitparams (list):
                list with values for fitting parameters or list of lists with values
                for fitting parameters (first dimension is the number of parameters,
                second dimension is the number of neurons)

        Returns:
            return_dict (dict):
                dictionary needed for optimization with hyperopt, containing the loss,
                the loss variance (in case of noisy models with multiple runs per loss
                calculation), and the status (STATUS_OK for hyperopt).
        """

        ### initialize manager and generate m_list = dictionary to save data
        manager = multiprocessing.Manager()
        m_list = manager.dict()

        ### in case of noisy models, here optionally run multiple simulations, to mean the loss
        loss_list_over_runs = []

        return_results = False
        for _ in range(self.num_rep_loss):
            ### initialize for each run a new rng (--> not always have same noise in case of noisy models/simulations)
            rng = np.random.default_rng()
            ### run simulator with multiprocessign manager
            proc = Process(
                target=self._simulator, args=(fitparams, rng, m_list, return_results)
            )
            proc.start()
            proc.join()
            ### get simulation results/loss (list of losses for each neuron)
            loss_list_over_runs.append(m_list[0])

        ### create loss array, first dimension is the number of runs, second dimension
        ### is the number of neurons
        loss_arr = np.array(loss_list_over_runs)

        ### calculate mean and std of loss over runs
        if self.num_rep_loss > 1:
            ### multiple runs, mean over runs
            ### -> resulting in 1D arrays for neurons
            loss_ret_arr = np.mean(loss_arr, 0)
            std_ret_arr = np.std(loss_arr, 0)
        else:
            ### just take the first entry (the only one)
            ### -> resulting in 1D arrays for neurons
            loss_ret_arr = loss_arr[0]
            std_ret_arr = np.array([None] * self.popsize)

        ### if only one neuron, return loss and std as single values
        if self.popsize == 1:
            loss = loss_ret_arr[0]
            std = std_ret_arr[0]
        else:
            loss = loss_ret_arr
            std = std_ret_arr

        ### append best loss and time since start to loss_history
        self.loss_history.append([af.get_minimum(loss), time() - self.start_time])

        ### return loss and other things for optimization, if multiple neurons
        ### --> loss and std are arrays with loss/std for each neuron
        if self.num_rep_loss > 1:
            return {"status": STATUS_OK, "loss": loss, "loss_variance": std}
        else:
            return {"status": STATUS_OK, "loss": loss}

    def _sbi_simulation_wrapper(self, fitparams):
        """
        This function is called by sbi. It calls the simulator function and
        returns the loss and adjusts the format of the input parameters.

        Args:
            fitparams (tensor):
                either a batch of parameters (tensor with two dimensions) or a single
                parameter set

        Returns:
            loss (tensor):
                loss as tensor for sbi inference
        """
        fitparams = np.asarray(fitparams)
        if len(fitparams.shape) == 2:
            ### batch parameters!
            data = []
            ### TODO the run_simulator_function can now handle multiple parameter sets
            ### and directly can return the loss for each parameter set, but the model
            ### has to have the corrects size, i.e., the number of neurons has to be
            ### the same as the number of parameter sets, maybe adjust sbi to this
            for idx in range(fitparams.shape[0]):
                data.append(self._run_simulator(fitparams[idx])["loss"])
        else:
            ### single parameter set!
            data = [self._run_simulator(fitparams)["loss"]]

        return torch.as_tensor(data)

    def _deap_simulation_wrapper(self, population: list):
        """
        This function is called by deap. It calls the simulator function and
        returns the loss and adjusts the format of the input parameters.

        Args:
            population (list):
                list of lists with values for fitting parameters (first dimension is
                the number of neurons, second dimension is the number of parameters)
                given by deap
        """
        ### transpose population list (now first dimension is the number of parameters,)
        populationT = np.array(population).T.tolist()
        ### get loss list
        loss_list = self._run_simulator(populationT)["loss"]
        ### return loss list as list of tuples (deap needs this format)
        return [(loss_list[neuron_idx],) for neuron_idx in range(len(population))]

    def _bads_simulation_wrapper(self, fitparams: list):
        """
        This function is called by bads. It calls the simulator function and
        returns the loss.
        """
        return self._run_simulator(fitparams)["loss"]

    def _run_simulator_with_results(self, fitparams, pop=None):
        """
        Runs the function simulator with the multiprocessing manager (if function is
        called multiple times this saves memory, otherwise same as calling simulator
        directly) and also returns the results.

        Args:
            fitparams (list):
                list with values for fitting parameters or list of lists with values
                for fitting parameters (first dimension is the number of parameters,
                second dimension is the number of neurons)

            pop (str, optional):
                ANNarchy population name. Default: None, i.e., the tuned population
                is used.

        Returns:
            return_dict (dict):
                dictionary needed for optimization with hyperopt, containing the loss,
                the loss variance (in case of noisy models with multiple runs per loss
                calculation), and the status (STATUS_OK for hyperopt) and the results
                generated by the experiment.
        """
        ### check if pop is given
        if pop is None:
            pop = self.pop
        ### initialize manager and generate m_list = dictionary to save data
        manager = multiprocessing.Manager()
        m_list = manager.dict()

        ### in case of noisy models, here optionally run multiple simulations, to mean
        ### the loss
        loss_list_over_runs = []
        all_loss_list_over_runs = []
        return_results = True
        if self.verbose:
            print(f"OptNeuron: run simulator with results {self.num_rep_loss} times")
        for _ in range(self.num_rep_loss):
            ### initialize for each run a new rng (--> not always have same noise in
            ### case of noisy models/simulations)
            rng = np.random.default_rng()
            ### run simulator with multiprocessign manager
            proc = Process(
                target=self._simulator,
                args=(fitparams, rng, m_list, return_results, pop),
            )
            proc.start()
            proc.join()
            ### get simulation results/loss
            ### list of losses for each neuron
            loss_list_over_runs.append(m_list[0])
            ### results object of experiment
            results_ist = m_list[1]
            ### list of the all_loss_list for each neuron
            all_loss_list_over_runs.append(m_list[2])

        ### create loss array, first dimension is the number of runs, second dimension
        ### is the number of neurons
        loss_arr = np.array(loss_list_over_runs)
        ### create all_loss array, first dimension is the number of runs, second
        ### dimension is the number of neurons, third dimension is the number of
        ### individual losses
        all_loss_arr = np.array(all_loss_list_over_runs)

        ### calculate mean and std of loss over runs
        if self.num_rep_loss > 1:
            ### resulting in 1D arrays for neurons
            loss = np.mean(loss_arr, 0)
            std = np.std(loss_arr)
            ### resulting in 2D array for neurons (1st dim) and individual losses (2nd dim)
            all_loss = np.mean(all_loss_arr, 0)
        else:
            ### just take the first entry (the only one)
            ### resulting in 1D arrays for neurons
            loss = loss_arr[0]
            std = np.array([None] * self.popsize)
            ### resulting in 2D array for neurons (1st dim) and individual losses (2nd dim)
            all_loss = all_loss_arr[0]

        ### if only one neuron, return loss and std as single values and all_loss as
        ### single 1D array (length is the number of individual losses)
        if self.popsize == 1:
            loss = loss[0]
            std = std[0]
            all_loss = all_loss[0]
        else:
            loss = loss
            std = std
            all_loss = all_loss

        ### return loss and other things for optimization and results
        if self.num_rep_loss > 1:
            return {
                "status": STATUS_OK,
                "loss": loss,
                "loss_variance": std,
                "std": std,
                "all_loss": all_loss,
                "results": results_ist,
            }
        else:
            return {
                "status": STATUS_OK,
                "loss": loss,
                "std": std,
                "all_loss": all_loss,
                "results": results_ist,
            }

    def _simulator(
        self, fitparams, rng, m_list=[0, 0, 0], return_results=False, pop=None
    ):
        """
        Runs the experiment with the given parameters and 'returns' the loss and
        optionally the results and all individual losses of the get_loss_function. The
        'returned' values are saved in m_list.

        Args:
            fitparams (list):
                list with values for fitting parameters or list of lists with values
                for fitting parameters (first dimension is the number of parameters,
                second dimension is the number of neurons)

            rng (numpy random generator):
                random generator for the simulation

            m_list (list, optional):
                list with the loss, the results, and the all_loss. Default: [0, 0, 0].

            return_results (bool, optional):
                If True, the results are returned. Default: False.

            pop (str, optional):
                ANNarchy population name. Default: None, i.e., the tuned population
                is used.
        """
        ### TODO use rng here and add it to CompNeuroExp
        ### check if pop is given
        if pop is None:
            pop = self.pop

        ### set parameters which should not be optimized and parameters which should be
        ### optimized before the experiment, they should not be resetted by the
        ### experiment!
        self._set_fitting_parameters(fitparams, pop=pop)
        if self.verbose_run:
            param_dict = {
                param_name: (
                    fitparams[param_name_idx]
                    if isinstance(fitparams[param_name_idx], list)
                    else [fitparams[param_name_idx]]
                )
                for param_name_idx, param_name in enumerate(
                    self.fitting_variables_name_list
                )
            }
            print("OptNeuron: run simulator with parameters:")
            ef.print_df(pd.DataFrame(param_dict))

        ### conduct loaded experiment
        self.experiment.store_model_state(compartment_list=[pop])
        results = self.experiment.run(pop)
        self.experiment.reset()

        if self.results_soll is not None:
            ### compute loss_list, loss for each neuron
            loss_list = []
            ### wrapper_get_loss returns list (neurons) of lists (individual losses)
            all_loss_list = self._wrapper_get_loss(results, self.results_soll)
            ### loop over neurons
            for all_loss in all_loss_list:
                ### if all_loss is list, sum up individual losses
                if isinstance(all_loss, list) or isinstance(
                    all_loss, type(np.zeros(1))
                ):
                    loss_list.append(sum(all_loss))
                ### if all_loss is single value, just append to loss_list
                else:
                    loss_list.append(all_loss)
        else:
            all_loss_list = [999] * self.popsize
            loss_list = [999] * self.popsize

        ### "return" loss and other optional things
        m_list[0] = loss_list
        if return_results:
            m_list[1] = results
            m_list[2] = all_loss_list

    def _set_fitting_parameters(
        self,
        fitparams,
        pop,
    ):
        """
        Sets all given parameters for the population pop.

        Args:
            fitparams (list):
                list with values for fitting parameters, either a single list or a list
                of lists (first dimension is the number of parameters, second dimension
                is the number of neurons)
            pop (str, optional):
                ANNarchy population name. Default: None, i.e., the tuned population
                is used.
        """
        ### only set parameters of the fitted neuron model
        if pop != self.pop:
            return

        ### get all variables dict (combine fitting variables and const variables)
        all_variables_dict = self.const_params.copy()
        if self.verbose:
            print("OptNeuron: set fitting parameters:")
            print(f"  fitparams: {fitparams} ({self.fitting_variables_name_list})")
            print(f"  starting with const: {all_variables_dict}")

        ### multiply const params for number of neurons
        for const_param_key, const_param_val in all_variables_dict.items():
            if not (isinstance(const_param_val, str)):
                all_variables_dict[const_param_key] = [
                    all_variables_dict[const_param_key]
                ] * self.popsize
        if self.verbose:
            print(f"  adjusting for pop size: {all_variables_dict}")

        ### add fitting variables
        for fitting_variable_idx, fitting_variable_name in enumerate(
            self.fitting_variables_name_list
        ):
            if not (isinstance(fitparams[fitting_variable_idx], list)):
                add_params = [fitparams[fitting_variable_idx]] * self.popsize
            else:
                add_params = fitparams[fitting_variable_idx]
            all_variables_dict[fitting_variable_name] = add_params
        if self.verbose:
            print(f"  add fitting variables: {all_variables_dict}")

        ### evaluate variables defined by a str
        for key, val in all_variables_dict.items():
            if isinstance(val, str):
                all_variables_dict[key] = [
                    ef.evaluate_expression_with_dict(
                        val,
                        {
                            all_variables_key: all_variables_dict[all_variables_key][
                                neuron_idx
                            ]
                            for all_variables_key in all_variables_dict.keys()
                            if not (
                                isinstance(all_variables_dict[all_variables_key], str)
                            )
                        },
                    )
                    for neuron_idx in range(self.popsize)
                ]
        if self.verbose:
            print(f"  add str variables: {all_variables_dict}")

        ### set parameters
        for param_name, param_val in all_variables_dict.items():
            pop_parameter_names = get_population(pop).attributes
            ### only if param_name in parameter attributes
            if param_name in pop_parameter_names:
                if self.popsize == 1:
                    setattr(get_population(pop), param_name, param_val[0])
                else:
                    setattr(get_population(pop), param_name, param_val)

    def _test_fit(self, fitparams_dict):
        """
        Runs the experiment with the optimized parameters obtained with hyperopt and
        returns the loss, the results and all individual losses of the
        get_loss_function.

        Args:
            fitparams_dict (dict):
                dictionary with parameter names (keys) and their values (values)

        Returns:
            fit (dict):
                dictionary containing the loss, the loss variance (in case of noisy
                models with multiple runs per loss calculation), and the status
                (STATUS_OK for hyperopt) and the results generated by the experiment.
        """
        results = self._run_simulator_with_results(
            [fitparams_dict[name] for name in self.fitting_variables_name_list]
        )
        ### if self.popsize > 1 --> transform results, loss etc. to only 1 neuron
        if self.popsize > 1:
            results["loss"] = results["loss"][0]
            results["std"] = results["std"][0]
            results["all_loss"] = results["all_loss"][0]
            results["results"] = self._get_results_of_single_neuron(
                results["results"], 0
            )
        return results

    def _run_with_sbi(self, max_evals, sbi_plot_file):
        """
        Runs the optimization with sbi.

        Args:
            max_evals (int):
                number of runs the optimization method performs

            sbi_plot_file (str):
                If you use "sbi": the name of the figure which will be saved and shows
                the posterior.

        Returns:
            best (dict):
                dictionary containing the optimized parameters and the posterior.
        """
        ### get prior bounds
        prior_min = []
        prior_max = []
        for _, param_bounds in self.variables_bounds.items():
            if isinstance(param_bounds, list):
                prior_min.append(param_bounds[0])
                prior_max.append(param_bounds[1])

        ### run sbi
        simulator, prior = prepare_for_sbi(
            self._sbi_simulation_wrapper,
            self.prior,
            {
                "lower_bound": torch.as_tensor(prior_min),
                "upper_bound": torch.as_tensor(prior_max),
            },
        )
        inference = SNPE(prior, density_estimator="mdn")
        theta, x = simulate_for_sbi(
            simulator=simulator,
            proposal=prior,
            num_simulations=max_evals,
            num_workers=1,
        )
        density_estimator = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior(density_estimator)
        x_o = torch.as_tensor([0])  # data which should be obtained: loss==0
        posterior = posterior.set_default_x(x_o)

        ### get best params
        posterior_samples = posterior.sample(
            (10000,)
        )  # posterior = distribution P(params|data) --> set data and then sample possible parameters
        best_params = posterior_samples[
            torch.argmax(posterior.log_prob(posterior_samples))
        ].numpy()  # sampled parameters with highest prob in posterior

        ### create best dict with best parameters
        best = {}
        for param_idx, param_name in enumerate(self.fitting_variables_name_list):
            best[param_name] = best_params[param_idx]

        ### also return posterior
        best["posterior"] = posterior

        ### plot posterior
        plot_limits = [
            [prior_min[idx], prior_max[idx]] for idx in range(len(prior_max))
        ]
        analysis.pairplot(
            posterior_samples,
            limits=plot_limits,
            ticks=plot_limits,
            fig_size=(5, 5),
            labels=self.fitting_variables_name_list,
        )

        ### save plot
        sf.create_dir("/".join(sbi_plot_file.split("/")[:-1]))
        plt.savefig(sbi_plot_file, dpi=300)

        return best

    def _run_with_bads(self, max_evals):
        """
        TODO
        """

        ### prepare bads
        target = self._bads_simulation_wrapper
        lower, upper, x0, lower_guess, upper_guess = self._get_lower_upper_x0()

        ### TODO bads can handle noisy functions, one can retunr two values, loss and std
        self.bads_params_dict["uncertainty_handling"] = False
        self.bads_params_dict["max_fun_evals"] = max_evals

        ### run bads
        bads = BADS(
            fun=target,
            x0=x0,
            lower_bounds=lower,
            upper_bounds=upper,
            plausible_lower_bounds=lower_guess,
            plausible_upper_bounds=upper_guess,
            options=self.bads_params_dict,
        )
        optimize_result = bads.optimize()

        ### create best dict with best parameters
        best = {
            fitting_variable_name: optimize_result["x"][idx]
            for idx, fitting_variable_name in enumerate(
                self.fitting_variables_name_list
            )
        }
        return best

    @check_types()
    def run(
        self,
        max_evals: int,
        results_file_name: str = "opt_neuron_results/best",
        sbi_plot_file: str = "opt_neuron_plots/posterior.png",
        deap_plot_file: str = "opt_neuron_plots/logbook.png",
        verbose: bool = False,
    ):
        """
        Runs the optimization.

        Args:
            max_evals (int):
                number of runs the optimization method performs
            results_file_name (str, optional):
                name of the file which is saved. The file contains the optimized and
                target results, the obtained parameters, the loss, and the SD of the
                loss (in case of noisy models with multiple runs per loss calculation)
                Default: "best".
            sbi_plot_file (str, optional):
                If you use "sbi": the name of the figure which will be saved and shows
                the posterior. Default: "posterior.png".
            deap_plot_file (str, optional):
                If you use "deap": the name of the figure which will be saved and shows
                the logbook. Default: "logbook.png".
            verbose (bool, optional):
                If True, detailed information is printed. Default: False.

        Returns:
            best (dict):
                dictionary containing the optimized parameters (as keys) and:

                - "loss": the loss
                - "all_loss": the individual losses of the get_loss_function
                - "std": the SD of the loss (in case of noisy models with multiple
                    runs per loss calculation)
                - "results": the results generated by the experiment
                - "results_soll": the target results
        """
        self.verbose = False
        self.verbose_run = verbose
        self.loss_history = []
        self.start_time = time()
        if self.method == "hyperopt":
            ### run optimization with hyperopt and return best dict
            best = fmin(
                fn=self._run_simulator,
                space=self.fv_space,
                algo=tpe.suggest,
                max_evals=max_evals,
            )
        elif self.method == "sbi":
            ### run optimization with sbi and return best dict
            best = self._run_with_sbi(max_evals, sbi_plot_file)
        elif self.method == "deap":
            best = self._run_with_deap(max_evals, deap_plot_file)
        elif self.method == "bads":
            if max_evals < 4:
                raise ValueError("bads needs at least 4 evaluations")
            best = self._run_with_bads(max_evals)
        else:
            print("ERROR run; method should be 'hyperopt', 'sbi', 'deap', or 'bads'")
            quit()
        ### obtain loss for the best parameters
        fit = self._test_fit(best)
        best["loss"] = float(fit["loss"])
        best["all_loss"] = fit["all_loss"]
        best["std"] = fit["std"]
        best["results"] = fit["results"]
        best["results_soll"] = self.results_soll
        self.results = best

        ### create loss history array
        self.loss_history = np.array(self.loss_history)

        ### SAVE OPTIMIZED PARAMS AND LOSS
        ### save as pkl file
        sf.save_variables(
            [best],
            [results_file_name.split("/")[-1]],
            (
                "/".join(results_file_name.split("/")[:-1])
                if len(results_file_name.split("/")) > 1
                else "./"
            ),
        )
        ### save human readable as json file
        json.dump(
            {key: best[key] for key in self.fitting_variables_name_list + ["loss"]},
            open(
                f"{results_file_name}.json",
                "w",
            ),
            indent=4,
        )

        return best

    def _run_with_deap(self, max_evals, deap_plot_file):
        """
        Runs the optimization with deap.

        Args:
            max_evals (int):
                number of runs (here generations) the optimization method performs

            deap_plot_file (str):
                the name of the figure which will be saved and shows the logbook
        """

        return self._deap_cma.run(
            max_evals=max_evals,
            plot_file=deap_plot_file,
        )


### old name for backward compatibility, TODO remove
opt_neuron = OptNeuron
