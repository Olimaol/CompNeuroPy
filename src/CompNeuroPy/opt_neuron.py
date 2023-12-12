from ANNarchy import setup, Population, get_population, reset, Neuron, clear
import numpy as np
import traceback
from CompNeuroPy import system_functions as sf
from CompNeuroPy import extra_functions as ef
from CompNeuroPy.monitors import CompNeuroMonitors
from CompNeuroPy.generate_model import CompNeuroModel
from CompNeuroPy.experiment import CompNeuroExp
import matplotlib.pyplot as plt
import sys
from typing import Callable, Any

# multiprocessing
from multiprocessing import Process
import multiprocessing

try:
    # hyperopt
    from hyperopt import fmin, tpe, hp, STATUS_OK

    # torch
    import torch

    # sbi
    from sbi import analysis as analysis
    from sbi import utils as utils
    from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
except:
    print(
        "opt_neuron: Error: You need to install hyperopt, torch, and sbi to use opt_neuron (e.g. use pip install hyperopt torch sbi))"
    )
    sys.exit()


class opt_neuron:
    """
    This class is used to optimize neuron models with ANNarchy.
    """

    opt_created = []

    def __init__(
        self,
        experiment: CompNeuroExp,
        get_loss_function: Callable[[Any, Any], float | list[float]],
        variables_bounds: dict[str, float | list[float]],
        neuron_model: Neuron,
        results_soll: Any | None = None,
        target_neuron_model: Neuron | None = None,
        time_step: float | None = 1.0,
        compile_folder_name: str = "annarchy_opt_neuron",
        num_rep_loss: int = 1,
        method: str = "hyperopt",
        prior=None,
        fv_space: list = None,
        record: list[str] = [],
    ):
        """
        This prepares the optimization. To run the optimization call the run function.

        Args:
            experiment (CompNeuroExp):
                CompNeuroExp object containing a 'run' function which defines the
                simulations and recordings

            get_loss_function (function):
                function which takes results_ist and results_soll as arguments and
                calculates/returns the loss

            variables_bounds (dict):
                Dictionary with parameter names (keys) and their bounds (values). If
                single values are given as values, the parameter is constant, i.e., not
                optimized. If a list is given as value, the parameter is optimized and
                the list contains the lower and upper bound of the parameter (order is
                not important).

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

            compile_folder_name (string, optional):
                The name of the annarchy compilation folder within annarchy_folders/.
                Default: 'annarchy_opt_neuron'.

            num_rep_loss (int, optional):
                Only interesting for noisy simulations/models. How often should the
                simulaiton be run to calculate the loss (the defined number of losses
                is obtained and averaged). Default: 1.

            method (str, optional):
                Either 'sbi' or 'hyperopt'. If 'sbi' is used, the optimization is
                performed with sbi. If 'hyperopt' is used, the optimization is
                performed with hyperopt. Default: 'hyperopt'.

            prior (distribution, optional):
                The prior distribution used by sbi. Default: None, i.e., uniform
                distributions between the variable bounds are assumed.

            fv_space (list, optional):
                The search space for hyperopt. Default: None, i.e., uniform
                distributions between the variable bounds are assumed.

            record (list, optional):
                List of strings which define what variables of the tuned neuron should
                be recorded. Default: [].
        """

        if len(self.opt_created) > 0:
            print(
                "opt_neuron: Error: Already another opt_neuron created. Only create one per python session!"
            )
            quit()
        else:
            print(
                "opt_neuron: Initialize opt_neuron... do not create anything with ANNarchy before!"
            )

            ### set object variables
            self.opt_created.append(1)
            self.record = record
            self.results_soll = results_soll
            self.experiment = experiment
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
            self.__get_loss__ = get_loss_function

            ### check target_neuron/results_soll
            self._check_target()
            ### check neuron models
            self._check_neuron_models()

            ### setup ANNarchy
            setup(dt=time_step)

            ### create and compile model
            ### if neuron models and target neuron model --> create both models then test,
            ### then clear and create only model for neuron model
            model, target_model, monitors = self._generate_models()

            self.pop = model.populations[0]
            if target_model is not None:
                self.pop_target = target_model.populations[0]
            else:
                self.pop_target = None
            self.monitors = monitors

            ### check variables of model
            self._test_variables()

            ### check neuron models, experiment, get_loss
            ### if results_soll is None -_> generate results_soll
            self._check_get_loss_function()

            ### after checking neuron models, experiment, get_loss
            ### if two models exist --> clear ANNarchy and create/compile again only standard model
            clear()
            model, _, monitors = self._generate_models()
            self.monitors = monitors

    def _generate_models(self):
        """
        Generates the tuned model and the target_model (only if results_soll is None).

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
        with ef.suppress_stdout():
            model = None
            target_model = None
            monitors = None
            if self.results_soll is None:
                ### create two models
                model = CompNeuroModel(
                    model_creation_function=self._raw_neuron,
                    model_kwargs={"neuron": self.neuron_model, "name": "model_neuron"},
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
                    },
                    name="target_model",
                    do_create=True,
                    do_compile=True,
                    compile_folder_name=self.compile_folder_name,
                )

                ### create monitors
                if len(self.record) > 0:
                    monitors = CompNeuroMonitors(
                        {
                            f"pop;{pop_name}": self.record
                            for pop_name in [
                                model.populations[0],
                                target_model.populations[0],
                            ]
                        }
                    )

            else:
                ### create one model
                model = CompNeuroModel(
                    model_creation_function=self._raw_neuron,
                    model_kwargs={"neuron": self.neuron_model, "name": "model_neuron"},
                    name="single_model",
                    do_create=True,
                    do_compile=True,
                    compile_folder_name=self.compile_folder_name,
                )
                ### create monitors
                if len(self.record) > 0:
                    monitors = CompNeuroMonitors(
                        {f"pop;{model.populations[0]}": self.record}
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
                "opt_neuron: Error: neuron_model and/or target_neuron_model have to be ANNarchy neuron models"
            )
            quit()

    def _check_target(self):
        """
        Check if either results_soll or target_neuron are provided and not both.
        """
        if self.target_neuron is None and self.results_soll is None:
            print(
                "opt_neuron: Error: Either provide results_soll or target_neuron_model"
            )
            quit()
        elif self.target_neuron is not None and self.results_soll is not None:
            print(
                "opt_neuron: Error: Either provide results_soll or target_neuron_model, not both"
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
            self.__get_loss__(results_ist, self.results_soll)
        except:
            print(
                "\nThe get_loss_function, experiment and neuron model(s) are not compatible:\n"
            )
            traceback.print_exc()
            quit()
        print("Done\n")

    def _raw_neuron(self, neuron, name):
        """
        Generates a population with one neuron of the given neuron model.

        Args:
            neuron (ANNarchy Neuron):
                The neuron model.

            name (str):
                The name of the population.
        """
        Population(1, neuron=neuron, name=name)

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
                "opt_neuron: WARNING: attributes",
                pop_parameter_names,
                "are not used/initialized.",
            )
        if len(all_vars_names) > 0:
            print(
                "opt_neuron: WARNING: The neuron_model does not contain parameters",
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
                list with values for fitting parameters

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
        lossAr = np.zeros(self.num_rep_loss)

        return_results = False
        for nr_run in range(self.num_rep_loss):
            ### initialize for each run a new rng (--> not always have same noise in case of noisy models/simulations)
            rng = np.random.default_rng()
            ### run simulator with multiprocessign manager
            proc = Process(
                target=self._simulator, args=(fitparams, rng, m_list, return_results)
            )
            proc.start()
            proc.join()
            ### get simulation results/loss
            lossAr[nr_run] = m_list[0]

        ### calculate mean and std of loss
        if self.num_rep_loss > 1:
            loss = np.mean(lossAr)
            std = np.std(lossAr)
        else:
            loss = lossAr[0]
            std = None

        ### return loss and other things for optimization
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
            for idx in range(fitparams.shape[0]):
                data.append(self._run_simulator(fitparams[idx])["loss"])
        else:
            ### single parameter set!
            data = [self._run_simulator(fitparams)["loss"]]

        return torch.as_tensor(data)

    def _run_simulator_with_results(self, fitparams, pop=None):
        """
        Runs the function simulator with the multiprocessing manager (if function is
        called multiple times this saves memory, otherwise same as calling simulator
        directly) and also returns the results.

        Args:
            fitparams (list):
                list with values for fitting parameters

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

        ### in case of noisy models, here optionally run multiple simulations, to mean the loss
        lossAr = np.zeros(self.num_rep_loss)
        all_loss_list = []
        return_results = True
        for nr_run in range(self.num_rep_loss):
            ### initialize for each run a new rng (--> not always have same noise in case of noisy models/simulations)
            rng = np.random.default_rng()
            ### run simulator with multiprocessign manager
            proc = Process(
                target=self._simulator,
                args=(fitparams, rng, m_list, return_results, pop),
            )
            proc.start()
            proc.join()
            ### get simulation results/loss
            lossAr[nr_run] = m_list[0]
            results_ist = m_list[1]
            all_loss_list.append(m_list[2])

        all_loss_arr = np.array(all_loss_list)
        ### calculate mean and std of loss
        if self.num_rep_loss > 1:
            loss = np.mean(lossAr)
            std = np.std(lossAr)
            all_loss = np.mean(all_loss_arr, 0)
        else:
            loss = lossAr[0]
            std = None
            all_loss = all_loss_arr[0]

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
        fitparams: from hyperopt

        m_list: variable to store results, from multiprocessing
        """
        if pop is None:
            pop = self.pop

        ### reset model and set parameters which should not be optimized and parameters which should be optimized
        self.__set_fitting_parameters__(fitparams, pop=pop)

        ### conduct loaded experiment
        reset_function = self.__set_fitting_parameters__
        reset_kwargs = {"fitparams": fitparams, "pop": pop}

        exp_obj = self.experiment(
            monitors=self.monitors,
            reset_function=reset_function,
            reset_kwargs=reset_kwargs,
        )
        results = exp_obj.run(pop)

        if self.results_soll is not None:
            ### compute loss
            all_loss = self.__get_loss__(results, self.results_soll)
            if isinstance(all_loss, list) or isinstance(all_loss, type(np.zeros(1))):
                loss = sum(all_loss)
            else:
                loss = all_loss
        else:
            all_loss = 999
            loss = 999
        ### "return" loss and other optional things
        m_list[0] = loss
        if return_results:
            m_list[1] = results
            m_list[2] = all_loss

    def __replace_substrings_except_within_braces__(
        self, input_string, replacement_mapping
    ):
        result = []
        inside_braces = False
        i = 0

        while i < len(input_string):
            if input_string[i] == "{":
                inside_braces = True
                result.append(input_string[i])
                i += 1
            elif input_string[i] == "}":
                inside_braces = False
                result.append(input_string[i])
                i += 1
            else:
                if not inside_braces:
                    found_match = False
                    for old_substr, new_substr in replacement_mapping.items():
                        if input_string[i : i + len(old_substr)] == old_substr:
                            result.append(new_substr)
                            i += len(old_substr)
                            found_match = True
                            break
                    if not found_match:
                        result.append(input_string[i])
                        i += 1
                else:
                    result.append(input_string[i])
                    i += 1

        return "".join(result)

    def __replace_keys_with_values__(self, dictionary, value_key, value):
        try:
            new_value = value
            sorted_keys = sorted(list(dictionary.keys()), key=len, reverse=True)
            ### first replace largest keys --> if smaller keys are within larger keys this should not cause a problem
            for key in sorted_keys:
                if key in new_value:
                    ### replace the key in the value
                    ### only replace things which are not between {}
                    new_value = self.__replace_substrings_except_within_braces__(
                        new_value, {key: "{" + key + "}"}
                    )
            ### evaluate the value with the values of the dictionary
            new_value = eval(new_value.format(**dictionary))
        except:
            exc_type, exc_value, _ = sys.exc_info()
            error_message = traceback.format_exception_only(exc_type, exc_value)
            raise ValueError(
                " ".join(
                    [
                        f"ERROR opt_neuron: evaluate the value {value} of parameter {value_key}"
                    ]
                    + error_message
                )
            )

        return new_value

    def __set_fitting_parameters__(
        self,
        fitparams,
        pop=None,
        populations=True,
        projections=False,
        synapses=False,
        monitors=True,
    ):
        """
        self.pop: ANNarchy population name
        fitparams: list with values for fitting parameters
        self.fv_space: hyperopt variable space list
        self.const_params: dictionary with constant variables

        Sets all given parameters for the population self.pop.
        """
        if pop is None:
            pop = self.pop

        ### reset model to compilation state
        reset(
            populations=populations,
            projections=projections,
            synapses=synapses,
            monitors=monitors,
        )

        ### get all variables dict (combine fitting variables and const variables)
        all_variables_dict = self.const_params.copy()

        for fitting_variable_idx, fitting_variable_name in enumerate(
            self.fitting_variables_name_list
        ):
            all_variables_dict[fitting_variable_name] = fitparams[fitting_variable_idx]

        ### evaluate variables defined by a str
        for key, val in all_variables_dict.items():
            if isinstance(val, str):
                all_variables_dict[key] = self.__replace_keys_with_values__(
                    all_variables_dict, key, val
                )

        ### only set parameters of the fitted neuron model (in case target neuron model is given)
        if pop == self.pop:
            ### set parameters
            for param_name, param_val in all_variables_dict.items():
                pop_parameter_names = get_population(pop).attributes
                ### only if param_name in parameter attributes
                if param_name in pop_parameter_names:
                    setattr(
                        get_population(pop),
                        param_name,
                        param_val,
                    )

    def __test_fit__(self, fitparamsDict):
        """
        fitparamsDict: dictionary with parameters, format = as hyperopt returns fit results

        Thus, this function can be used to run the simulator function directly with fitted parameters obtained with hyperopt

        Returns the loss computed in simulator function.
        """
        return self._run_simulator_with_results(
            [fitparamsDict[name] for name in self.fitting_variables_name_list]
        )

    def __run_with_sbi__(self, max_evals, sbi_plot_file):
        """
        runs the optimization with sbi
        """
        ### get prior bounds
        prior_min = []
        prior_max = []
        for _, param_bounds in self.variables_bounds.items():
            if isinstance(param_bounds, list):
                prior_min.append(param_bounds[0])
                prior_max.append(param_bounds[1])

        ### obtain posterior
        """posterior = infer(
            self._sbi_simulation_wrapper,
            self.prior,
            method="SNPE",
            num_simulations=max_evals,
            num_workers=1,
            custom_prior_wrapper_kwargs={
                "lower_bound": torch.as_tensor(prior_min),
                "upper_bound": torch.as_tensor(prior_max),
            },
        )
        x_o = torch.as_tensor([0])  # data which should be obtained: loss==0
        posterior = posterior.set_default_x(x_o)"""

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
        plt.savefig(sbi_plot_file)

        return best

    def run(
        self, max_evals, results_file_name="best.npy", sbi_plot_file="posterior.svg"
    ):
        """
        run the optimization

        Args:
            max_evals: int
                number of runs (sample: paramter -> loss) the optimization method performs

            results_file_name: str, optional, default="best.npy"
                name of the file which is saved. The file contains the optimized and
                target results, the obtained parameters, the loss, and the SD of the
                loss (in case of noisy models with multiple runs per loss calculation)

            sbi_plot_file: str, optional, default="posterior.svg"
                If you use "sbi": the name of the figure which will be saved and shows
                the posterior.

        """
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
            best = self.__run_with_sbi__(max_evals, sbi_plot_file)
        else:
            print("ERROR run; method should be 'hyperopt' or 'sbi'")
            quit()
        fit = self.__test_fit__(best)
        best["loss"] = fit["loss"]
        if self.method == "sbi":
            print("\tbest loss:", best["loss"])
        best["all_loss"] = fit["all_loss"]
        best["std"] = fit["std"]
        best["results"] = fit["results"]
        best["results_soll"] = self.results_soll
        self.results = best

        ### SAVE OPTIMIZED PARAMS AND LOSS
        sf.save_data([best], ["parameter_fit/" + results_file_name])
