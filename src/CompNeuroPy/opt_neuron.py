from ANNarchy import setup, Population, get_population, Neuron, clear
import numpy as np
import traceback
from CompNeuroPy import system_functions as sf
from CompNeuroPy import extra_functions as ef
from CompNeuroPy.monitors import CompNeuroMonitors
from CompNeuroPy.generate_model import CompNeuroModel
from CompNeuroPy.experiment import CompNeuroExp
import matplotlib.pyplot as plt
import sys
from typing import Callable, Any, Type
from typingchecker import check_types

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
        "OptNeuron: Error: You need to install hyperopt, torch, and sbi to use OptNeuron (e.g. use pip install hyperopt torch sbi))"
    )
    sys.exit()


class OptNeuron:
    """
    This class is used to optimize neuron models with ANNarchy.
    """

    opt_created = []

    @check_types()
    def __init__(
        self,
        experiment: Type[CompNeuroExp],
        get_loss_function: Callable[[Any, Any], float | list[float]],
        variables_bounds: dict[str, float | list[float]],
        neuron_model: Neuron,
        results_soll: Any | None = None,
        target_neuron_model: Neuron | None = None,
        time_step: float = 1.0,
        compile_folder_name: str = "annarchy_OptNeuron",
        num_rep_loss: int = 1,
        method: str = "hyperopt",
        prior=None,
        fv_space: list = None,
        record: list[str] = [],
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
                Default: 'annarchy_OptNeuron'.

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
                "OptNeuron: Error: Already another OptNeuron created. Only create one per python session!"
            )
            quit()
        else:
            print(
                "OptNeuron: Initialize OptNeuron... do not create anything with ANNarchy before!"
            )

            ### set object variables
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
            self.__get_loss__ = get_loss_function

            ### check target_neuron/results_soll
            self._check_target()
            ### check neuron models
            self._check_neuron_models()

            ### setup ANNarchy
            setup(dt=time_step)

            ### create and compile model
            ### if neuron models and target neuron model --> create both models then
            ### test, then clear and create only model for neuron model
            model, target_model, monitors = self._generate_models()

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
            ### if two models exist --> clear ANNarchy and create/compile again only
            ### standard model, thus recreate also monitors and experiment
            clear()
            model, _, monitors = self._generate_models()
            self.monitors = monitors
            self.experiment = experiment(monitors=monitors)

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
                            pop_name: self.record
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
                    monitors = CompNeuroMonitors({model.populations[0]: self.record})

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
        Runs the experiment with the given parameters and 'returns' the loss and
        optionally the results and all individual losses of the get_loss_function. The
        'returned' values are saved in m_list.

        Args:
            fitparams (list):
                list with values for fitting parameters

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

        ### conduct loaded experiment
        results = self.experiment.run(pop)

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

    def _set_fitting_parameters(
        self,
        fitparams,
        pop=None,
    ):
        """
        Sets all given parameters for the population pop.

        Args:
            pop (str, optional):
                ANNarchy population name. Default: None, i.e., the tuned population
                is used.
        """
        if pop is None:
            pop = self.pop

        ### get all variables dict (combine fitting variables and const variables)
        all_variables_dict = self.const_params.copy()

        for fitting_variable_idx, fitting_variable_name in enumerate(
            self.fitting_variables_name_list
        ):
            all_variables_dict[fitting_variable_name] = fitparams[fitting_variable_idx]

        ### evaluate variables defined by a str
        for key, val in all_variables_dict.items():
            if isinstance(val, str):
                all_variables_dict[key] = ef.evaluate_expression_with_dict(
                    val, all_variables_dict
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
        return self._run_simulator_with_results(
            [fitparams_dict[name] for name in self.fitting_variables_name_list]
        )

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
        plt.savefig(sbi_plot_file)

        return best

    @check_types()
    def run(
        self,
        max_evals: int,
        results_file_name: str = "best",
        sbi_plot_file: str = "posterior.svg",
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
                the posterior. Default: "posterior.svg".

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
        else:
            print("ERROR run; method should be 'hyperopt' or 'sbi'")
            quit()
        fit = self._test_fit(best)
        best["loss"] = fit["loss"]
        if self.method == "sbi":
            print("\tbest loss:", best["loss"])
        best["all_loss"] = fit["all_loss"]
        best["std"] = fit["std"]
        best["results"] = fit["results"]
        best["results_soll"] = self.results_soll
        self.results = best

        ### SAVE OPTIMIZED PARAMS AND LOSS
        sf.save_variables([best], [results_file_name], "parameter_fit")

        return best


### old name for backward compatibility, TODO remove
opt_neuron = OptNeuron
