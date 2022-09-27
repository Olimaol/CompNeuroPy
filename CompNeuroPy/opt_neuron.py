from ANNarchy import setup, Population, get_population, reset, Neuron, clear
import numpy as np
import traceback
from CompNeuroPy import system_functions as sf
from CompNeuroPy import generate_model as gm
from CompNeuroPy import extra_functions as ef
import matplotlib.pyplot as plt

# hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK

# multiprocessing
from multiprocessing import Process
import multiprocessing
from sbi import analysis as analysis

# sbi
import torch
from sbi import utils as utils
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


class opt_neuron:

    opt_created = []

    def __init__(
        self,
        experiment,
        get_loss_function,
        variables_bounds,
        neuron_model,
        results_soll=None,
        target_neuron_model=None,
        time_step=1,
        compile_folder_name="annarchy_opt_neuron",
        num_rep_loss=1,
        method="hyperopt",
        prior=None,
        fv_space=None,
    ):
        """
        This class prepares the optimization. To run the optimization call opt_neuron.run().

        Args:
            experiment: CompNeuroPy Experiment class
                the experiment class has to contain the 'run' function which defines the simulations and recordings

            get_loss_function: function
                function which takes results_ist and results_soll as arguments and calculates the loss

            variables_bounds: dict
                keys = parameter names, values = either list with len=2 (lower and upper bound) or a single value (constant parameter)

            neuron_model: ANNarchy Neuron object
                the neuron model used during optimization

            results_soll: dict, optional, default=None
                some variable which contains the target data and can be used by the get_loss_function (second argument of get_loss_function)
                either provide results_soll or a target_neuron_model not both!

            target_neuron_model: ANNarchy Neuron object, optional, default=None
                the neuron model which produces the target data by running the experiment
                either provide results_soll or a target_neuron_model not both!

            time_step: float, optional, default=1
                the time step for the simulation in ms

            compile_folder_name: string, default = 'annarchy_opt_neuron'
                the name of the annarchy compilation folder

            num_rep_loss: int, default = 1
                only interesting for noisy simulations/models
                how often should the model be run to calculate the loss (the defined number of losses is obtained and averaged)

            method: str, default = 'hyperopt'
                either 'sbi' or 'hyperopt'

            prior: distribution, default = None
                the prior distribution used by sbi
                if none is given, uniform distributions between the variable bounds are assumed

            fv_space: list, default = None
                the search space for hyperopt
                if none is given, uniform distributions between the variable bounds are assumed
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
            self.results_soll = results_soll
            self.experiment = experiment
            self.variables_bounds = variables_bounds
            self.fitting_variables_name_list = (
                self.__get_fitting_variables_name_list__()
            )
            self.method = method
            if method == "hyperopt":
                if fv_space is None:
                    self.fv_space = self.__get_hyperopt_space__()
                else:
                    self.fv_space = fv_space
            self.const_params = self.__get_const_params__()
            self.num_rep_loss = num_rep_loss
            self.neuron_model = neuron_model
            if method == "sbi":
                self.prior = self.__get_prior__(prior)
            self.target_neuron = target_neuron_model
            self.compile_folder_name = compile_folder_name
            self.__get_loss__ = get_loss_function

            ### check target_neuron/results_soll
            self.__check_target__()
            ### check neuron models
            self.__check_neuron_models__()

            ### setup ANNarchy
            setup(dt=time_step)

            ### create and compile model
            ### if neuron models and target neuron model --> create both models then test, then clear and create only model for neuron model
            model, target_model = self.__generate_models__()

            self.pop = model.populations[0]
            if target_model is not None:
                self.pop_target = target_model.populations[0]
            else:
                self.pop_target = None

            ### check variables of model
            self.__test_variables__()

            ### check neuron models, experiment, get_loss
            ### if results_soll is None -_> generate results_soll
            self.__check_get_loss_function__()

            ### after checking neuron models, experiment, get_loss
            ### if two models exist --> clear ANNarchy and create/compile again only standard model
            clear()
            model, _ = self.__generate_models__()

    def __generate_models__(self):
        """
        generates the model and target_model (only if results_soll is None --> they have to be generated)

        returns model and target_model

        if there are already results_soll --> target_model=None
        """
        with ef.suppress_stdout():
            model = None
            target_model = None
            if self.results_soll is None:
                ### create two models
                model = gm.generate_model(
                    model_creation_function=self.__raw_neuron__,
                    model_kwargs={"neuron": self.neuron_model, "name": "model_neuron"},
                    name="standard_model",
                    do_create=True,
                    do_compile=False,
                    compile_folder_name=self.compile_folder_name,
                )

                target_model = gm.generate_model(
                    model_creation_function=self.__raw_neuron__,
                    model_kwargs={
                        "neuron": self.target_neuron,
                        "name": "target_model_neuron",
                    },
                    name="target_model",
                    do_create=True,
                    do_compile=True,
                    compile_folder_name=self.compile_folder_name,
                )

            else:
                ### create one model
                model = gm.generate_model(
                    model_creation_function=self.__raw_neuron__,
                    model_kwargs={"neuron": self.neuron_model, "name": "model_neuron"},
                    name="single_model",
                    do_create=True,
                    do_compile=True,
                    compile_folder_name=self.compile_folder_name,
                )
        return [model, target_model]

    def __check_neuron_models__(self):
        if not (isinstance(self.neuron_model, type(Neuron()))) or (
            self.target_neuron is not None
            and not (isinstance(self.target_neuron, type(Neuron())))
        ):
            print(
                "opt_neuron: Error: neuron_model and/or target_neuron_model have to be ANNarchy neuron models"
            )
            quit()

    def __check_target__(self):
        """
        check if either results_soll or target_neuron are provided
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

    def __get_prior__(self, prior):
        """
        returns the prior for sbi optimization
        if prior == None --> uniform distributions between variable bounds
        else use given prior
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

    def __get_fitting_variables_name_list__(self):
        """
        returns list with fitting variable names
        """
        name_list = []
        for param_name, param_bounds in self.variables_bounds.items():
            if isinstance(param_bounds, list):
                name_list.append(param_name)
        return name_list

    def __get_hyperopt_space__(self):
        """
        generates the hyperopt variable space from the fitting variable bounds
        the variable space is a uniform distribution between the bounds
        """
        fitting_variables_space = []
        for param_name, param_bounds in self.variables_bounds.items():
            if isinstance(param_bounds, list):
                fitting_variables_space.append(
                    hp.uniform(param_name, param_bounds[0], param_bounds[1])
                )
        return fitting_variables_space

    def __get_const_params__(self):
        """
        returns a dict with cosntant parameter names (keys) and their values
        """
        const_params = {}
        for param_name, param_bounds in self.variables_bounds.items():
            if not (isinstance(param_bounds, list)):
                const_params[param_name] = param_bounds
        return const_params

    def __check_get_loss_function__(self):
        """
        function: function whith arguments (results_ist, results_soll) which computes and returns a loss value

        Checks if function is compatible to the experiment.
        To test, function is run with results_soll (also for results_ist argument).
        """

        ### TODO: self.pop and self.pop_target
        ### if self.pop_target is None --> only self.pop and results soll --> simulate with self.pop --> create results_ist
        ### if not --> no results_soll --> simulate with self.pop/create results_ist and simulate with self.pop_target and create results_soll

        print("checking neuron_models, experiment, get_loss...", end="")

        fitparams = []
        for bounds in self.variables_bounds:
            if isinstance(bounds, list):
                fitparams.append(bounds[0])

        if self.results_soll is not None:
            ### only generate results_ist with standard neuron model
            results_ist = self.__run_simulator_with_results__(fitparams)["results"]
        else:
            ### run simulator with both populations (standard neuron model and target neuron model) and generatate results_ist and results_soll
            results_ist = self.__run_simulator_with_results__(fitparams)["results"]
            self.results_soll = self.__run_simulator_with_results__(
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

    def __raw_neuron__(self, neuron, name):
        """
        generates a population with one neuron
        """
        Population(1, neuron=neuron, name=name)

    def __test_variables__(self):
        """
        self.pop: name of created neuron population
        self.fv_space: hyperopt variable space list
        self.const_params: dictionary with constant variables

        Checks if self.pop contains all variables of self.fv_space and self.const_params and if self.pop has more variables.
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
                "opt_neuron: Error: The neuron_model does not contain parameters",
                all_vars_names,
                "!",
            )
            quit()

    def __run_simulator__(self, fitparams):
        """
        runs the function simulator with the multiprocessing manager (if function is called multiple times this saves memory, otherwise same as calling simulator directly)

        fitparams: list, for description see function simulator
        return: returns dictionary needed for optimization with hyperopt
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
                target=self.__simulator__, args=(fitparams, rng, m_list, return_results)
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

    def __sbi_simulation_wrapper__(self, fitparams):
        """
        this function is called by sbi infer
        fitparams: tensor
            either a batch of parameters (tenor with two dimensions) or a single parameter set
        returns: loss as tensor for sbi inference
        """
        fitparams = np.asarray(fitparams)
        if len(fitparams.shape) == 2:
            ### batch parameters!
            data = []
            for idx in range(fitparams.shape[0]):
                data.append(self.__run_simulator__(fitparams[idx])["loss"])
        else:
            ### single parameter set!
            data = [self.__run_simulator__(fitparams)["loss"]]

        return torch.as_tensor(data)

    def __run_simulator_with_results__(self, fitparams, pop=None):
        """
        runs the function simulator with the multiprocessing manager (if function is called multiple times this saves memory, otherwise same as calling simulator directly)

        fitparams: list, for description see function simulator
        return: returns dictionary needed for optimization with hyperopt
        """
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
                target=self.__simulator__,
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

        ### return loss and other things for optimization
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

    def __simulator__(
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
            reset_function=reset_function, reset_kwargs=reset_kwargs
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

        ### set fitting parameters
        for idx in range(len(fitparams)):
            setattr(
                get_population(pop),
                self.fitting_variables_name_list[idx],
                fitparams[idx],
            )

        ### set constant parameters
        for key, val in self.const_params.items():
            if isinstance(val, str):
                try:
                    ### value is str --> name of variable in fitting parameters
                    setattr(
                        get_population(pop),
                        key,
                        fitparams[
                            np.where(np.array(self.fitting_variables_name_list) == val)[
                                0
                            ][0]
                        ],
                    )
                except:
                    try:
                        ### or name of variable in other const parameters
                        setattr(get_population(pop), key, self.const_params[val])
                    except:
                        print(
                            "ERROR: during setting const parameter "
                            + key
                            + " value for "
                            + val
                            + " not found in fitting parameters or other const parameters!"
                        )
                        quit()
            else:
                setattr(get_population(pop), key, val)

    def __test_fit__(self, fitparamsDict):
        """
        fitparamsDict: dictionary with parameters, format = as hyperopt returns fit results

        Thus, this function can be used to run the simulator function directly with fitted parameters obtained with hyperopt

        Returns the loss computed in simulator function.
        """
        return self.__run_simulator_with_results__(
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
            self.__sbi_simulation_wrapper__,
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
            self.__sbi_simulation_wrapper__,
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
        folder_test = "/".join(sbi_plot_file.split("/")[:-1])
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
                fn=self.__run_simulator__,
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
