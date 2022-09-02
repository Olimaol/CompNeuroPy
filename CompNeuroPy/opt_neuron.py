from ANNarchy import setup, Population, get_population, reset
import numpy as np
import traceback
import CompNeuroPy.neuron_models as nm
import CompNeuroPy.model_functions as mf
import CompNeuroPy.system_functions as syf
from .Experiment import Experiment
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
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


class opt_neuron:

    opt_created = []

    def __init__(
        self,
        results_soll,
        experiment,
        get_loss_function,
        variables_bounds,
        compile_folder_name="annarchy_opt_neuron",
        num_rep_loss=1,
        neuron_model=None,
        method="hyperopt",
        prior=None,
        fv_space=None,
    ):
        """
        Args:
            results_soll: dict
                a results dictionary with recordings from the experiment (target results)

            experiment: function
                the 'run' function from a CompNeuroPy Experiment class

            get_loss_function: function
                fucniton which takes results_ist and results_soll as arguments and calculates the loss

            variables_bounds: dict
                keys = parameter names, values = either list with len=2 (lower and upper bound) or a single value (constant parameter)

            compile_folder_name: string, default = 'annarchy_opt_neuron'
                the name of the annarchy compilation folder

            num_rep_loss: int, default = 1
                only interesting for noisy simulations/models
                how often should the model be run to calculate the loss (the defined number of losses is obtained and averaged)

            neuron_model: ANNarchy Neuron obj, default = None
                the neuron model used during optimization

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
                "opt_neuron: Initialize opt_neuron... better not create anything with ANNarchy before!"
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

            ### check get_loss function compatibility with experiment
            self.__get_loss__ = self.__check_get_loss_function__(get_loss_function)

            ### setup ANNarchy
            setup(dt=results_soll["recordings"][0]["dt"])

            ### create and compile model
            model = self.__raw_neuron__(
                do_compile=True, compile_folder_name=compile_folder_name
            )
            self.iz = model[0]

            ### check variables
            self.__test_variables__()

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

    def __check_get_loss_function__(self, function):
        """
        function: function whith arguments (results_ist, results_soll) which computes and returns a loss value

        Checks if function is compatible to the experiment.
        To test, function is run with results_soll (also for results_ist argument).
        """

        try:
            function(self.results_soll, self.results_soll)
        except:
            print(
                "\nThe get_loss_function is not compatible to the specified experiment:\n"
            )
            traceback.print_exc()
            quit()

        return function

    def __raw_neuron__(
        self, do_compile=False, compile_folder_name="annarchy_opt_neuron"
    ):
        """
        generates one neuron, default=Izhikevich neuron model, and optionally compiles the network

        returns a list of the names of the populations (for later access)
        """
        if isinstance(self.neuron_model, type(nm.Izhikevich2007)):
            pop = Population(1, neuron=self.neuron_model, name="user_defined_neuron")
            ret = ["user_defined_neuron"]
        else:
            pop = Population(1, neuron=nm.Izhikevich2007, name="Iz_neuron")
            ret = ["Iz_neuron"]

        if do_compile:
            mf.compile_in_folder(compile_folder_name)

        return ret

    def __test_variables__(self):
        """
        self.iz: name of created neuron population
        self.fv_space: hyperopt variable space list
        self.const_params: dictionary with constant variables

        Checks if self.iz contains all variables of self.fv_space and self.const_params and if self.iz has more variables.
        """
        ### collect all names
        all_vars_names = np.concatenate(
            [
                np.array(list(self.const_params.keys())),
                np.array(self.fitting_variables_name_list),
            ]
        ).tolist()
        ### check if pop has these parameters
        pop_parameter_names = get_population(self.iz).attributes.copy()
        for name in pop_parameter_names.copy():
            if name in all_vars_names:
                all_vars_names.remove(name)
                pop_parameter_names.remove(name)
        if len(pop_parameter_names) > 0:
            print(
                "opt_neuron: WARNING: attributes",
                pop_parameter_names,
                "of population",
                self.iz,
                "are not used/initialized.",
            )
        if len(all_vars_names) > 0:
            print(
                "opt_neuron: Error: Population",
                self.iz,
                "does not contain parameters",
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

    def __run_simulator_with_results__(self, fitparams):
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
        all_loss_list = []
        return_results = True
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

    class myexp(Experiment):
        """
        cnp Experiment class
        The method "run" is given as an argument during object instantiation
        """

        def __init__(self, reset_function, reset_kwargs, experiment_function):
            """
            runs the init of the standard (parent) Experiment class, which defines the reset function of the Experiment object
            additionally stores the experiment funciton which can later be used in the run() function
            the experiment function is given, this enables to use a loaded run function from a previously defined Experiment Object
            """
            super().__init__(reset_function, reset_kwargs)
            self.experiment_function = experiment_function

        def run(self, experiment_kwargs):
            return self.experiment_function(self, **experiment_kwargs)

    def __simulator__(self, fitparams, rng, m_list=[0, 0, 0], return_results=False):
        """
        fitparams: from hyperopt

        m_list: variable to store results, from multiprocessing
        """

        ### reset model and set parameters which should not be optimized and parameters which should be optimized
        self.__set_fitting_parameters__(fitparams)

        ### conduct loaded experiment
        experiment_function = self.experiment
        experiment_kwargs = {"population": self.iz}
        reset_function = self.__set_fitting_parameters__
        reset_kwargs = {"fitparams": fitparams}

        exp_obj = self.myexp(
            reset_function=reset_function,
            reset_kwargs=reset_kwargs,
            experiment_function=experiment_function,
        )
        results = exp_obj.run(experiment_kwargs)

        ### compute loss
        all_loss = self.__get_loss__(results, self.results_soll)
        if isinstance(all_loss, list) or isinstance(all_loss, type(np.zeros(1))):
            loss = sum(all_loss)
        else:
            loss = all_loss

        ### "return" loss and other optional things
        m_list[0] = loss
        if return_results:
            m_list[1] = results
            m_list[2] = all_loss

    def __set_fitting_parameters__(self, fitparams):
        """
        self.iz: ANNarchy population name
        fitparams: list with values for fitting parameters
        self.fv_space: hyperopt variable space list
        self.const_params: dictionary with constant variables

        Sets all given parameters for the population self.iz.
        """
        ### reset model to compilation state
        reset()

        ### set fitting parameters
        for idx in range(len(fitparams)):
            setattr(
                get_population(self.iz),
                self.fitting_variables_name_list[idx],
                fitparams[idx],
            )

        ### set constant parameters
        for key, val in self.const_params.items():
            if isinstance(val, str):
                try:
                    ### value is str --> name of variable in fitting parameters
                    setattr(
                        get_population(self.iz),
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
                        setattr(get_population(self.iz), key, self.const_params[val])
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
                setattr(get_population(self.iz), key, val)

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
        syf.create_dir("/".join(sbi_plot_file.split("/")[:-1]))
        plt.savefig(sbi_plot_file)

        return best

    def run(
        self, max_evals, results_file_name="best.npy", sbi_plot_file="posterior.svg"
    ):

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
        self.results = best

        ### SAVE OPTIMIZED PARAMS AND LOSS
        syf.save_data([best], ["parameter_fit/" + results_file_name])
