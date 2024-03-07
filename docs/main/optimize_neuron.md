## Introduction
CompNeuroPy provides the [`OptNeuron`](#CompNeuroPy.opt_neuron.OptNeuron) class which can be used to define your optimization of an ANNarchy neuron model (tuning the parameters). You can either optimize your neuron model to some data or try to reproduce the dynamics of a different neuron model (for example to reduce a more complex model). In both cases, you have to define the experiment which generates the data of interest with your neuron model.

!!! warning
    OptNeuron has to be imported from "CompNeuroPy.opt_neuron" and you have to install torch, sbi, pybads and hyperopt (e.g. pip install torch sbi pybads hyperopt) separately.

Used optimization methods:

- [hyperopt](http://hyperopt.github.io/hyperopt/) (using the [Tree of Parzen Estimators (TPE)](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf))

    * Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. TProc. of the 30th International Conference on Machine Learning (ICML 2013), June 2013, pp. I-115 to I-23. [pdf](http://proceedings.mlr.press/v28/bergstra13.pdf)

- [sbi](https://sbi-dev.github.io/sbi/)

    * Tejero-Cantero et al., (2020). sbi: A toolkit for simulation-based inference. Journal of Open Source Software, 5(52), 2505, [https://doi.org/10.21105/joss.02505](https://doi.org/10.21105/joss.02505)

- [deap](https://github.com/deap/deap) (using the [CMAES](https://deap.readthedocs.io/en/master/api/algo.html#module-deap.cma) strategy)

    * Fortin, F. A., De Rainville, F. M., Gardner, M. A. G., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. The Journal of Machine Learning Research, 13(1), 2171-2175. [pdf](https://www.jmlr.org/papers/volume13/fortin12a/fortin12a.pdf)

- [pybads](https://acerbilab.github.io/pybads/)

    * Singh, G. S., & Acerbi, L. (2023). PyBADS: Fast and robust black-box optimization in Python. arXiv preprint [arXiv:2306.15576](https://arxiv.org/abs/2306.15576).
    * Acerbi, L., & Ma, W. J. (2017). Practical Bayesian optimization for model fitting with Bayesian adaptive direct search. Advances in neural information processing systems, 30. [pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/df0aab058ce179e4f7ab135ed4e641a9-Paper.pdf)

### Example:
```python
opt = OptNeuron(
    experiment=my_exp,
    get_loss_function=get_loss,
    variables_bounds=variables_bounds,
    results_soll=experimental_data["results_soll"],
    time_step=experimental_data["time_step"],
    compile_folder_name="annarchy_opt_neuron_example",
    neuron_model=my_neuron,
    method="hyperopt",
    record=["r"],
)
```

A full example is available in the [Examples](../examples/opt_neuron.md).

## Run the optimization
To run the optimization simply call the _run()_ function of the [`OptNeuron`](#CompNeuroPy.opt_neuron.OptNeuron) object.

## Define the experiment
You have to define a [`CompNeuroExp`](define_experiment.md#CompNeuroPy.experiment.CompNeuroExp) object containing a _run()_ function. In the _run()_ function simulations and recordings are performed.

!!! warning
    While defining the [`CompNeuroExp`](define_experiment.md#CompNeuroPy.experiment.CompNeuroExp) _run()_ function for the optimization with [`OptNeuron`](#CompNeuroPy.opt_neuron.OptNeuron) you must observe the following rules:

- the _run()_ function has to take a single argument (besides self) which contains the name of the population consiting of a single neuron or multiple neurons of the optimized neuron model (you can use this to access the population)
- thus, the simulation has to be compatible with a population consisting of a single or multiple neurons
- use the _self.reset()_ function within the _run()_ function to create new recording chunks and reset the model parameters/variables!
- [`OptNeuron`](#CompNeuroPy.opt_neuron.OptNeuron) automatically sets the parameters/variables defined in the _variables_bounds_ before each run, _self.reset()_ will reset the model to this state (all parameters/variables not defined in _variables_bounds_ are reset to compile state)
- be aware that the target neuron model is always resetted to compile state (this affects results_soll)!
- using _self.reset(parameters=False)_ in the _run()_ function keeps all parameter changes you do during the experiment
- start the monitors before you want to record something by calling _self.monitors.start()_
- the best parameters, the corresponding loss, and the corresponding results of the experiment will be available after the optimization, you can store any additional data which should be available after optimiozation in the _self.data_ attribute
- do not call the functions _store_model_state()_ and _reset_model_state()_ of the [`CompNeuroExp`](define_experiment.md#CompNeuroPy.experiment.CompNeuroExp) class within the _run()_ function!


### Example:
```python
class my_exp(CompNeuroExp):
    """
    Define an experiment by inheriting from CompNeuroExp.

    CompNeuroExp provides the attributes:

        monitors (CompNeuroMonitors):
            a CompNeuroMonitors object to do recordings, define during init otherwise
            None
        data (dict):
            a dictionary for storing any optional data

    and the functions:
        reset():
            resets the model and monitors
        results():
            returns a results object
    """

    def run(self, population_name):
        """
        Do the simulations and recordings.

        To use the CompNeuroExp class, you need to define a run function which
        does the simulations and recordings. The run function should return the
        results object which can be obtained by calling self.results().

        For using the CompNeuroExp for OptNeuron, the run function should have
        one argument which is the name of the population which is automatically created
        by OptNeuron, containing a single neuron of the model which should be optimized.

        Args:
            population_name (str):
                name of the population which contains a single neuron, this will be
                automatically provided by opt_neuron

        Returns:
            results (CompNeuroExp._ResultsCl):
                results object with attributes:
                    recordings (list):
                        list of recordings
                    recording_times (recording_times_cl):
                        recording times object
                    mon_dict (dict):
                        dict of recorded variables of the monitors
                    data (dict):
                        dict with optional data stored during the experiment
        """
        ### you have to start monitors within the run function, otherwise nothing will
        ### be recorded
        self.monitors.start()

        ### run the simulation, if you reset the monitors/model the model_state argument
        ### has to be True (Default)
        ...
        simulate(100)
        self.reset()
        ...

        ### optional: store anything you want in the data dict. For example infomration
        ### about the simulations. This is not used for the optimization but can be
        ### retrieved after the optimization is finished
        self.data["sim"] = sim_step.simulation_info()
        self.data["population_name"] = population_name
        self.data["time_step"] = dt()

        ### return results, use the object's self.results()
        return self.results()
```

## The get_loss_function
The _get_loss_function_ must have two arguments. When this function is called during optimization, the first argument is always the _results_ object returned by the _experiment_, i.e. the results of the neuron you want to optimize. The second argument depends on whether you have specified _results_soll_, i.e. data to be reproduced by the _neuron_model_, or whether you have specified a _target_neuron_model_ whose results are to be reproduced by the _neuron_model_. Thus, the second argument is either _results_soll_ provided to the [`OptNeuron`](#CompNeuroPy.opt_neuron.OptNeuron) class during initialization or another _results_ object (returned by the [`CompNeuroExp`](define_experiment.md#CompNeuroPy.experiment.CompNeuroExp) _run_ function), generated with the _target_neuron_model_.

!!! warning
    You always have to work with the neuron rank 0 within the _get_loss_function_!

### Example:
In this example we assume, that _results_soll_ was provided during initialization of the [`OptNeuron`](#CompNeuroPy.opt_neuron.OptNeuron) class (no _target_neuron_model_ used).
```python
def get_loss(results_ist: CompNeuroExp._ResultsCl, results_soll):
    """
    Function which has to have the arguments results_ist and results_soll and should
    calculates and return the loss. This structure is needed for the OptNeuron class.

    Args:
        results_ist (object):
            the results object returned by the run function of experiment (see above)
        results_soll (any):
            the target data directly provided to OptNeuron during initialization

    Returns:
        loss (float or list of floats):
            the loss
    """
    ### get the recordings and other important things for calculating the loss from
    ### results_ist, we do not use all available information here, but you could
    rec_ist = results_ist.recordings
    pop_ist = results_ist.data["population_name"]
    neuron = 0

    ### get the data for calculating the loss from the results_soll
    r_target_0 = results_soll[0]
    r_target_1 = results_soll[1]

    ### get the data for calculating the loss from the recordings
    r_ist_0 = rec_ist[0][f"{pop_ist};r"][:, neuron]
    r_ist_1 = rec_ist[1][f"{pop_ist};r"][:, neuron]

    ### calculate the loss, e.g. the root mean squared error
    rmse1 = rmse(r_target_0, r_ist_0)
    rmse2 = rmse(r_target_1, r_ist_1)

    ### return the loss, one can return a singel value or a list of values which will
    ### be summed during the optimization
    return [rmse1, rmse2]
```

::: CompNeuroPy.opt_neuron.OptNeuron