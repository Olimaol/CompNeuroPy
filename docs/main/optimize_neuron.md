## Introduction
CompNeuroPy provides the [`OptNeuron`](#CompNeuroPy.opt_neuron.OptNeuron) class which can be used to define your optimization of an ANNarchy neuron model (tuning the parameters). You can either optimize your neuron model to some data or try to reproduce the dynamics of a different neuron model (for example to reduce a more complex model). In both cases, you have to define the experiment which generates the data of interest with your neuron model.

!!! warning
    OptNeuron has to be imported from "CompNeuroPy.opt_neuron" and you have to install torch, sbi and hyperopt (e.g. pip install torch sbi hyperopt)

Used optimization methods:

- [hyperopt](http://hyperopt.github.io/hyperopt/)

    Bergstra, J., Yamins, D., Cox, D. D. (2013) Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures. TProc. of the 30th International Conference on Machine Learning (ICML 2013), June 2013, pp. I-115 to I-23.

- [sbi](https://sbi-dev.github.io/sbi/)

    Tejero-Cantero et al., (2020). sbi: A toolkit for simulation-based inference. Journal of Open Source Software, 5(52), 2505, [https://doi.org/10.21105/joss.02505](https://doi.org/10.21105/joss.02505)

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

- the _run()_ function has to take a single argument (besides self) which contains the name of the population consiting of a single neuron of the optimized neuron model (you can use this to access the population)
- call _self.reset(parameters=False)_ at the beginning of the run function, thus the neuron will be in its compile state (except the paramters) at the beginning of each simulation run
- always set _parameters=False_ while calling the _self.reset()_ function (otherwise the parameter optimization will not work)
- besides the optimized parameters and the loss, the results of the experiment (using the optimized parameters) will be available after the optimization, you can store any additional data in the _self.data_ attribute


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
        ### For OptNeuron you have to reset the model and monitors at the beginning of
        ### the run function! Do not reset the parameters, otherwise the optimization
        ### will not work!
        self.reset(parameters=False)

        ### you have to start monitors within the run function, otherwise nothing will
        ### be recorded
        self.monitors.start()

        ### run the simulation, remember setting parameters=False in the reset function!
        ...
        simulate(100)
        self.reset(parameters=False)
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