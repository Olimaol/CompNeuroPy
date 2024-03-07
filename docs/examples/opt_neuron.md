## Optimize neuron model from data
### Introduction
This example demonstrates how to use the OptNeuron class to fit an ANNarchy neuron
model to some experimental data.


### Code
```python
from CompNeuroPy import CompNeuroExp, CompNeuroSim, current_step, rmse
from CompNeuroPy.opt_neuron import OptNeuron
import numpy as np
from ANNarchy import Neuron, dt

### in this example we want to fit an ANNarchy neuron model to some data (which ca be
### somehow obtained by simulating the neuron and recording variables) for this example,
### we have the following simple neuron model, you must not use the :population flag
### for the parameters!
my_neuron = Neuron(
    parameters="""
        I_app = 0
        a = 0
        b = 0
    """,
    equations="""
        r = a*I_app + b
    """,
)


### Now we need some "experimental data" which will be provided to the OptNeuron class
### with the argument results_soll.
def get_experimental_data():
    """
    Return experimental data.

    Assume we have two recordings of the rate r of a single neuron from two different
    current step experiments. Both have length = 1000 ms and after 500 ms the current is
    changed, thus also the rate.

    Returns:
        return_dict (dict):
            Dictionary with keys "results_soll" and "time_step" and values the
            experimental data and the time step in ms with which the date was obtained,
            respectively.
    """
    r_arr = np.empty((2, 1000))
    ### first recording
    r_arr[0, :500] = 2
    r_arr[0, 500:] = 6
    ### second recording
    r_arr[1, :500] = 2
    r_arr[1, 500:] = 10
    ### time step in ms
    time_step = 1

    return_dict = {"results_soll": r_arr, "time_step": time_step}
    return return_dict


### We know how our experimental data was obtained. This is what we have to define as an
### CompNeuroExp for the OptNeuron class.
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
        by OptNeuron, containing a single or multiple neurons of the neuron model which
        should be optimized. Thus, the run function should be able to run the experiment
        with a single or multiple neurons in the given population!

        Args:
            population_name (str):
                name of the population with neurons of the tuned neuron model, this will
                be automatically provided by OptNeuron

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

        ### do simulations and recordings using the provided CompNeuroMonitors object
        ### (recording the varables specified during the initialization of OptNeuron
        ### class) and e.g. the CompNeuroSim class
        sim_step = CompNeuroSim(
            simulation_function=current_step,
            simulation_kwargs={
                "pop": population_name,
                "t1": 500,
                "t2": 500,
                "a1": 0,
                "a2": 5,
            },
            kwargs_warning=False,
            name="test",
            monitor_object=self.monitors,
        )

        ### run the simulation
        sim_step.run()
        ### Here we reset the model and monitors, creating a new chunk for the next sim
        ### use the self.reset() function to reset the model and monitors!
        ### OptNeuron sets the parameters (defined in the variables_bounds dict) of the
        ### neuron model before each run.
        ### By using the reset function of the CompNeuroExp class you reset the model to
        ### this state (all varaiables/parameters not defined in variable bounds are
        ### reset to compile state)
        self.reset()
        sim_step.run({"a2": 10})

        ### optional: store anything you want in the data dict. For example infomration
        ### about the simulations. This is not used for the optimization but can be
        ### retrieved after the optimization is finished
        self.data["sim"] = sim_step.simulation_info()
        self.data["population_name"] = population_name
        self.data["time_step"] = dt()

        ### return results, use the object's self.results()
        return self.results()


### Next, the OptNeuron class needs a function to calculate the loss.
def get_loss(results_ist: CompNeuroExp._ResultsCl, results_soll):
    """
    Function which has to have the arguments results_ist and results_soll and should
    calculate and return the loss. This structure is needed for the OptNeuron class.

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

    ### the get_loss function should always calculate the loss for neuron rank 0!
    neuron = 0

    ### get the data for calculating the loss from the results_soll
    r_target_0 = results_soll[0]
    r_target_1 = results_soll[1]

    ### get the data for calculating the loss from the recordings of the
    ### optimized neuron model
    r_ist_0 = rec_ist[0][f"{pop_ist};r"][:, neuron]
    r_ist_1 = rec_ist[1][f"{pop_ist};r"][:, neuron]

    ### calculate the loss, e.g. the root mean squared error
    rmse1 = rmse(r_target_0, r_ist_0)
    rmse2 = rmse(r_target_1, r_ist_1)

    ### return the loss, one can return a singel value or a list of values which will
    ### be summed during the optimization
    return [rmse1, rmse2]


### now we need to define which variables should be optimized and between which bounds
variables_bounds = {"a": [-10, 10], "b": [-10, 10]}


def main():
    ### get experimental data
    experimental_data = get_experimental_data()

    ### intitialize optimization
    opt = OptNeuron(
        experiment=my_exp,
        get_loss_function=get_loss,
        variables_bounds=variables_bounds,
        neuron_model=my_neuron,
        results_soll=experimental_data["results_soll"],
        time_step=experimental_data["time_step"],
        compile_folder_name="annarchy_opt_neuron_example_from_data",
        method="deap",
        record=["r"],
    )

    ### run the optimization, define how often the experiment should be repeated
    fit = opt.run(max_evals=100, results_file_name="best_from_data")

    ### print optimized parameters, we should get around a=0.8 and b=2
    print("a", fit["a"])
    print("b", fit["b"])
    print(list(fit.keys()))


if __name__ == "__main__":
    main()
```

### Console Output
```console
$ python run_opt_neuron_from_data.py 
ANNarchy 4.7 (4.7.3b) on linux (posix).
OptNeuron: Initialize OptNeuron... do not create anything with ANNarchy before!
OptNeuron: WARNING: attributes ['I_app', 'r'] are not used/initialized.
checking neuron_models, experiment, get_loss...Done

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 26.66gen/s, best loss: 0.00000]
a 0.8000000007960777
b 1.9999999939091158
['a', 'b', 'logbook', 'deap_pop', 'loss', 'all_loss', 'std', 'results', 'results_soll']
```

## Optimize neuron model from other neuron model
### Introduction
This example demonstrates how to use the OptNeuron class to fit an ANNarchy neuron
model to the dynamics of another ANNarchy neuron model in a specific experiment.

The experiment and variable_bounds used are imported from the other example
[run_opt_neuron_from_data.py](#optimize-neuron-model-from-data).

### Code
```python
from CompNeuroPy import CompNeuroExp, rmse
from CompNeuroPy.opt_neuron import OptNeuron
from ANNarchy import Neuron

### import the experiment and variables_bounds
from run_opt_neuron_from_data import my_exp, variables_bounds
from run_opt_neuron_from_data import my_neuron as simple_neuron


### for this example we want to fit a simple neuron model to replicate the dynamics of a
### more complex neuron model, the simple model is imported from the other example
### 'run_opt_neuron_from_data.py' and the complex model is defined here
complex_neuron = Neuron(
    parameters="""
        I_app = 0
        m0 = 1
        m1 = 2
        m2 = 3
        n0 = 1
        n1 = 0
        n2 = -1
    """,
    equations="""
        r = m0*I_app + n0 + m1*I_app + n1 + m2*I_app + n2
    """,
)


### Next, the OptNeuron class needs a function to calculate the loss.
def get_loss(
    results_ist: CompNeuroExp._ResultsCl, results_soll: CompNeuroExp._ResultsCl
):
    """
    Function which has to have the arguments results_ist and results_soll and should
    calculate and return the loss. This structure is needed for the OptNeuron class.

    Args:
        results_ist (object):
            the results object returned by the run function of experiment (see above),
            conducting the experiment with the optimized neuron model
        results_soll (any):
            the results object returned by the run function of experiment (see above),
            conducting the experiment with the target neuron model

    Returns:
        loss (float or list of floats):
            the loss
    """

    ### get the recordings and other important things from the results_ist (results
    ### generated during the optimization using the defrined CompNeuroExp from above)
    rec_ist = results_ist.recordings
    pop_ist = results_ist.data["population_name"]
    rec_soll = results_soll.recordings
    pop_soll = results_soll.data["population_name"]

    ### the get_loss function should always calculate the loss for neuron rank 0! For
    ### both, the target and the optimized neuron model.
    neuron = 0

    ### get the data for calculating the loss from the recordings of the
    ### target neuron model
    v_soll_0 = rec_soll[0][pop_soll + ";r"][:, neuron]
    v_soll_1 = rec_soll[1][pop_soll + ";r"][:, neuron]

    ### get the data for calculating the loss from the recordings of the
    ### optimized neuron model
    v_ist_0 = rec_ist[0][pop_ist + ";r"][:, neuron]
    v_ist_1 = rec_ist[1][pop_ist + ";r"][:, neuron]

    ### calculate the loss, e.g. the root mean squared error
    rmse1 = rmse(v_soll_0, v_ist_0)
    rmse2 = rmse(v_soll_1, v_ist_1)

    ### return the loss, one can return a singel value or a list of values which will
    ### be summed during the optimization
    return [rmse1, rmse2]


def main():
    ### define optimization
    opt = OptNeuron(
        experiment=my_exp,
        get_loss_function=get_loss,
        variables_bounds=variables_bounds,
        neuron_model=simple_neuron,
        target_neuron_model=complex_neuron,
        time_step=1,
        compile_folder_name="annarchy_opt_neuron_example_from_neuron",
        method="deap",
        record=["r"],
    )

    ### run the optimization, define how often the experiment should be repeated
    fit = opt.run(max_evals=100, results_file_name="best_from_neuron")

    ### print optimized parameters, we should get around a=6 and b=0
    print("a", fit["a"])
    print("b", fit["b"])
    print(list(fit.keys()))


if __name__ == "__main__":
    main()
```

### Console Output
```console
$ python run_opt_neuron_from_neuron.py 
ANNarchy 4.7 (4.7.3b) on linux (posix).
OptNeuron: Initialize OptNeuron... do not create anything with ANNarchy before!
OptNeuron: WARNING: attributes ['I_app', 'r'] are not used/initialized.
checking neuron_models, experiment, get_loss...Done

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:03<00:00, 26.96gen/s, best loss: 0.00000]
a 5.99999999677215
b 2.8379565285300652e-08
['a', 'b', 'logbook', 'deap_pop', 'loss', 'all_loss', 'std', 'results', 'results_soll']
```