## Introduction
This example demonstrates how to use the CompNeuroModel class to create and compile
models. It is shown how to define a model creation function, how to initialize, create,
compile a model and how to get information about the model.

The model "my_model" is imported in other examples [run_and_monitor_simulations.py](./run_and_monitor_simulations.md).

## Code
```python
from ANNarchy import Population
from CompNeuroPy import CompNeuroModel
from CompNeuroPy.neuron_models import PoissonNeuron
from tabulate import tabulate


### define model_creation_function
def two_poisson(params, a):
    """
    Generates two Poisson neuron populations.

    Args:
        params (dict):
            Dictionary containing some paramters for the model with following keys:
                's1'/'s2' : sizes of pop1/pop2
                'n1'/'n2' : names of pop1/pop2
        a (int):
            Unused parameter for demonstration purposes only.
    """
    ### create two populations
    Population(params["s1"], neuron=PoissonNeuron, name=params["n1"])
    Population(params["s2"], neuron=PoissonNeuron, name=params["n2"])
    ### print unused parameter
    print(f"created model, other parameters: {a}")


### Let's initialize a first model
### define the parameters argument of the model creation function
params = {"s1": 3, "s2": 3, "n1": "first_poisson", "n2": "second_poisson"}

### use CompNeuroModel to initialize the model, not create or compile it yet
my_model = CompNeuroModel(
    model_creation_function=two_poisson,
    model_kwargs={
        "params": params,
        "a": 1,
    },
    name="my_model",
    description="my simple Poisson neuron model",
    do_create=False,
    do_compile=False,
    compile_folder_name="annarchy_my_model",
)

### this initialized the first model
### we could now create and compile it, but we will do this inside main
### it could also be imported in other scripts and then created/compiled there


def main():
    ### initialize a second model
    ### this time directly create it, but not compile it yet, models can only be created
    ### if not compiled yet
    params = {"s1": 1, "s2": 1, "n1": "pop1", "n2": "pop2"}
    my_model2 = CompNeuroModel(
        model_creation_function=two_poisson,
        model_kwargs={"params": params, "a": 2},
        do_compile=False,
    )

    ### now create also first model, and compile everything (automatically since we did
    ### not set do_compile=False)
    my_model.create()

    ### print some name, description, populations and projections of the models in
    ### tabular form
    models_data = [
        [
            my_model.name,
            my_model.description,
            my_model.populations,
            my_model.projections,
        ],
        [
            my_model2.name,
            my_model2.description,
            my_model2.populations,
            my_model2.projections,
        ],
    ]
    headers = ["Model", "Description", "Populations", "Projections"]
    print(tabulate(models_data, headers, tablefmt="grid"))

    return 1


if __name__ == "__main__":
    main()

```

## Console Output
```console
$ python create_model.py 
ANNarchy 4.7 (4.7.3b) on linux (posix).
created model, other parameters: 2
created model, other parameters: 1
Compiling ...  OK 
+----------+--------------------------------+-------------------------------------+---------------+
| Model    | Description                    | Populations                         | Projections   |
+==========+================================+=====================================+===============+
| my_model | my simple Poisson neuron model | ['first_poisson', 'second_poisson'] | []            |
+----------+--------------------------------+-------------------------------------+---------------+
| model1   |                                | ['pop1', 'pop2']                    | []            |
+----------+--------------------------------+-------------------------------------+---------------+
```