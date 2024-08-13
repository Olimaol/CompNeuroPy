from CompNeuroPy import DeapCma, load_variables, save_variables
import numpy as np
from test2 import (
    regression_func,
    OPTIMIZE_FOLDER,
    N_PARAMS_REGRESS,
    REGRESS_FOLDER,
)
import sys


def regression_evaluate_function(population, X, z):
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = regression_objective_function(
            individual=individual, X=X, z=z
        )
        loss_list.append((loss_of_individual,))
    return loss_list


def regression_objective_function(individual, X, z):
    is_data = regression_func(X, *individual)
    target_data = z
    return np.sum((is_data - target_data) ** 2)


if __name__ == "__main__":
    # Load the variables for regression from the previous optimization
    loaded_variables = load_variables(
        name_list=[
            "p_list",
            "n_list",
            "mean_shift_opt_list",
            "std_scale_opt_list",
        ],
        path=OPTIMIZE_FOLDER,
    )
    p_list = loaded_variables["p_list"]
    n_list = loaded_variables["n_list"]
    mean_shift_opt_list = loaded_variables["mean_shift_opt_list"]
    std_scale_opt_list = loaded_variables["std_scale_opt_list"]

    ### optimize regression of mean_shift
    # TODO do some transofrmations
    deap_cma = DeapCma(
        lower=np.array([-1] * len(N_PARAMS_REGRESS)),
        upper=np.array([1] * len(N_PARAMS_REGRESS)),
        evaluate_function=lambda population: regression_evaluate_function(
            population=population, X=(n_list, p_list), z=mean_shift_opt_list
        ),
        hard_bounds=False,
        display_progress_bar=False,
    )
    deap_cma_result = deap_cma.run(max_evals=2000)
    popt = [deap_cma_result[param_name] for param_name in deap_cma.param_names]

    # Save the variables
    save_variables(
        name_list=[
            f"popt_mean_shift_{sys.argv[1]}",
            f"best_fitness_mean_shift_{sys.argv[1]}",
        ],
        variable_list=[popt, deap_cma_result["best_fitness"]],
        path=REGRESS_FOLDER,
    )

    ### optimize regression of std_scale
    # TODO do some transofrmations
    deap_cma = DeapCma(
        lower=np.array([-1] * len(N_PARAMS_REGRESS)),
        upper=np.array([1] * len(N_PARAMS_REGRESS)),
        evaluate_function=lambda population: regression_evaluate_function(
            population=population, X=(n_list, p_list), z=std_scale_opt_list
        ),
        hard_bounds=False,
        display_progress_bar=False,
    )
    deap_cma_result = deap_cma.run(max_evals=2000)
    popt = [deap_cma_result[param_name] for param_name in deap_cma.param_names]

    # Save the variables
    save_variables(
        name_list=[
            f"popt_std_scale_{sys.argv[1]}",
            f"best_fitness_std_scale_{sys.argv[1]}",
        ],
        variable_list=[popt, deap_cma_result["best_fitness"]],
        path=REGRESS_FOLDER,
    )
