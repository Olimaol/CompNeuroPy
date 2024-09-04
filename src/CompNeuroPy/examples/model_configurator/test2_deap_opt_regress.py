from CompNeuroPy import DeapCma, load_variables, save_variables
import numpy as np
from test2 import (
    regression_func,
    OPTIMIZE_FOLDER,
    N_PARAMS_REGRESS,
    REGRESS_FOLDER,
    preprocess_for_regress,
)
import sys


def regression_evaluate_function(population, X, z, weights):
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = regression_objective_function(
            individual=individual, X=X, z=z, weights=weights
        )
        loss_list.append((loss_of_individual,))
    return loss_list


def regression_objective_function(individual, X, z, weights):
    is_data = regression_func(X=X, denormalize=None, args=individual)
    target_data = z
    error_arr = (is_data - target_data) ** 2
    ### weight the error array
    error_arr = error_arr * weights
    return np.sum(error_arr)


if __name__ == "__main__":
    ### Load the p, n variables for regression from the
    ### previous optimization
    loaded_variables = load_variables(
        name_list=[
            "p_list",
            "n_list",
        ],
        path=OPTIMIZE_FOLDER,
    )
    p_arr = np.array(loaded_variables["p_list"])
    n_arr = np.array(loaded_variables["n_list"])

    ### Load mean_shift and std_scale values and the weight_error array to weight the
    ### regression errors
    loaded_variables = load_variables(
        name_list=[
            "weight_error_arr",
            "mean_shift_opt_arr_for_regress",
            "std_scale_opt_arr_for_regress",
        ],
        path=REGRESS_FOLDER,
    )
    weight_error_arr = np.array(loaded_variables["weight_error_arr"])
    mean_shift_opt_arr = np.array(loaded_variables["mean_shift_opt_arr_for_regress"])
    std_scale_opt_arr = np.array(loaded_variables["std_scale_opt_arr_for_regress"])

    ### normalize the data before regression
    n_arr = preprocess_for_regress(var_value=n_arr, var_name="n")
    p_arr = preprocess_for_regress(var_value=p_arr, var_name="p")
    mean_shift_opt_arr = preprocess_for_regress(
        var_value=mean_shift_opt_arr, var_name="mean_shift"
    )
    std_scale_opt_arr = preprocess_for_regress(
        var_value=std_scale_opt_arr, var_name="std_scale"
    )

    ### optimize regression of mean_shift
    deap_cma = DeapCma(
        lower=np.array([-1] * N_PARAMS_REGRESS),
        upper=np.array([1] * N_PARAMS_REGRESS),
        evaluate_function=lambda population: regression_evaluate_function(
            population=population,
            X=(n_arr, p_arr),
            z=mean_shift_opt_arr,
            weights=weight_error_arr,
        ),
        hard_bounds=False,
        display_progress_bar=False,
    )
    deap_cma_result = deap_cma.run(max_evals=2000)
    popt = [deap_cma_result[f"param{param_id}"] for param_id in range(N_PARAMS_REGRESS)]

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
    deap_cma = DeapCma(
        lower=np.array([-1] * N_PARAMS_REGRESS),
        upper=np.array([1] * N_PARAMS_REGRESS),
        evaluate_function=lambda population: regression_evaluate_function(
            population=population,
            X=(n_arr, p_arr),
            z=std_scale_opt_arr,
            weights=weight_error_arr,
        ),
        hard_bounds=False,
        display_progress_bar=False,
    )
    deap_cma_result = deap_cma.run(max_evals=2000)
    popt = [deap_cma_result[f"param{param_id}"] for param_id in range(N_PARAMS_REGRESS)]

    # Save the variables
    save_variables(
        name_list=[
            f"popt_std_scale_{sys.argv[1]}",
            f"best_fitness_std_scale_{sys.argv[1]}",
        ],
        variable_list=[popt, deap_cma_result["best_fitness"]],
        path=REGRESS_FOLDER,
    )
