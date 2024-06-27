from CompNeuroPy import DeapCma, load_variables, save_variables
import numpy as np
from test2 import deap_opt_transform_path, objective_function, M
import sys

# # Load the variables
# variables = load_variables(
#     name_list=[
#         "n",
#         "p",
#     ],
#     path=deap_opt_transform_path,
# )
# n = variables["n"]
# p = variables["p"]


# def evaluate_function(population):
#     loss_list = []
#     ### the population is a list of individuals which are lists of parameters
#     for individual in population:
#         loss_of_individual = objective_function(
#             mean_shift=individual[0], std_scale=individual[1], n=n, p=p, m=M
#         )
#         loss_list.append((loss_of_individual,))
#     return loss_list


# ### bounds for optimized parameters
# shift_mean_bounds = [-1, 1]
# scale_std_bounds = [0.5, 2]
# lb = np.array([shift_mean_bounds[0], scale_std_bounds[0]])
# ub = np.array([shift_mean_bounds[1], scale_std_bounds[1]])

# ### create an instance of the DeapCma class
# deap_cma = DeapCma(
#     lower=lb,
#     upper=ub,
#     evaluate_function=evaluate_function,
#     param_names=["mean_shift", "std_scale"],
#     hard_bounds=True,
# )

# ### run the optimization
# deap_cma_result = deap_cma.run(max_evals=1000)

# ### get the optimized parameters and best error
# mean_shift = deap_cma_result["mean_shift"]
# std_scale = deap_cma_result["std_scale"]
# error_improved = deap_cma_result["best_fitness"]
mean_shift = 0.0
std_scale = 1.0
error_improved = 0.0

# Save the variables
save_variables(
    name_list=[
        f"mean_shift_{sys.argv[1]}",
        f"std_scale_{sys.argv[1]}",
        f"error_improved_{sys.argv[1]}",
    ],
    variable_list=[mean_shift, std_scale, error_improved],
    path=deap_opt_transform_path,
)
