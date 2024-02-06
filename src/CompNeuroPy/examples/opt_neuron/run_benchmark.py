from CompNeuroPy import run_script_parallel, load_variables, print_df
from itertools import product
import pandas as pd
import numpy as np

from CompNeuroPy.statistic_functions import anova_between_groups

MAX_EVALS = 1000
N_PER_COND = 500
PARALLEL_WORKERS = 24
DO_OPT = False
DO_STAT = True


def FILE_APPENDIX(method, features, sim_id):
    return f"_{method}_{features}_{sim_id}"


if __name__ == "__main__":
    ### define the arguments for the script
    method_list = ["hyperopt", "deap", "bads"]
    features_list = ["voltage", "efel"]
    args_list = [
        [str(sim_id), method, features]
        for sim_id, method, features in product(
            range(N_PER_COND), method_list, features_list
        )
    ]
    if DO_OPT:
        ### run the script in parallel
        run_script_parallel(
            "benchmark_izhikevich.py",
            PARALLEL_WORKERS,
            args_list,
        )

    if DO_STAT:
        ### load the benchmark variables TODO
        benchmark_dict = {
            "features": [],
            "method": [],
            "best_loss": [],
            "best_loss_time": [],
            "best_loss_evals": [],
            "parameter_difference": [],
        }
        loss_history_array = np.empty(
            ((len(method_list) * len(features_list)), N_PER_COND), dtype=object
        )
        for sim_id, method, features in args_list:
            ### get file appendix
            file_appendix = FILE_APPENDIX(method, features, sim_id)
            ### load variables
            variables_dict = load_variables(
                name_list=[f"benchmark_variables{file_appendix}"],
                path="benchmark_izhikevich_results/",
            )[f"benchmark_variables{file_appendix}"]
            ### add variables to banchmark dict
            benchmark_dict["features"].append(features)
            benchmark_dict["method"].append(method)
            benchmark_dict["best_loss"].append(variables_dict["best_loss"])
            benchmark_dict["best_loss_time"].append(variables_dict["best_loss_time"])
            benchmark_dict["best_loss_evals"].append(variables_dict["best_loss_evals"])
            benchmark_dict["parameter_difference"].append(
                variables_dict["parameter_difference"]
            )
            ### add loss history to array
            first_idx = int(
                round(
                    method_list.index(method) * len(features_list)
                    + features_list.index(features)
                )
            )
            second_idx = int(sim_id)
            loss_history_array[first_idx, second_idx] = variables_dict["loss_history"]

        ### find largest time in loss history TODO
        max_time = 0
        for condition in range(len(method_list) * len(features_list)):
            for sim_id in range(N_PER_COND):
                max_time = max(max_time, loss_history_array[condition, sim_id][-1, 1])

        ### create dataframe
        df = pd.DataFrame(benchmark_dict)
        df = df.sort_values(by=["features", "method"])
        ### for every features and method combination only keep the 10 best results
        df = df.groupby(["features", "method"]).apply(
            lambda x: x.nsmallest(10, "best_loss")
        )
        df = df.reset_index(drop=True)
        ### do the statistics
        anova_between_groups(df, "best_loss", ["features", "method"])

        ### create array for loss history
        loss_history_array = np.array(loss_history_array)
        ### average the loss history over all simulations of the same method and features
        loss_history_array = np.mean(loss_history_array, axis=0)
        print(loss_history_array.shape)
