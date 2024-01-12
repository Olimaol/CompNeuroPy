import os
import traceback
import shutil
from time import time
import pickle
from functools import wraps


def clear_dir(path):
    """
    Deletes all files and subdirectories in the specified folder.

    Args:
        path (str):
            Path to the folder to clear.
    """
    try:
        if not os.path.exists(path):
            print(f"The folder '{path}' does not exist.")
            return

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                print(traceback.format_exc())
                print(f"Failed to delete {file_path}")
    except Exception:
        print(traceback.format_exc())
        print(f"Failed to clear {path}")


def create_dir(path, print_info=False, clear=False):
    """
    Creates a directory.

    Args:
        path (str):
            Path to the directory to create.

        print_info (bool, optional):
            Whether to print information about the directory creation. Default: False.

        clear (bool, optional):
            Whether to clear the directory if it already exists. Default: False.
    """
    try:
        if isinstance(path, str):
            if len(path) > 0:
                os.makedirs(path)
        else:
            print("create_dir, ERROR: path is no str")
    except Exception:
        if os.path.isdir(path):
            if print_info:
                print(path + " already exists")
            if clear:
                ### clear folder
                ### do you really want?
                answer = input(f"Do you really want to clear {path} (y/n):")
                while answer != "y" and answer != "n":
                    print("please enter y or n")
                    answer = input(f"Do you really want to clear {path} (y/n):")
                ### clear or not depending on answer
                if answer == "y":
                    clear_dir(path)
                    if print_info:
                        print(path + " already exists and was cleared.")
                else:
                    if print_info:
                        print(path + " already exists and was not cleared.")
        else:
            print(traceback.format_exc())
            print("could not create " + path + " folder")
            quit()


def save_variables(variable_list: list, name_list: list, path: str | list = "./"):
    """
    Args:
        variable_list (list):
            variables to save
        name_list (list):
            names of the save files of the variables
        path (str or list):
            save path for all variables, or save path for each variable of the
            variable_list. Default: "./"

    Examples:
        ```python
        import numpy as np
        from CompNeuroPy import save_variables, load_variables

        ### create variables
        var1 = np.random.rand(10)
        var2 = np.random.rand(10)

        ### save variables
        save_variables([var1, var2], ["var1_file", "var2_file"], "my_variables_folder")

        ### load variables
        loaded_variables = load_variables(["var1", "var2"], "my_variables_folder")

        ### use loaded variables
        print(loaded_variables["var1_file"])
        print(loaded_variables["var2_file"])
        ```
    """
    for idx in range(len(variable_list)):
        ### set save path
        if isinstance(path, str):
            save_path = path
        else:
            save_path = path[idx]
        if save_path.endswith("/"):
            save_path = save_path[:-1]
        ### set file name
        file_name = f"{name_list[idx]}.pkl"
        ### set variable
        variable = variable_list[idx]
        ### generate save folder
        create_dir(save_path)
        ### Saving a variable to a file
        with open(f"{save_path}/{file_name}", "wb") as file:
            pickle.dump(variable, file)


def load_variables(name_list: list, path: str | list = "./"):
    """
    Args:
        name_list (list):
            names of the save files of the variables
        path (str or list, optional):
            save path for all variables, or save path for each variable of the
            variable_list. Default: "./"

    Returns:
        variable_dict (dict):
            dictionary with the loaded variables, keys are the names of the
            files, values are the loaded variables

    Examples:
        ```python
        import numpy as np
        from CompNeuroPy import save_variables, load_variables

        ### create variables
        var1 = np.random.rand(10)
        var2 = np.random.rand(10)

        ### save variables
        save_variables([var1, var2], ["var1_file", "var2_file"], "my_variables_folder")

        ### load variables
        loaded_variables = load_variables(["var1", "var2"], "my_variables_folder")

        ### use loaded variables
        print(loaded_variables["var1_file"])
        print(loaded_variables["var2_file"])
        ```
    """
    variable_dict = {}
    for idx in range(len(name_list)):
        ### set save path
        if isinstance(path, str):
            save_path = path
        else:
            save_path = path[idx]
        if save_path.endswith("/"):
            save_path = save_path[:-1]
        ### set file name
        file_name = f"{name_list[idx]}.pkl"
        ### Loading the variable from the file
        with open(f"{save_path}/{file_name}", "rb") as file:
            loaded_variable = pickle.load(file)
        ### store variable in variable_dict
        variable_dict[name_list[idx]] = loaded_variable

    return variable_dict


def timing_decorator(threshold=0.1):
    """
    Decorator to measure the execution time of a function.

    Args:
        threshold (float, optional):
            Threshold in seconds. If the execution time of the function is
            larger than this threshold, the execution time is printed. Default: 0.1.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time()
            result = func(*args, **kwargs)
            end_time = time()
            execution_time = end_time - start_time
            if execution_time >= threshold:
                print(f"{func.__name__} took {execution_time:.4f} seconds")
            return result

        return wrapper

    return decorator
