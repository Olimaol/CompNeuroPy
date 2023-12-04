import os
import sys
import numpy as np
import marshal
import types
import gc
import traceback
import shutil
from time import time
import pickle


def clear_dir(path):
    """
    Deletes all files and subdirectories in the specified folder
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
    creates a directory path
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


def save_data(data_list, path_list):
    """
    data_list: list of variables (e.g. numpy arrays)

    path_list: save path for each variable of the data_list, everything is saved under ./dataRaw/defined_path
    """
    for idx in range(len(data_list)):
        ### split file path into path and file name
        path = "dataRaw/" + "/".join(path_list[idx].split("/")[:-1]) + "/"
        name = path_list[idx].split("/")[-1]
        ### generate save folder
        create_dir(path)
        ### save date
        np.save(path + name, data_list[idx])


def save_objects(object_list, path_list):
    """
    object_list: list of objects (e.g. objects or functions)

    path_list: save path for each variable of the object_list
    """
    for idx in range(len(object_list)):
        ### split file path into path and file name
        path = "dataRaw/" + "/".join(path_list[idx].split("/")[:-1]) + "/"
        name = path_list[idx].split("/")[-1]
        ### generate save folder
        create_dir(path)
        ### save object
        code = object_list[idx].__code__
        with open(path + name + ".marshal", "wb") as output_file:
            marshal.dump(marshal.dumps(code), output_file)


def save_variables(variable_list: list, name_list: list, path: str | list = "./"):
    """
    Args:
        variable_list: list
            variables to save

        name_list: list
            names of the save files of the variables

        path: str or list, optional, defautl="./"
            save path for all variables, or save path for each variable of the variable_list
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
        name_list: list
            names of the save files of the variables

        path: str or list, optional, defautl="./"
            save path for all variables, or save path for each variable of the variable_list
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


def load_object(file_name):
    """
    file_name: load path with file name of loaded object
    """
    with open(file_name + ".marshal", "rb") as input_file:
        loaded_code = marshal.loads(marshal.load(input_file))
    return types.FunctionType(loaded_code, globals())


def top_ten_memory():
    object_list = get_all_objects()
    nr_objects = len(object_list)
    object_size_arr = np.zeros(nr_objects)
    for idx, obj in enumerate(object_list):
        object_size_arr[idx] = sys.getsizeof(obj)
    total_size = sum(object_size_arr)
    top_ten = np.argsort(object_size_arr)[::-1][: min([10, nr_objects])]
    top_ten_obj = [object_list[idx] for idx in top_ten]
    top_ten_size = object_size_arr[top_ten]
    print("###################", total_size)
    for idx in range(min([10, nr_objects])):
        if top_ten_size[idx] > 3000000 and False:
            print(sys.getrefcount(top_ten_obj[idx]))
            print(gc.get_referrers(top_ten_obj[idx])[1])
            quit()
        else:
            print(
                top_ten_size[idx],
                "\t",
                100 * top_ten_size[idx] / total_size,
                "\t",
                type(top_ten_obj[idx]),
            )
    print("###################")
    del object_list
    del nr_objects
    del object_size_arr
    del top_ten
    del top_ten_obj
    del top_ten_size
    del total_size


def _getr(slist, olist, seen):
    """
    # Recursively expand slist's objects
    # into olist, using seen to track
    # already processed objects.
    """
    for e in slist:
        if id(e) in seen:
            continue
        seen[id(e)] = None
        olist.append(e)
        tl = gc.get_referents(e)
        if tl:
            _getr(tl, olist, seen)


def get_all_objects():
    """Return a list of all live Python
    objects, not including the list itself."""
    gcl = gc.get_objects()
    olist = []
    seen = {}
    # Just in case:
    seen[id(gcl)] = None
    seen[id(olist)] = None
    seen[id(seen)] = None
    # _getr does the real work.
    _getr(gcl, olist, seen)
    return olist


def find_folder_with_prefix(base_path, prefix):
    # List all items (files and directories) in the base_path
    items = os.listdir(base_path)

    # Iterate through the items to find a folder with the specified prefix
    for item in items:
        item_path = os.path.join(base_path, item)

        # Check if the item is a directory and its name starts with the given prefix
        if os.path.isdir(item_path) and item.startswith(prefix):
            return item

    # If no folder with the specified prefix is found, return None
    return None


def timing_decorator(threshold=0.1):
    def decorator(func):
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
