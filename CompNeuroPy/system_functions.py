import os
import numpy as np
import marshal, types

def create_dir(path, print_info=0):
    """
        creates a directory path
    """
    try:
        os.makedirs(path)
    except:
        if os.path.isdir(path):
            if print_info:
                print(path+' already exists')
        else:
            print('could not create '+path+' folder')
            quit()
            
            
def save_data(data_list, path_list):
    """
        data_list: list of variables (e.g. numpy arrays)
        
        path_list: save path for each variable of the data_list
    """
    for idx in range(len(data_list)):
        ### split file path into path and file name
        path = 'dataRaw/'+'/'.join(path_list[idx].split('/')[:-1])+'/'
        name = path_list[idx].split('/')[-1]
        ### generate save folder
        create_dir(path)
        ### save date
        np.save(path+name, data_list[idx])
              
        
def save_objects(object_list, path_list):
    """
        object_list: list of objects (e.g. objects or functions)
        
        path_list: save path for each variable of the object_list
    """
    for idx in range(len(object_list)):
        ### split file path into path and file name
        path = 'dataRaw/'+'/'.join(path_list[idx].split('/')[:-1])+'/'
        name = path_list[idx].split('/')[-1]
        ### generate save folder
        create_dir(path)
        ### save object
        code=object_list[idx].__code__
        with open(path+name+'.marshal', 'wb') as output_file:
            marshal.dump(marshal.dumps(code), output_file)


def load_object(file_name):
    """
        file_name: load path with file name of loaded object
    """
    with open(file_name+'.marshal', 'rb') as input_file:
        loaded_code = marshal.loads(marshal.load(input_file))
    return types.FunctionType(loaded_code, globals())













