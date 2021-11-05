import os
import numpy as np

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
