import os
import sys
import numpy as np
import marshal, types
import gc

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


def top_ten_memory():
    object_list=get_all_objects()
    nr_objects = len(object_list)
    object_size_arr=np.zeros(nr_objects)
    for idx, obj in enumerate(object_list):
        object_size_arr[idx]=sys.getsizeof(obj)
    total_size = sum(object_size_arr)
    top_ten=np.argsort(object_size_arr)[::-1][:min([10, nr_objects])]
    top_ten_obj=[object_list[idx] for idx in top_ten]
    top_ten_size=object_size_arr[top_ten]
    print('###################', total_size)
    for idx in range(min([10, nr_objects])):
        if top_ten_size[idx]>3000000 and False:
            print(sys.getrefcount(top_ten_obj[idx]))
            print(gc.get_referrers(top_ten_obj[idx])[1])
            quit()
        else:
            print(top_ten_size[idx],'\t',100*top_ten_size[idx]/total_size, '\t',type(top_ten_obj[idx]))
    print('###################')
    del(object_list)
    del(nr_objects)
    del(object_size_arr)
    del(top_ten)
    del(top_ten_obj)
    del(top_ten_size)    
    del(total_size)    
    

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










