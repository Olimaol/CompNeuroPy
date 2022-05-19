from CompNeuroPy.model_functions import get_full_model
from CompNeuroPy.model_functions import compile_in_folder

class generate_model:

    def __init__(self, name='model', do_compile=False, compile_folder_name='annarchy', description='', model_creation_function=None, model_kwargs=None):
        self.name=name
        self.description=description
        
        initial_existing_model = get_full_model()
        
        if model_creation_function!=None:
        
            ### create model populaitons and projections
            if model_kwargs != None:
                model_creation_function(**model_kwargs)
            else:
                model_creation_function()
                
            ### check which populations and projections have been added
            post_existing_model = get_full_model()
            ### save only added not all projections/populations
            for initial_pop in initial_existing_model['populations']:
                post_existing_model['populations'].remove(initial_pop)
            for initial_proj in initial_existing_model['projections']:
                post_existing_model['projections'].remove(initial_proj)
            self.populations = post_existing_model['populations']
            self.projections = post_existing_model['projections']
        
            ### optionally compile model
            if do_compile:
                compile_in_folder(compile_folder_name)
                
    def compile(self, compile_folder_name=None):
        if compile_folder_name==None:
            compile_folder_name=self.compile_folder_name
        compile_in_folder(compile_folder_name)
                
        
