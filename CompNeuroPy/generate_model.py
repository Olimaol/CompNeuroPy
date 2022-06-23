from CompNeuroPy.model_functions import get_full_model, compile_in_folder
from gc import get_objects

class generate_model:

    def __init__(self, model_creation_function, model_kwargs=None, name='model', description='', do_create=True, do_compile=True, compile_folder_name='annarchy'):
        self.name = name
        if name=='model': self.name = name+str(self.__nr_models__())
        self.description = description
        self.model_creation_function = model_creation_function
        self.compile_folder_name = compile_folder_name
        self.model_kwargs = model_kwargs
        self.populations=[]
        self.projections=[]
        self.created = False
        if do_create:
            self.create(do_compile=do_compile, compile_folder_name=compile_folder_name)
                
    def compile(self, compile_folder_name=None):
        """
            compiles a created model
        """
        ### check if this model is created
        if self.created:
            if compile_folder_name==None:
                compile_folder_name=self.compile_folder_name
                
            ### check if other models were initialized but not created --> warn that they are not compiled
            not_created_model_list = self.__check_if_models_created__()
            if len(not_created_model_list)>0:
                print('\nWARNING during compile of model '+self.name+': There are initialized models which are not created, thus not compiled! models:\n'+'\n'.join(not_created_model_list)+'\n')
            compile_in_folder(compile_folder_name)
        else:
            print('\n')
            assert False, 'ERROR during compile of model '+self.name+': Only compile the model after it has been created!'
        
    def create(self, do_compile=True, compile_folder_name=None):
        """
            creates a model and optionally compiles it directly
        """
        if self.created:
            print('model',self.name,'already created!')
        else:
            initial_existing_model = get_full_model()
            ### create model populations and projections
            if self.model_kwargs!=None:
                self.model_creation_function(**self.model_kwargs)
            else:
                self.model_creation_function()
            self.created = True
                
            ### check which populations and projections have been added
            post_existing_model = get_full_model()
            ### save only added not all projections/populations
            for initial_pop in initial_existing_model['populations']:
                post_existing_model['populations'].remove(initial_pop)
            for initial_proj in initial_existing_model['projections']:
                post_existing_model['projections'].remove(initial_proj)
            self.populations = post_existing_model['populations']
            self.projections = post_existing_model['projections']
                
            if do_compile:
                self.compile(compile_folder_name)
            
    def __check_if_models_created__(self):
        """
            checks which CompNeuroPy models are created
            returns a list with all initialized CompNeuroPy models which are not created yet
        """
        
        not_created_model_list = []
        object_list=get_objects()
        for obj in object_list:
            test=str(obj)
            compare='<CompNeuroPy.generate_model.generate_model object'
            if len(test)>=len(compare):
                if compare == test[:len(compare)]:
                    if vars(obj)['created']==False:
                        not_created_model_list.append(vars(obj)['name'])
        del(object_list)
        return not_created_model_list
        
    def __nr_models__(self):
        """
            returns the current number of initialized (not considering "created") CompNeuroPy models
        """
        
        model_list = []
        object_list=get_objects()
        for obj in object_list:
            test=str(obj)
            compare='<CompNeuroPy.generate_model.generate_model object'
            if len(test)>=len(compare):
                if compare == test[:len(compare)]:
                    model_list.append(vars(obj)['name'])
        del(object_list)    
        return len(model_list)
            
