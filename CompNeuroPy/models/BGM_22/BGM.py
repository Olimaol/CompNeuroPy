import numpy as np
from ANNarchy import get_population, get_projection
from ANNarchy.core.Random import *
from CompNeuroPy import generate_model
import csv
import os
import importlib

class BGM(generate_model):
    """
        The basal ganglia model based on the model from Goenner et al. (2021)
    """

    def __init__(self, name='BGM_v01_p01', do_create=True, do_compile=True, compile_folder_name=None, seed=None):
        """
            runs the standard init but with already predefined model_creation_function and description
            one can still adjust name, do_compile and compile_folder_name
            
            seed: int, default=None, the seed for the random number generator used during model creation
        """
        ### check if name is correct
        self._name_split_ = name.split('_')
        assert len(self._name_split_)==3 and self._name_split_[0]=='BGM' and self._name_split_[1][0]=='v' and self._name_split_[2][0]=='p', 'ERROR generate_model BGM: "name" must have form "BGM_vXX_pXX"'
        self._model_version_name_ = '_'.join(self._name_split_[:2])
        
        ### init default compile_folder_name
        if compile_folder_name==None: compile_folder_name = 'annarchy_'+self._model_version_name_
        
        ### set description
        description = 'The basal ganglia model based on the model from Goenner et al. (2021)'
        
        ### init random number generator
        self.rng = np.random.default_rng(seed)
        
        ### get model parameters before init
        self.params = self.__get_params__(name)
        
        ### init
        super().__init__(model_creation_function=self.__model_creation_function__, name=name, description=description, do_create=do_create, do_compile=do_compile, compile_folder_name=compile_folder_name)
       
        
    def __model_creation_function__(self):
        model_creation_function = eval("importlib.import_module('CompNeuroPy.models.BGM_22.model_creation_functions')."+self._model_version_name_)
        model_creation_function(self)
        

    def create(self, do_compile=True, compile_folder_name=None):
        super().create(do_compile=False, compile_folder_name=compile_folder_name)
        ### after creating the model, self.populations and self.projections are available --> now set the parameters
        ### do not compile during create but after setting parameters --> parameter values are included in compilation state
        self.__set_params__()
        self.__set_noise_values__()
        self.__set_connections__()
        if do_compile:
            super().compile(compile_folder_name)
        
        
    def __set_params__(self):
        """
            sets params of all populations
        """

        ### loop over all params
        for key, param_val in self.params.items():
            ### split key in param object and param name
            param_object = key.split('.')[0]
            param_name = key.split('.')[1]
            
            ### if param is a noise param --> skip (separate function)
            if param_name.split('_')[-1]=='noise': continue
            
            ### if param name ends with init --> actual param_name (in pop) is without init
            if param_name.split('_')[-1]=='init': param_name='_'.join(param_name.split('_')[:-1])
            
            ### if param_object is a pop in network
            if param_object in self.populations:
                ### and the param_name is an attribute of the pop --> set param of pop
                if param_name in vars(get_population(param_object))['attributes']:
                    setattr(get_population(param_object), param_name, param_val)
                    
                    
    def __set_noise_values__(self):
        """
            sets noise params of all populations
        """

        ### loop over all params
        for key, param_val in self.params.items():
            ### split key in param object and param name
            param_object = key.split('.')[0]
            param_name = key.split('.')[1]

            ### if param_object is a pop in network and param_name ends with noise --> set noise param of pop
            if param_object in self.populations and param_name.split('_')[-1]=='noise':
                if param_name=='mean_rate_noise':
                    ### for mean and sd the actual parameter of the pop has to be calculated
                    mean=param_val
                    try:
                        ### noise values defined by mean and sd
                        sd=self.params[param_object+'.rate_sd_noise']
                        get_population(param_object).rates_noise = self.rng.normal(mean, sd, get_population(param_object).size)
                    except:
                        ### if only mean is available, only set mean
                        get_population(param_object).rates_noise = mean
                elif param_name in vars(get_population(param_object))['attributes']:
                    ### noise parameters which are actual attributes of the pop are simply set
                    setattr(get_population(param_object), param_name, param_val)
                else:
                    continue
                
                
    def __set_connections__(self):
        """
            sets the connectivity and parameters of all projections
        """
        
        already_set_params = {}# dict for each projection, which params were already set during connectivity definition
        
        ### set connectivity
        ### loop over all projections
        for proj_name in self.projections:
            ### get the type of connectivity for projection
            try:
                connectivity = self.params[proj_name+'.connectivity']
            except:
                print('\nERROR: missing connectivity parameter for',proj_name,'\n',proj_name+'.connectivity', 'needed!\n','parameters id:', self.params['general.id'],'\n')
                quit()
                
            if connectivity=='connect_fixed_number_pre':
                get_projection(proj_name).connect_fixed_number_pre(number=self.params[proj_name+'.nr_con'], weights=eval(str(self.params[proj_name+'.weights'])), delays=eval(str(self.params[proj_name+'.delays'])))
                already_set_params[proj_name] = ['connectivity', 'nr_con', 'weights', 'delays']
            elif connectivity=='connect_all_to_all':
                get_projection(proj_name).connect_all_to_all(weights=eval(str(self.params[proj_name+'.weights'])), delays=eval(str(self.params[proj_name+'.delays'])))
                already_set_params[proj_name] = ['connectivity', 'weights', 'delays']
            elif connectivity=='connect_one_to_one':
                get_projection(proj_name).connect_one_to_one(weights=eval(str(self.params[proj_name+'.weights'])), delays=eval(str(self.params[proj_name+'.delays'])))
                already_set_params[proj_name] = ['connectivity', 'weights', 'delays']
            else:
                print('\nERROR: wrong connectivity parameter for',proj_name+'.connectivity!\n','parameters id:', params['general.id'],'\n')
                quit()
                

        ### set parameters
        ### loop over all params
        for key, param_val in self.params.items():
            ### split key in param object and param name
            param_object = key.split('.')[0]
            param_name = key.split('.')[1]
                    
            ### if param_object is proj in network and param not already used and param is an attribute of proj --> set param of proj
            if param_object in self.projections and not(param_name in already_set_params[param_object]) and param_name in vars(get_projection(param_object))['attributes']:
                setattr(get_projection(param_object), param_name, param_val)
                    
                    
    def __get_params__(self, name):
        """
        read all parameters for specified model name

        name : str
            specifies which column in the csv file is used
        """
        

        csvPath = os.path.dirname(os.path.realpath(__file__))+'/parameters.csv'
        csvfile = open(csvPath, newline='')

        params = {}
        reader = csv.reader(csvfile, delimiter=',')
        fileRows = []
        idx = -1
        ### check if name is in the .csv file
        for row in reader:
            if row[0]=='': continue
            fileRows.append(row)
            if 'general.id'==row[0] and True in [name == row[i] for i in range(1,len(row))]:
                idx = [name == row[i] for i in range(1,len(row))].index(True)+1
            elif 'general.id'==row[0]:
                print('No Parameters available for given model name '+name+'! (file '+csvPath+')')
                quit()
        if idx==-1:
            print('No general.id in parameter csv file!')
            quit()
        ### read the column corresponding to name
        for row in fileRows:
            if '###' in row[0]: continue
            if row[idx]=='': continue
            
            value=row[idx]
            try:
                ### if float(value) works value is a number --> check if it is int
                if float(value)-int(float(value))==0:
                    params[row[0]] = int(float(value))
                else:
                    params[row[0]] = float(value)
            except:
                ### value is a string
                if value[0]=='$' and value[-1]=='$':
                    ### value is a formula
                    params[row[0]] = float(eval(value[1:-1]))
                else:
                    ### value is some other string
                    params[row[0]] = value

        csvfile.close()
        
        ### ADD additional params
        params['extra.toRGB']                 = {'blue':np.array([3,67,223])/255., 'cyan':np.array([0,255,255])/255., 'gold':np.array([219,180,12])/255., 'orange':np.array([249,115,6])/255., 'red':np.array([229,0,0])/255., 'purple':np.array([126,30,156])/255., 'grey':np.array([146,149,145])/255., 'light brown':np.array([173,129,80])/255., 'lime':np.array([170,255,50])/255., 'green':np.array([21,176,26])/255., 'yellow':np.array([255,255,20])/255., 'lightgrey':np.array([216,220,214])/255.}
        params['extra.Fig7_order']            = ['GPeArky', 'StrD1', 'StrD2', 'STN', 'cortexGo', 'GPeCp', 'GPeProto', 'SNr', 'Thal', 'cortexStop', 'StrFSI']
        params['extra.titles_Code_to_Script'] = {'cortexGo':'cortex-Go', 'cortexStop':'cortex-Stop', 'cortexPause':'cortex-Pause', 'StrD1':'StrD1', 'StrD2':'StrD2', 'StrFSI':'StrFSI', 'GPeProto':'GPe-Proto', 'GPeArky':'GPe-Arky', 'GPeCp':'GPe-Cp', 'STN':'STN', 'SNr':'SNr', 'Thal':'thalamus', 'IntegratorGo':'Integrator-Go', 'IntegratorStop':'Integrator-Stop'}

        
        return params










