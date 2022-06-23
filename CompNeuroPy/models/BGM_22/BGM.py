import numpy as np
from ANNarchy import Population, Projection, get_population, get_projection
from ANNarchy.core.Random import *
from CompNeuroPy.neuron_models import poisson_neuron_up_down, poisson_neuron, Izhikevich2007_noisy_AMPA, Izhikevich2007_fsi_noisy_AMPA, Izhikevich2003_noisy_AMPA, Izhikevich2003_flexible_noisy_AMPA, integrator_neuron
from CompNeuroPy.synapse_models import factor_synapse
from CompNeuroPy import generate_model
import csv
import os

class BGM(generate_model):
    """
        The basal ganglia model based on the model from Goenner et al. (2021)
    """

    def __init__(self, name='BGM_v0_1', do_create=True, do_compile=True, compile_folder_name='annarchy_BGM_v0_1'):
        """
            runs the standard init but with already predefined model_creation_function and description
            one can still adjust name, do_compile and compile_folder_name
        """
        description = 'The basal ganglia model based on the model from Goenner et al. (2021)'
        self.params = self.__get_params__(name)
        super().__init__(model_creation_function=self.__model_creation_function__, name=name, description=description, do_create=do_create, do_compile=do_compile, compile_folder_name=compile_folder_name)
       
        
    def __model_creation_function__(self):
        
        #######   POPULATIONS   ######
        ### cortex / input populations
        cor_go    = Population(self.params['cor_go__size'],    poisson_neuron_up_down, name="cor_go")
        cor_pause = Population(self.params['cor_pause__size'], poisson_neuron_up_down, name="cor_pause")
        cor_stop  = Population(self.params['cor_stop__size'],  poisson_neuron_up_down, name="cor_stop")
        ### Str Populations
        str_d1  = Population(self.params['str_d1__size'],  Izhikevich2007_noisy_AMPA, name="str_d1")
        str_d2  = Population(self.params['str_d2__size'],  Izhikevich2007_noisy_AMPA, name="str_d2")
        str_fsi = Population(self.params['str_fsi__size'], Izhikevich2007_fsi_noisy_AMPA, name="str_fsi")
        ### BG Populations
        stn       = Population(self.params['stn__size'],       Izhikevich2003_noisy_AMPA, name="stn")
        snr       = Population(self.params['snr__size'],       Izhikevich2003_noisy_AMPA, name="snr")
        gpe_proto = Population(self.params['gpe_proto__size'], Izhikevich2003_flexible_noisy_AMPA, name="gpe_proto")
        gpe_arky  = Population(self.params['gpe_arky__size'],  Izhikevich2003_flexible_noisy_AMPA, name="gpe_arky")
        gpe_cp    = Population(self.params['gpe_cp__size'],    Izhikevich2003_flexible_noisy_AMPA, name="gpe_cp")
        thal      = Population(self.params['thal__size'],      Izhikevich2003_noisy_AMPA, name="thal")
        ### integrator Neurons
        integrator_go   = Population(self.params['integrator_go__size'],   integrator_neuron, stop_condition="decision == -1", name="integrator_go")
        integrator_stop = Population(self.params['integrator_stop__size'], integrator_neuron, stop_condition="decision == -1", name="integrator_stop")
        
        
        ######   PROJECTIONS   ######
        ### cortex go output
        cor_go__str_d1  = Projection(pre=cor_go, post=str_d1,  target='ampa', synapse=factor_synapse, name='cor_go__str_d1')
        cor_go__str_d2  = Projection(pre=cor_go, post=str_d2,  target='ampa', synapse=factor_synapse, name='cor_go__str_d2')
        cor_go__str_fsi = Projection(pre=cor_go, post=str_fsi, target='ampa', synapse=factor_synapse, name='cor_go__str_fsi')
        cor_go__thal    = Projection(pre=cor_go, post=thal,    target='ampa', synapse=factor_synapse, name='cor_go__thal')
        ### cortex stop output
        cor_stop__gpe_arky = Projection(pre=cor_stop, post=gpe_arky, target='ampa', synapse=factor_synapse, name='cor_stop__gpe_arky')
        cor_stop__gpe_cp   = Projection(pre=cor_stop, post=gpe_cp,   target='ampa', synapse=factor_synapse, name='cor_stop__gpe_cp')
        ### cortex pause output
        cor_pause__stn = Projection(pre=cor_pause, post=stn, target='ampa', synapse=factor_synapse, name='cor_pause__stn')
        ### str d1 output
        str_d1__snr    = Projection(pre=str_d1, post=snr,    target='gaba', synapse=factor_synapse, name='str_d1__snr')
        str_d1__gpe_cp = Projection(pre=str_d1, post=gpe_cp, target='gaba', synapse=factor_synapse, name='str_d1__gpe_cp')
        str_d1__str_d1 = Projection(pre=str_d1, post=str_d1, target='gaba', synapse=factor_synapse, name='str_d1__str_d1')
        str_d1__str_d2 = Projection(pre=str_d1, post=str_d2, target='gaba', synapse=factor_synapse, name='str_d1__str_d2')
        ### str d2 output
        str_d2__gpe_proto = Projection(pre=str_d2, post=gpe_proto, target='gaba', synapse=factor_synapse, name='str_d2__gpe_proto')
        str_d2__gpe_arky  = Projection(pre=str_d2, post=gpe_arky,  target='gaba', synapse=factor_synapse, name='str_d2__gpe_arky')
        str_d2__gpe_cp    = Projection(pre=str_d2, post=gpe_cp,    target='gaba', synapse=factor_synapse, name='str_d2__gpe_cp')
        str_d2__str_d1    = Projection(pre=str_d2, post=str_d1,    target='gaba', synapse=factor_synapse, name='str_d2__str_d1')
        str_d2__str_d2    = Projection(pre=str_d2, post=str_d2,    target='gaba', synapse=factor_synapse, name='str_d2__str_d2')
        ### str fsi output
        str_fsi__str_d1  = Projection(pre=str_fsi, post=str_d1,  target='gaba', synapse=factor_synapse, name='str_fsi__str_d1')
        str_fsi__str_d2  = Projection(pre=str_fsi, post=str_d2,  target='gaba', synapse=factor_synapse, name='str_fsi__str_d2')
        str_fsi__str_fsi = Projection(pre=str_fsi, post=str_fsi, target='gaba', synapse=factor_synapse, name='str_fsi__str_fsi')
        ### stn output
        stn__snr       = Projection(pre=stn, post=snr,       target='ampa', synapse=factor_synapse, name='stn__snr')
        stn__gpe_proto = Projection(pre=stn, post=gpe_proto, target='ampa', synapse=factor_synapse, name='stn__gpe_proto')
        stn__gpe_arky  = Projection(pre=stn, post=gpe_arky,  target='ampa', synapse=factor_synapse, name='stn__gpe_arky')
        stn__gpe_cp    = Projection(pre=stn, post=gpe_cp,    target='ampa', synapse=factor_synapse, name='stn__gpe_cp')
        ### gpe proto output
        gpe_proto__stn      = Projection(pre=gpe_proto, post=stn,      target='gaba', synapse=factor_synapse, name='gpe_proto__stn')
        gpe_proto__snr      = Projection(pre=gpe_proto, post=snr,      target='gaba', synapse=factor_synapse, name='gpe_proto__snr')
        gpe_proto__gpe_arky = Projection(pre=gpe_proto, post=gpe_arky, target='gaba', synapse=factor_synapse, name='gpe_proto__gpe_arky')
        gpe_proto__gpe_cp   = Projection(pre=gpe_proto, post=gpe_cp,   target='gaba', synapse=factor_synapse, name='gpe_proto__gpe_cp')
        gpe_proto__str_fsi  = Projection(pre=gpe_proto, post=str_fsi,  target='gaba', synapse=factor_synapse, name='gpe_proto__str_fsi')
        ### gpe arky output
        gpe_arky__str_d1    = Projection(pre=gpe_arky, post=str_d1,    target='gaba', synapse=factor_synapse, name='gpe_arky__str_d1')
        gpe_arky__str_d2    = Projection(pre=gpe_arky, post=str_d2,    target='gaba', synapse=factor_synapse, name='gpe_arky__str_d2')
        gpe_arky__str_fsi   = Projection(pre=gpe_arky, post=str_fsi,   target='gaba', synapse=factor_synapse, name='gpe_arky__str_fsi')
        gpe_arky__gpe_proto = Projection(pre=gpe_arky, post=gpe_proto, target='gaba', synapse=factor_synapse, name='gpe_arky__gpe_proto')
        gpe_arky__gpe_cp    = Projection(pre=gpe_arky, post=gpe_cp,    target='gaba', synapse=factor_synapse, name='gpe_arky__gpe_cp')
        ### gpe cp output
        gpe_cp__str_d1          = Projection(pre=gpe_cp, post=str_d1,           target='gaba', synapse=factor_synapse, name='gpe_cp__str_d1')
        gpe_cp__str_d2          = Projection(pre=gpe_cp, post=str_d2,           target='gaba', synapse=factor_synapse, name='gpe_cp__str_d2')
        gpe_cp__str_fsi         = Projection(pre=gpe_cp, post=str_fsi,          target='gaba', synapse=factor_synapse, name='gpe_cp__str_fsi')
        gpe_cp__gpe_proto       = Projection(pre=gpe_cp, post=gpe_proto,        target='gaba', synapse=factor_synapse, name='gpe_cp__gpe_proto')
        gpe_cp__gpe_arky        = Projection(pre=gpe_cp, post=gpe_arky,         target='gaba', synapse=factor_synapse, name='gpe_cp__gpe_arky')
        gpe_cp__integrator_stop = Projection(pre=gpe_cp, post=integrator_stop,  target='ampa', synapse=factor_synapse, name='gpe_cp__integrator_stop')
        ### snr output
        snr__thal = Projection(pre=snr, post=thal, target='gaba', synapse=factor_synapse, name='snr__thal')
        ### thal output
        thal__integrator_go = Projection(pre=thal, post=integrator_go, target='ampa', synapse=factor_synapse, name='thal__integrator_go')
        thal__str_d1        = Projection(pre=thal, post=str_d1,        target='ampa', synapse=factor_synapse, name='thal__str_d1')
        thal__str_d2        = Projection(pre=thal, post=str_d2,        target='ampa', synapse=factor_synapse, name='thal__str_d2')
        thal__str_fsi       = Projection(pre=thal, post=str_fsi,       target='ampa', synapse=factor_synapse, name='thal__str_fsi')
        

    def create(self, do_compile=True, compile_folder_name=None):
        super().create(do_compile=False, compile_folder_name=compile_folder_name)
        ### after creating the model, self.populations and self.projections are available --> now set the parameters
        ### do not compile during create but after setting parameters --> parameter values are included in compilation state
        self.__set_params__()
        self.__set_noise_values__()
        self.__set_connections__()
        if do_compile:
            self.compile(compile_folder_name)
        
        
    def __set_params__(self):
        """
            sets params of all populations
        """

        ### loop over all params
        for key, val in self.params.items():
        
            ### split param key in pop and param name
            key_split=key.split('__')
            if len(key_split)>=2:
                pop_name = key_split[0]
                param_name = key_split[1]
                
                if param_name.split('_')[-1]=='noise':# for noise params separate function
                    continue
                
                if param_name.split('_')[-1]=='init':
                    param_name='_'.join(param_name.split('_')[:-1])
                
                ### if pop is in network --> set param
                if pop_name in self.populations:
                    setattr(get_population(pop_name), param_name, val)
                    
                    
    def __set_noise_values__(self):
        """
            sets noise params of all populations
        """

        ### loop over all params
        for key, val in self.params.items():
        
            ### split param key in pop and param name
            key_split=key.split('__')
            if len(key_split)>=2:
                pop_name = key_split[0]
                param_name = key_split[1]
                
                ### if pop is in network --> set param
                if pop_name in self.populations and param_name.split('_')[-1]=='noise':
                    if param_name=='mean_rate_noise':
                        mean=val
                        print(pop_name,param_name,val)
                        try:
                            sd=self.params[pop_name+'__rate_sd_noise']
                            get_population(pop_name).rates_noise = np.random.normal(mean, sd, get_population(pop_name).size)
                        except:
                            get_population(pop_name).rates_noise = mean
                    elif param_name=='rate_sd_noise':
                        continue
                    else:
                        setattr(get_population(pop_name), param_name, val)
                
                
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
                connectivity = self.params[proj_name+'__connectivity']
            except:
                print('\nERROR: missing connectivity parameter for',proj_name,'\n',proj_name+'__connectivity', 'needed!\n','parameters id:', self.params['general__id'],'\n')
                quit()
                
            if connectivity=='connect_fixed_number_pre':
                get_projection(proj_name).connect_fixed_number_pre(number=self.params[proj_name+'__nr_con'], weights=eval(str(self.params[proj_name+'__weights'])), delays=eval(str(self.params[proj_name+'__delays'])))
                already_set_params[proj_name] = ['connectivity', 'nr_con', 'weights', 'delays']
            elif connectivity=='connect_all_to_all':
                get_projection(proj_name).connect_all_to_all(weights=eval(str(self.params[proj_name+'__weights'])), delays=eval(str(self.params[proj_name+'__delays'])))
                already_set_params[proj_name] = ['connectivity', 'weights', 'delays']
            elif connectivity=='connect_one_to_one':
                get_projection(proj_name).connect_one_to_one(weights=eval(str(self.params[proj_name+'__weights'])), delays=eval(str(self.params[proj_name+'__delays'])))
                already_set_params[proj_name] = ['connectivity', 'weights', 'delays']
            else:
                print('\nERROR: wrong connectivity parameter for',proj_name+'__connectivity!\n','parameters id:', params['general__id'],'\n')
                quit()
                

        ### set parameters
        ### loop over all params
        for key, val in self.params.items():
        
            ### split param key in proj and param name
            key_split=key.split('__')
            if len(key_split)>=3:
                proj_name = '__'.join(key_split[:2])
                param_name = key_split[2]
                            
                ### if proj is in network --> set param
                if proj_name in self.projections and not(param_name in already_set_params[proj_name]):
                    setattr(get_projection(proj_name), param_name, val)
                    
                    
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
            if 'general__id'==row[0] and True in [name == row[i] for i in range(1,len(row))]:
                idx = [name == row[i] for i in range(1,len(row))].index(True)+1
            elif 'general__id'==row[0]:
                print('No Parameters available for given model name '+name+'! (file '+csvPath+')')
                quit()
        if idx==-1:
            print('No general__id in parameter csv file!')
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
        params['toRGB']                 = {'blue':np.array([3,67,223])/255., 'cyan':np.array([0,255,255])/255., 'gold':np.array([219,180,12])/255., 'orange':np.array([249,115,6])/255., 'red':np.array([229,0,0])/255., 'purple':np.array([126,30,156])/255., 'grey':np.array([146,149,145])/255., 'light brown':np.array([173,129,80])/255., 'lime':np.array([170,255,50])/255., 'green':np.array([21,176,26])/255., 'yellow':np.array([255,255,20])/255., 'lightgrey':np.array([216,220,214])/255.}
        params['Fig7_order']            = ['GPeArky', 'StrD1', 'StrD2', 'STN', 'cortexGo', 'GPeCp', 'GPeProto', 'SNr', 'Thal', 'cortexStop', 'StrFSI']
        params['titles_Code_to_Script'] = {'cortexGo':'cortex-Go', 'cortexStop':'cortex-Stop', 'cortexPause':'cortex-Pause', 'StrD1':'StrD1', 'StrD2':'StrD2', 'StrFSI':'StrFSI', 'GPeProto':'GPe-Proto', 'GPeArky':'GPe-Arky', 'GPeCp':'GPe-Cp', 'STN':'STN', 'SNr':'SNr', 'Thal':'thalamus', 'IntegratorGo':'Integrator-Go', 'IntegratorStop':'Integrator-Stop'}

        
        return params










