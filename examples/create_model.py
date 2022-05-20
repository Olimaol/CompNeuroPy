from CompNeuroPy.model_functions import get_full_model
from ANNarchy import Population, Neuron
from CompNeuroPy import generate_model

### Create a neuron model (there are also predefined neuron models provided in ANNarchy and CompNeuroPy)
poisson_neuron = Neuron(        
    parameters ="""
        rates   = 0
    """,
    equations ="""
        p       = Uniform(0.0, 1.0) * 1000.0 / dt      
    """,
    spike ="""    
        p <= rates
    """,    
    reset ="""    
        p = 0.0
    """,
    name = "poisson_neuron",
    description = "Poisson neuron whose rate can be specified and is reached instanteneous."
)


### define model in function
def two_poisson(params=None, a=0, b=0, c=0):
    """
        generates two poisson neuron populations
        params : parameter dictionary
            's1'/'s2' : sizes of pop1/pop2
            'n1'/'n2' : names of pop1/pop2    
        a/b/c : parameters for demonstration purposes only      
    """
    ### check if params dictionary is available
    if params==None:
        assert False, 'ERROR: parameter dicitonary missing for "two_poisson"'
    ### create two populations
    Population(params['s1'], neuron=poisson_neuron, name=params['n1'])
    Population(params['s2'], neuron=poisson_neuron, name=params['n2'])
    
    print('created model, other parameters:',a,b,c)
    
    
### Let's initialize a first model
### define the parameters argument of the model creation function
params = {'s1':1, 's2':1, 'n1':'first_poisson', 'n2':'second_poisson'}
### use generate_model to initialize the model
### this is outside of __name__=='__main__' to demonstrate that already initialized models can be imported and created in other scripts (see example/run_and_monitor_simulations.py)
my_model = generate_model(model_creation_function=two_poisson,          ### the most important part, this funciton creates the model (populations, projections)
                          model_kwargs={'params':params, 'b':1, 'c':2}, ### if the model_creation_function uses arguments they can be specified in this dictionary
                          name='my_model',                              ### you can give the model a name
                          description='my simple Poisson neuron model', ### you can give the model a description
                          do_create=False,                              ### this model is not created yet
                          do_compile=False,                             ### there is no compilation yet (would not compile if True because do_create==False)
                          compile_folder_name='annarchy_my_model')      ### if do_create and do_compile would be True it would directly compile using this compile_folder
                          
### this initialized the first model
### we could now create and compile it, but we will do this inside __name__=='__main__'
### thus, we can import models in other scripts without automatically creating/compiling them here (see example/run_and_monitor_simulations.py)

### Let's create a second model inside __name__=='__main__'
### creating further models is only possible as long as no compilation has taken place
if __name__=='__main__':
    ### again define parameters, do use unique names for populations/projections!
    params_2nd = {'s1':1, 's2':1, 'n1':'pop1', 'n2':'pop2'}
    ### this time the model is directly created but not compiled
    my_2nd_model = generate_model(model_creation_function=two_poisson, model_kwargs={'params':params_2nd}, do_compile=False)
    
    ### now also create our first model and then compile all created models
    ### the create function will automatically start the compilation unless one sets do_compile=False
    my_model.create()
    
    ### we initialized, created and compiled 2 models... we could now run simulations and monitor variables

    ### print some information about the models
    print('\n{:<45} {:<45}'.format('first model','second model'))
    print('{:<45} {:<45}'.format(my_model.name,my_2nd_model.name))                         ### names (if no name was given --> 'model{+number}')
    print('{:<45} {:<45}'.format(str(my_model.populations),str(my_2nd_model.populations))) ### created populations
    print('{:<45} {:<45}'.format(str(my_model.projections),str(my_2nd_model.projections))) ### created projections
    print('{:<45} {:<45}'.format(my_model.description,my_2nd_model.description))           ### descriptions
    print('\n',get_full_model())                                                           ### all created populations and projections
    
    
    
    ### console output of this file:
    """
    created model, other parameters: 0 0 0
    created model, other parameters: 0 1 2
    Compiling ...  OK 

    first model                                   second model                                 
    my_model                                      model2                                       
    ['first_poisson', 'second_poisson']           ['pop1', 'pop2']                             
    []                                            []                                           
    my simple Poisson neuron model                                                             

     {'populations': ['pop1', 'pop2', 'first_poisson', 'second_poisson'], 'projections': []}
    """
    
