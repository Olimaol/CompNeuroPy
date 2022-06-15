from CompNeuroPy import generate_model as gm
from CompNeuroPy import system_functions as sf
from CompNeuroPy import Experiment
from CompNeuroPy.models import H_and_H_model_Bischop
from ANNarchy import setup, reset, get_population


### setup ANNarchy
setup(dt=0.01)

### create and compile model
model = H_and_H_model_Bischop()
population = model.populations[0]

### define experiment
class myexp(Experiment):

    def run(self, population):
        ### CompNeuroPy is provided locally
        cnp=self.cnp
        
        ### define recordings
        self.mon = cnp.Monitors({'pop;'+population:['v','spike']})
        
        ### define the simulation
        my_sim = cnp.generate_simulation(cnp.simulation_functions.current_step,
                                         simulation_kwargs={'pop':population, 't1':500, 't2':500, 'a1':0, 'a2':100},
                                         name='current_step',
                                         description='simulate two input phases with different durations and inputs for a specified population',
                                         requirements=[{'req':cnp.simulation_requirements.req_pop_attr, 'pop':'simulation_kwargs.pop', 'attr':'I_app'}],
                                         kwargs_warning=False)

        ### simulate the model, reset between separate simulations
        self.mon.start()
        my_sim.run()
        self.mon.pause()
        
        ### reset the model between different stimulation protocols
        ### ATTENTION:
        ### mon.reset is necessary to call before other reset, so that the monitor recordings are saved from being emptied
        ### here it does not reset the model, because this is done by the other reset function
        ### self.reset_function(**self.reset_kwargs) replaces the classical ANNarchy reset() in the Experiment Object
        ### this makes it possible to save and load the run() function but use a different reset function (load the run function and use it in a differently initialized Experiment object --> self.reset can be something different)
        ### this is usefull for optimizations, so that one can reset the model (as here, with the standard reset) and additionally maintain things which should not be resetted, e.g. optimization parameters which were defined after compile
        self.mon.reset(model=False)
        self.reset_function(**self.reset_kwargs)
        self.mon.start()
        my_sim.run({'t1':500, 't2':0, 'a1':0, 'a2':0})
        my_sim.run({'t1':500, 't2':500, 'a1':-10, 'a2':0})
        
        ### SIMULATION END
        
        
        ### store the simulation protocols for return results
        self.sim.append(my_sim)

        ### get recordings
        self.recordings=self.mon.get_recordings()
        
        ### dict for some additional data , here recording times
        self.data['recording_times']=self.mon.get_recording_times()
              
        ### return results
        return self.results()

results = myexp().run(population)


### save
sf.save_data([results],['generate_H_and_H_data/results.npy'])
sf.save_objects([myexp.run],['generate_H_and_H_data/experiment'])
