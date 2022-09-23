import CompNeuroPy.neuron_models as nm
import CompNeuroPy.model_functions as mf
import CompNeuroPy.simulation_functions as sif
import CompNeuroPy.system_functions as syf
import CompNeuroPy.extra_functions as ef
from ANNarchy import get_time, dt, reset
import numpy as np


class Monitors:

    
    def __init__(self, monDict={}):
    
        self.mon     = mf.addMonitors(monDict)
        self.monDict = monDict

        timings = {}
        for key, val in monDict.items():
            compartmentType, compartment = key.split(';')
            timings[compartment] = {'currently_paused':True, 'start':[], 'stop':[]}
        self.timings = timings
        
        self.recordings=[]
        self.recording_times=[]
        self.already_got_recordings=False
        self.already_got_recording_times=False
        
    def start(self, compartment_list=None):
        if compartment_list==None:
            compartment_list = list(self.monDict.keys())
        
        self.timings = mf.startMonitors(compartment_list,self.mon,self.timings)
        
    def pause(self, compartment_list=None):
        if compartment_list==None:
            compartment_list = list(self.monDict.keys())
        
        self.timings = mf.pauseMonitors(compartment_list,self.mon,self.timings)
        
    def get_recordings(self, monDict=[], reset_call=False):
        ### only if recordings in current chunk and get_recodings was not already called add current chunk to recordings
        if self.__any_recordings_in_current_chunk__() and self.already_got_recordings==False:
            if isinstance(monDict, list):
                monDict = self.monDict
            ### update recordings
            self.recordings.append(mf.getMonitors(monDict,self.mon))
            ### upade already_got_recordings --> it will not update recordings again
            self.already_got_recordings=True
            
            if not(reset_call):
                if len(self.recordings)==0: print('WARNING get_recordings: no recordings available, empty list returned. Maybe forgot start()?')
            return self.recordings
        else:
            if not(reset_call):
                if len(self.recordings)==0: print('WARNING get_recordings: no recordings available, empty list returned. Maybe forgot start()?')
            return self.recordings
            
        
    def get_recording_times(self, compartment_list=None, reset_call=False):
        if compartment_list==None:
            compartment_list = list(self.monDict.keys())
            
        temp_timings={}
        for key in compartment_list:
            compartmentType, compartment = key.split(';')
            if len(self.timings[compartment]['start']) > len(self.timings[compartment]['stop']):
                ### was started/resumed but never stoped after --> use current time for stop time
                self.timings[compartment]['stop'].append(get_time())
            ### calculate the idx of the recorded arrays which correspond to the timings and remove 'currently_paused'
            diff_timings = (np.array(self.timings[compartment]['stop']) - np.array(self.timings[compartment]['start']))/dt() ## the length of the periods defines the idx of the arrays + dt
            start_idx = [np.sum(diff_timings[0:i]).astype(int) for i in range(diff_timings.size)]
            stop_idx  = [np.sum(diff_timings[0:i+1]).astype(int) for i in range(diff_timings.size)]
            temp_timings[compartment]={'start':{'ms':self.timings[compartment]['start'], 'idx':start_idx}, 'stop':{'ms':self.timings[compartment]['stop'], 'idx':stop_idx}}

        ### only append temp_timings of current chunk if there are recordings in current chunk at all and if get_recordings was not already called (double call would add the same chunk again)
        if self.__any_recordings_in_current_chunk__() and self.already_got_recording_times==False:  self.recording_times.append(temp_timings)
            
        ### upade already_got_recording_times --> it will not update recording_times again
        self.already_got_recording_times=True

        ### generate a object from recording_times and return this instead of the dict
        recording_times_ob = recording_times_cl(self.recording_times)
        
        if not(reset_call):
            if len(self.recording_times)==0: print('WARNING get_recording_times: no recordings available, empty list returned. Maybe forgot start()?')
        return recording_times_ob
        
        
    def __any_recordings_in_current_chunk__(self, compartment_list=None):
        if compartment_list==None:
            compartment_list = list(self.monDict.keys())
            
        temp_timings={}
        for key in compartment_list:
            compartmentType, compartment = key.split(';')
            if len(self.timings[compartment]['start']) > len(self.timings[compartment]['stop']):
                ### was started/resumed but never stoped after --> use current time for stop time
                self.timings[compartment]['stop'].append(get_time())
            ### calculate the idx of the recorded arrays which correspond to the timings and remove 'currently_paused'
            diff_timings = (np.array(self.timings[compartment]['stop']) - np.array(self.timings[compartment]['start']))/dt() ## the length of the periods defines the idx of the arrays + dt
            start_idx = [np.sum(diff_timings[0:i]).astype(int) for i in range(diff_timings.size)]
            stop_idx  = [np.sum(diff_timings[0:i+1]).astype(int) for i in range(diff_timings.size)]
            temp_timings[compartment]={'start':{'ms':self.timings[compartment]['start'], 'idx':start_idx}, 'stop':{'ms':self.timings[compartment]['stop'], 'idx':stop_idx}}
        
        ### generate a temp object of temp timings to check if there were recordings at all
        recording_times_ob_temp = recording_times_cl([temp_timings])
        return recording_times_ob_temp.__any_recordings__(chunk=0)
        
        
    def reset(self, populations=True, projections=False, synapses=False, monitors=True, model=True, net_id=0):
        """
            get recordings before emptiing the monitors by reset
        """
        self.get_recordings(reset_call=True)
        self.get_recording_times(reset_call=True)
        self.already_got_recordings=False ### after reset one can still update recordings
        self.already_got_recording_times=False ### after reset one can still update recording_times
        ### reset timings, after reset, add a zero to start if the monitor is still running (this is not resetted by reset())
        for key in self.timings.keys():
            self.timings[key]['start']=[]
            self.timings[key]['stop']=[]
            if self.timings[key]['currently_paused']==False: self.timings[key]['start'].append(0)
        if model:
            reset(populations, projections, synapses, monitors, net_id=0)
            
    def __current_chunk__(self):
        ### if recordings are currently active --> return chunk in which these recordings will be saved
        ### check if there are currently active recordings
        active_recordings=False
        for key, val in self.monDict.items():
            compartmentType, compartment = key.split(';')
            if not(self.timings[compartment]['currently_paused']):
                ### tere are currently active recordings
                active_recordings = True
        
        if active_recordings:
            current_chunk_idx = len(self.recordings)
            return current_chunk_idx
        else:
            ### if currently no recordings are active return None
            return None
        
        
        
class recording_times_cl:

    def __init__(self, recording_times_list):
        self.recording_times_list = recording_times_list
        
    def time_lims(self, chunk=None, compartment=None, period=None):
        assert len(self.recording_times_list)>0, 'ERROR time_lims(): No recordings/recording_times available.'
        return self.__lims__('ms', chunk, compartment, period)
        
    def idx_lims(self, chunk=None, compartment=None, period=None):
        assert len(self.recording_times_list)>0, 'ERROR idx_lims(): No recordings/recording_times available.'
        return self.__lims__('idx', chunk, compartment, period)
        
    def all(self):
        return self.recording_times_list
        
    def combine_chunks(self, recordings, recording_data_str, mode='sequential'):
        """
            combines the data of all chunks of recordings, only possible if no pauses in between
            returns a single time array (time values in ms) and a single values array (of the recorded variable)
            
            recordings: recordings array of recording chunks
            recording_data_str: str of compartment + recorded variable
            mode: how should the time array be generated
                sequential: each chunk starts at zero e.g.: [0,100] + [0,250] --> [0, 1, ..., 100, 0, 1, ..., 250]
                consecutive: each chunk starts at the last stop time of the previous chunk e.g.: [0,100] + [0,250] --> [0, 1, ..., 100, 101, 102, ..., 350]
        """
        assert len(self.recording_times_list)>0, 'ERROR combine_chunks(): No recordings/recording_times available.'
        
        compartment = recording_data_str.split(';')[0]
        dt = recordings[0]['dt']
        nr_chunks = self.__get_nr_chunks__()
        data_list = []
        time_list = []
        pre_chunk_start_time = 0
        
        ### first test, if there are pauses in between
        for chunk in range(nr_chunks):
            time_lim=self.time_lims(chunk=chunk, compartment=compartment)
            idx_lim=self.idx_lims(chunk=chunk, compartment=compartment)
            if int(np.diff(time_lim)/dt)!=int(np.diff(idx_lim)):
                assert False, 'ERROR combine_chunks, time_lim and idx_lim do not fit! Maybe multiple periods separated by pauses in recordings?'
        
        for chunk in range(nr_chunks):
            ### append data list with data of all periods of this chunk
            data_list.append(recordings[chunk][recording_data_str])
            
            ### nr of periods in this chunk
            nr_periods = self.__get_nr_periods__(chunk, compartment)
            
            ### start time of chunk depends on mode
            if mode == 'sequential':
                chunk_start_time = 0
            elif mode == 'consecutive':
                if chunk == 0:
                    chunk_start_time = 0
                else:
                    last_stop_time = self.recording_times_list[chunk-1][compartment]['stop']['ms'][nr_periods-1]
                    chunk_start_time = pre_chunk_start_time + last_stop_time
                    pre_chunk_start_time = chunk_start_time
            else:
                print('ERROR recording_times.combine_data, Wrong mode.')
                quit()
            
            ### append the time list with all times of the periods
            for period in range(nr_periods):
                start_time = self.recording_times_list[chunk][compartment]['start']['ms'][period] + chunk_start_time
                end_time   = self.recording_times_list[chunk][compartment]['stop']['ms'][period] + chunk_start_time
                times      = np.arange(start_time, end_time, dt)
                time_list.append(times)

        ### flatten the two lists
        data_arr = np.array(ef.flatten_list(data_list))
        time_arr = np.array(ef.flatten_list(time_list))
        
        return [time_arr, data_arr]
        
    def __lims__(self, string, chunk=None, compartment=None, period=None):
        chunk              = self.__check_chunk__(chunk)
        compartment        = self.__check_compartment__(compartment, chunk)
        period_0, period_1 = self.__check_period__(period, chunk, compartment)
        lims = [self.recording_times_list[chunk][compartment]['start'][string][period_0], self.recording_times_list[chunk][compartment]['stop'][string][period_1]]
        return lims
        
    def __check_compartment__(self, compartment, chunk):
        if compartment==None:
            ### by default just use the first compartment
            compartment=list(self.recording_times_list[chunk].keys())[0]
        elif compartment in list(self.recording_times_list[chunk].keys()):
            compartment=compartment
        else:
            print('ERROR recording_times, given compartment "'+str(compartment)+'" not available')
            quit()
            
        return compartment
        
    def __check_period__(self, period, chunk, compartment):
        if period==None:
            ### by default use all periods
            period_0 = 0
            period_1 = len(self.recording_times_list[chunk][compartment]['start']['idx'])-1
        elif period<len(self.recording_times_list[chunk][compartment]['start']['idx']):
            period_0 = period
            period_1 = period
        else:
            print('ERROR recording_times, given period not available')
            quit()
            
        return [period_0, period_1]
        
    def __check_chunk__(self, chunk):
        if chunk==None:
            ### by default use first chunk
            chunk = 0
        elif chunk<self.__get_nr_chunks__():
            chunk = chunk
        else:
            print('ERROR recording_times, given chunk not available')
            quit()
            
        return chunk
        
    def __get_nr_chunks__(self):
        return len(self.recording_times_list)
        
    def __get_nr_periods__(self, chunk, compartment):
        return len(self.recording_times_list[chunk][compartment]['start']['idx'])
        
    def __any_recordings__(self, chunk):
        """
            check all periods and compartments if there are any recordings
            returns True/False if there are any recordings
        """
        compartment_list = list(self.recording_times_list[chunk].keys())
        found_recordings = False
        for compartment in compartment_list:
        
            nr_periods_of_compartment = len(self.recording_times_list[chunk][compartment]['start']['idx'])
            
            for period_idx in range(nr_periods_of_compartment):
                idx_lims = self.idx_lims(chunk=chunk, compartment=compartment, period=period_idx)
                if np.diff(idx_lims)[0]>0: found_recordings = True
                
        return found_recordings
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
