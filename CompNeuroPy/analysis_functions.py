import numpy as np
import pylab as plt
from ANNarchy import raster_plot
import warnings

def get_nanmean(a, axis=None, dtype=None):
    """
        np.nanmean without warnings
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret=np.nanmean(a, axis=axis, dtype=dtype)
    return ret
    
def get_nanstd(a, axis=None, dtype=None):
    """
        np.nanstd without warnings
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ret=np.nanstd(a, axis=axis, dtype=dtype)
    return ret

def hanning_split_overlap(seq, size, overlap):
    """
        splits a sequence (array) in as many hanning-windowed subsequences as possible
    
        seq: full original sequence
        size: size of subsequences to be generated from the full sequence, subsequences are hanning-windowed
        overlap: overlap of subsequences
    
        iterates over complete size of subsequences
        --> = start positions
        from startpositions iterate to reach end of fullsequence with stepsize subsequencesize-overlap
        --> for each start position n sequence positions are obtained = indizes of seq for building the subsequences
            first row:  [0, 0+(subsequencesize-overlap), 0+2*(subsequencesize-overlap), ...]
            second row: [1, 1+(subsequencesize-overlap), 1+2*(subsequencesize-overlap), ...]
        --> similar to matrix, columns = indizes for building the subsequences
    """
    #seq[i::stepsize] == seq[range(i,seq.size,stepsize)]
    return np.array([x*np.hanning(size) for x in zip(*[seq[i::size-overlap] for i in range(size)])])
    
    
def get_population_power_spectrum(spikes,presimulationTime,simulationTime,simulation_dt, spikesSamplingfrequency=250, fftSize=1024):
    """
        generates power spectrum of population spikes
        using the Welch methode (Welch,1967)
        
        spikes: ANNarchy spike dict of one population
        presimulationTime: simulation time which will not be analyzed
        simulationTime: analyzed simulation time
        
        spikesSamplingfrequency: to sample the spike times, in Hz --> max frequency = samplingfrequency / 2
        fftSize: signal size for FFT, duration (in s) = fftSize / samplingfrequency --> frequency resolution = samplingfrequency / fftSize
    """
    populationSize = len(list(spikes.keys()))
    
    if (simulationTime / 1000) < (fftSize / spikesSamplingfrequency):
        print('Simulation time has to be >=',fftSize / spikesSamplingfrequency,'s for FFT!')
        return [np.zeros(int(fftSize/2-2)),np.zeros(int(fftSize/2-2))]
    else:
        spektrum=np.zeros((populationSize,fftSize))
        for neuron in range(populationSize):
            ### sampling steps array
            spiketrain=np.zeros(int((simulationTime/1000.)*spikesSamplingfrequency))
            ### spike times as sampling steps
            idx=(((np.array(spikes[neuron])*simulation_dt)/1000.)*spikesSamplingfrequency).astype(np.int32)
            ### cut the spikes from presimulation
            idx=idx-(presimulationTime/1000.)*spikesSamplingfrequency
            idx=idx[idx>=0].astype(np.int32)
            ### sampling steps array, one if there was a spike at sampling step = spike train
            spiketrain[idx]=1
            
            ###generate multiple overlapping sequences out of the spike trains  
            spiketrainSequences=hanning_split_overlap(spiketrain,fftSize,int(fftSize/2))
        
            ###generate power spectrum
            spektrum[neuron]=get_nanmean(  np.abs(np.fft.fft(spiketrainSequences))**2  ,0)

        ###mean spectrum over all neurons
        spektrum=get_nanmean(spektrum,0)
        
        frequenzen=np.fft.fftfreq(fftSize,1./spikesSamplingfrequency)
        
        return [frequenzen[2:int(fftSize/2)],spektrum[2:int(fftSize/2)]]
        
        
def get_power_spektrum_from_time_array(arr,presimulationTime,simulationTime,simulation_dt, samplingfrequency=250, fftSize=1024):
    """
        generates power spectrum of time signal (returns [frequencies,power])
        using the Welch methode (Welch,1967)
        
        arr: time array, value for each timestep
        presimulationTime: simulation time which will not be analyzed
        simulationTime: analyzed simulation time
        
        samplingfrequency: to sample the arr, in Hz --> max frequency = samplingfrequency / 2
        fftSize: signal size for FFT, duration (in s) = fftSize / samplingfrequency --> frequency resolution = samplingfrequency / fftSize
    """
    
    if (simulationTime / 1000) < (fftSize / samplingfrequency):
        print('Simulation time has to be >=',fftSize / samplingfrequency,'s for FFT!')
        return [np.zeros(int(fftSize/2-2)),np.zeros(int(fftSize/2-2))]
    else:
        ### sampling steps array
        sampling_arr=arr[0::int((1/samplingfrequency)*1000/simulation_dt)]
        
        ###generate multiple overlapping sequences
        sampling_arr_sequences=hanning_split_overlap(sampling_arr,fftSize,int(fftSize/2))
    
        ###generate power spectrum
        spektrum=get_nanmean(  np.abs(np.fft.fft(sampling_arr_sequences))**2  ,0)
        
        frequenzen=np.fft.fftfreq(fftSize,1./samplingfrequency)
        
        return [frequenzen[2:int(fftSize/2)],spektrum[2:int(fftSize/2)]]
        
        
def get_pop_rate(spikes,duration,dt=1,t_start=0,t_smooth_ms=-1):#TODO: maybe makes errors with few/no spikes... check this
    """
        spikes: spikes dictionary from ANNarchy
        duration: duration of period after (optional) initial period (t_start) from which rate is calculated in ms
        dt: timestep of simulation
        t_start: starting simulation time, from which rates should be calculated
        t_smooth_ms: time window size for rate calculation in ms, optional, standard = -1 which means automatic window size
        
        returns smoothed population rate from period after rampUp period until duration
    """
    temp_duration=duration+t_start
    t,n = raster_plot(spikes)
    if len(t)>1:#check if there are spikes in population at all
        if t_smooth_ms==-1:
            ISIs = []
            minTime=np.inf
            duration=0
            for idx,key in enumerate(spikes.keys()):
                times = np.array(spikes[key]).astype(int)
                if len(times)>1:#check if there are spikes in neuron
                    ISIs += (np.diff(times)*dt).tolist()#ms
                    minTime=np.min([minTime,times.min()])
                    duration=np.max([duration,times.max()])
                else:# if there is only 1 spike set ISI to 10ms
                    ISIs+=[10]
            t_smooth_ms = np.min([(duration-minTime)/2.*dt,np.mean(np.array(ISIs))*10+10])

        rate=np.zeros((len(list(spikes.keys())),int(temp_duration/dt)))
        rate[:]=np.NaN
        binSize=int(t_smooth_ms/dt)
        bins=np.arange(0,int(temp_duration/dt)+binSize,binSize)
        binsCenters=bins[:-1]+binSize//2
        for idx,key in enumerate(spikes.keys()):
            times = np.array(spikes[key]).astype(int)
            for timeshift in np.arange(-binSize//2,binSize//2+binSize//10,binSize//10).astype(int):
                hist,edges=np.histogram(times,bins+timeshift)
                rate[idx,np.clip(binsCenters+timeshift,0,rate.shape[1]-1)]=hist/(t_smooth_ms/1000.)

        poprate=get_nanmean(rate,0)
        timesteps=np.arange(0,int(temp_duration/dt),1).astype(int)
        time=timesteps[np.logical_not(np.isnan(poprate))]
        poprate=poprate[np.logical_not(np.isnan(poprate))]
        poprate=np.interp(timesteps, time, poprate)

        ret = poprate[int(t_start/dt):]
    else:
        ret = np.zeros(int(duration/dt))

    return ret
    
def plot_recordings(figname, recordings, start_time, end_time, shape, plan, dpi=300):
    """
        recordings: dict of recordings
        shape: tuple, shape of subplots
        plan: list of strings, strings defin where to plot which data and how
    """
    
    plt.figure(figsize=([6.4*shape[1], 4.8*shape[0]]))
    for subplot in plan:
        try:
            nr, part, variable, mode = subplot.split(';')
            nr=int(nr)
            style=''
        except:
            try:
                nr, part, variable, mode, style = subplot.split(';')
                nr=int(nr)
            except:
                print('\nERROR plot_recordings: for each subplot give plan-string as: "nr;part;variable;mode" or "nr;part;variable;mode;style" if style is available!\n')
                quit()
        try:
            data=recordings[';'.join([part,variable])]
        except:
            print('\nERROR plot_recordings: data',';'.join([part,variable]),'not in recordings\n')
            quit()

        
        times=np.arange(start_time,end_time,recordings['dt'])
            
        plt.subplot(shape[0],shape[1],nr)
        if variable=='spike' and mode=='raster':
            t,n=raster_plot(data)
            if style!='':
                plt.scatter(t,n,s=0.4, color=style)
            else:
                plt.scatter(t,n,s=0.4, color='k')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.ylabel('# neurons')
            plt.title('Spikes '+part)
        elif variable=='spike' and mode=='mean':
            firing_rate = get_pop_rate(data,end_time-start_time,dt=recordings['dt'],t_start=start_time)
            plt.plot(times,firing_rate, color='k')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.ylabel('Mean firing rate [Hz]')
            plt.title('Mean firing rate '+part)
        elif variable=='spike' and mode=='hybrid':
            t,n=raster_plot(data)
            plt.plot(t,n,'k.')
            plt.ylabel('# neurons')
            ax=plt.gca().twinx()
            firing_rate = get_pop_rate(data,end_time-start_time,dt=recordings['dt'],t_start=start_time)
            ax.plot(times,firing_rate, color='r')
            plt.ylabel('Mean firing rate [Hz]', color='r')
            ax.tick_params(axis='y', colors='r')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.title('Activity '+part)
        elif variable!='spike' and mode=='line':
            if len(data.shape)==1:
                plt.plot(times,data, color='k')
            elif len(data.shape)==2:
                for idx in range(data.shape[1]):
                    plt.plot(times,data[:,idx], color='k')
            else:
                print('\nERROR plot_recordings: only data of 1D neuron populations accepted,',';'.join([part,variable]),'\n')
            plt.xlim(start_time, end_time)
            plt.xlabel('time [ms]')
            plt.title('Variable '+part+' '+variable)
        elif variable!='spike' and mode=='mean':
            a=0
        else:
            print('\nERROR plot_recordings: mode',mode,'not available for variable',variable,'\n')
    plt.tight_layout()
    plt.savefig(figname, dpi=dpi)
