import os
import pandas as pd
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns
from dlab.sglx_analysis import readMeta
from dlab import sorting_quality as sq
from matplotlib.gridspec import GridSpec

"""
Use of all functions in this file are based on a somewhat standardized data format unique to Juan Santiago's analysis pipeline. Data for 
matrices are stored in a dictionary containg array data (frames) and frame times. Data for gratings stimuli are stored in a large 
compiled dataframe. Imported dlab functions are sourced from the denmanlab Github repository
"""

#generates spike triggered average data used to approximate linear receptive fields into a numpy array format
#TODO add more flexibility for tau values. Currently only able to adjust number but not range
def rf_array(spike_data, probe, unit, stim_data, stim_times,num_taus=10):
    probe_spikes = spike_data[spike_data['probe']==probe]
    spiketimes = np.array(probe_spikes['times'][unit])
    taus = np.round(np.linspace(-0.01,0.28,num_taus),2)
    srf_list=[]
    
    for k,tau in tqdm(enumerate(taus)):
        idx_arr = []
        stimspikes = []
        spiketime_adj = spiketimes-tau
        for i in range(spiketimes.shape[0]):
            bool_idx = np.logical_and(spiketime_adj[i] > stim_times[0] , spiketime_adj[i] < stim_times[-1])
            idx_arr.append(bool_idx)
        spike_adj_stim = spiketime_adj[idx_arr]

        frame_index = []
        for j in range(spike_adj_stim.shape[0]):
            stim_frame = np.min(np.where(stim_times>spike_adj_stim[j]))-1
            frame_index.append(stim_frame)    

        srf_frames = stim_data[frame_index]
        srf = srf_frames.mean(axis=(0))
        srf_list.append(srf)

    output = dict(zip(taus,srf_list))
    return(output)

#Takes the array output from rf_array and puts out a nicely formatted grid of imshow heatmaps
def rf_grid(data, colormap ='viridis',interp='none',title='',nrows=5,ncols=2):
    fig,axs = plt.subplots(ncols, nrows,figsize=(10,10))
    axs = axs.ravel()
    taus = np.round(np.linspace(-0.01,0.28,len(data)),2)
    for i,j in tqdm(enumerate(taus)):
        data2 = data[j]
        out_mean = np.mean(data2)
        out_std = np.std(data2)
        col_min = out_mean-(out_std*3)
        col_max = out_mean+(out_std*3)
        #Generate subplots
        axs[i].imshow(data2, clim=(col_min,col_max),cmap=colormap, 
                      interpolation=interp
                     )
        axs[i].set_title(j)
        for ax in fig.get_axes():
            ax.set_frame_on(False);
            ax.set_xticklabels('',visible=False);
            ax.set_xticks([]);
            ax.set_yticklabels('',visible=False);
            ax.set_yticks([])
            ax.set_aspect(1.0)
            ax.set_xlim(0,64)
            ax.set_ylim(0,64)
        plt.tight_layout()
        plt.suptitle(title,y=0.75)
    return(fig)

#takes spike data and triggers and yields overlaid psth ribbon plots with variance
#TODO add colormap options
def psth_line_overlay(spike_data, probe, unit, stim_data, condition, title='', pre=0.5, post=2.5,binsize=0.05,variance=True):
    spike_data = spike_data[spike_data['probe']==probe]
    times = np.array(spike_data.times[unit])
    numbins = int((post+pre)/binsize)
    conds = np.unique(stim_data[condition])
    num_conds = len(conds)
    x = np.arange(-pre,post,binsize)
    colors = plt.cm.viridis(np.linspace(0,1,num_conds))
    
    psth_all=[]
    fig,ax = plt.subplots()
    
    for i,cond in enumerate(np.unique(stim_data[condition])):
        triggers = np.array(stim_data['times'][stim_data[condition] == cond])
        bytrial = np.zeros((len(triggers),numbins-1))
        for j, trigger in enumerate(triggers):
            trial = triggers[j]
            start = trial-pre
            end = trial+post
            bins_ = np.arange(start,end,binsize)
            trial_spikes = times[np.logical_and(times>=start, times<=end)]
            hist,edges = np.histogram(trial_spikes,bins=bins_)
            if len(hist)==numbins-1:
                bytrial[j]=hist
            elif len(hist)==numbins:
                bytrial[j]=hist[:-1]
        psth = np.mean(bytrial,axis=0)/binsize
        if isinstance(conds[i],float)==True:
            ax.plot(x[:-1],psth, color=colors[i], label=str(round(conds[i],2)))
#         if isinstance(conds[i],tuple)==True:
#             ax.plot(x[:-1],psth, color=colors[i], label=str(round(conds[i],2)))
        else:
            ax.plot(x[:-1],psth, color=colors[i], label=str(conds[i]))
        if variance == True:
            var = np.std(bytrial,axis=0)/binsize/np.sqrt((len(triggers)))
            upper = psth+var
            lower = psth-var
            ax.fill_between(x[:-1],upper,psth,alpha=0.1,color=colors[i])
            ax.fill_between(x[:-1],lower,psth,alpha=0.1,color=colors[i])
    ax.axvline(0,linestyle='dashed')
    plt.legend(loc=(1.05,0.48))
    plt.title(title)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.draw()
    return(fig)

#similar to psth_line, this function yields psth plots for gratings stimuli as thin bars with color intensiity indicating neuron response 
#in Hz. These plots, while superior in organization and readability provide only a weak indication of baseline firing rate and lack display
#of variance
#TODO Add colormap options
def psth_bars(spike_data,probe,unit,stim_data,condition,title='',pre=0.5,post=2.5,binsize=0.05):
    spike_data = spike_data[spike_data['probe']==probe]
    times = np.array(spike_data.times[unit])
    numbins = int((post+pre)/binsize)
    num_conds = len(np.unique(stim_data[condition]))
    
    psth_all=[]
    fig,ax = plt.subplots(num_conds,1)
    
    for i,cond in enumerate(np.unique(stim_data[condition])):
        triggers = np.array(stim_data['times'][stim_data[condition] == cond])
        bytrial = np.zeros((len(triggers),numbins-1))
        for j, trigger in enumerate(triggers):
            trial = triggers[j]
            start = trial-pre
            end = trial+post
            bins_ = np.arange(start,end,binsize)
            trial_spikes = times[np.logical_and(times>=start, times<=end)]
            hist,edges = np.histogram(trial_spikes,bins=bins_)
            if len(hist)==numbins-1:
                bytrial[j]=hist
            elif len(hist)==numbins:
                bytrial[j]=hist[:-1]
        psth = np.mean(bytrial,axis=0)/binsize
        psth = np.reshape(psth,(1,len(psth)))
        psth_all.append(psth)
        im = ax[i].imshow(psth,aspect=3, vmin=0,vmax=np.max(psth_all),interpolation='gaussian')
        if isinstance(cond,float)==True:
            ax[i].set_ylabel(str(round(cond,2)),rotation=0,labelpad=20,fontsize=12,va='center')
        #elif isinstance(cond,tuple)==True:
            ax[i].set_ylabel(str(cond),rotation=0,labelpad=32,fontsize=10,va='center')
        else:
            ax[i].set_ylabel(str(cond),rotation=0,labelpad=20,fontsize=12,va='center')
        ax[i].set_yticks([])
        ax[i].axvline(pre/binsize,color='r',linewidth=3)
        ax[i].set_xticks([])

    fig.subplots_adjust(hspace=0,wspace=0)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.03, 0.7])
    fig.suptitle(title, fontsize=15)
    fig.colorbar(im,cax=cbar_ax)
    cbar_ax.set_title('Hz')
    plt.show()
    return(fig)


#For a given neuron in a given dataset, this function yields plots for all stimuli responses of interest. Because of how matrix data/times
#are structured, the inputs for those arguments must be structured as a list. 
#e.g. mtx_data = [matrix_data['luminance_data'],matrix_data['green_data'],matrix_data['uv_data']]
#     mtx_times = [matrix_data['luminance_times'],matrix_data['color_times'],matrix_data['color_times']]
#
#TODO Adjust color options across functions to allow for unified color palette
#TODO Add flexibility to the folder/directory creation portion so the function doesn't fail when the folder exists
#TODO Add flexibility to stimulus inputs since sometimes stimulus fails during experiment
#TODO Add ability to use alternative save locations
def single_cell_summary(path,probe,spike_data,unit,matrix_data,matrix_times,gratings_data,
                        gratings_conditions = ['orientation','color','green','uv'],psth = 'lines',colors='viridis',save=False
                        ):
    
    raw_data = glob(path+'*'+str(probe)+'\*ap.bin')[0]
    imec_meta = readMeta(glob(path+'*'+str(probe))[0])
    sampRate = float(imec_meta['imSampRate']) #get sampling rate (Hz)
    if os.path.isdir(os.path.join(path,'Figures'))==False:
        os.mkdir(os.path.join(path,'Figures'))
    probe_folder = os.mkdir(os.path.join(path,'Figures',probe))
    image_folder = os.mkdir(os.path.join(path,'Figures',probe,'unit'+str(unit)))
    image_path = os.path.join(path,'Figures',probe,'unit'+str(unit))
    
    #Separate units by probe
    units_df = spike_data.loc[spike_data['probe']==probe]
    
    #generate waveform figure
    mean_wf = sq.mean_waveform(rawdata=raw_data,times=units_df.times[unit],channels=385,sampling_rate=float(sampRate))
    for i in mean_wf:
        plt.plot(i)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks([])
        plt.savefig(os.path.join(image_path,'unit'+str(unit)+'_waveform.svg'))
        
    #generate receptive field grid figures
    lum_rfs = rf_array(units_df, probe, unit, matrix_data[0], matrix_times[0])
    green_rfs = rf_array(units_df, probe, unit, matrix_data[1], matrix_times[1])
    uv_rfs = rf_array(units_df, probe, unit, matrix_data[2], matrix_times[2])
    
    rf_outputs = [lum_rfs,green_rfs, uv_rfs]
    rf_names = ['Luminance','Green','UV']
    
    for num,j in enumerate(rf_outputs):
        grid_fig = rf_grid(j,title=rf_names[num])
        grid_fig.savefig(os.path.join(image_path,'unit'+str(unit)+str(num)+'grid.svg'))
        
    color_condition = []
    for x in range(gratings_data.shape[0]):
        b = round(gratings_data['uv'][x],2)
        g = round(gratings_data['green'][x],2)
        condition = tuple([g, b])
        color_condition.append(condition)
        
    gratings_data['color']=color_condition
    
    for k in gratings_conditions:
        gratings_df = gratings_data[gratings_data['condition']==k]
        gratings_df = gratings_df.drop(columns=['index']).reset_index()

        if k == 'color':
            if psth == 'lines':
                #Generate PSTH for evoked response to green, uv, and combined color condition   
                psth_color = psth_line_overlay(spike_data=units_df,unit=unit,stim_data=gratings_df,condition='color',
                                               title = 'Color Condition')
                psth_green = psth_line_overlay(spike_data=units_df,unit=unit,stim_data=gratings_df,condition='green',
                                  title = 'Green')
                psth_uv = psth_line_overlay(spike_data=units_df,unit=unit,stim_data=gratings_df,condition='uv',
                                  title = 'UV')
                if save==True:
                    psth_color.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthColor.svg'))
                    psth_green.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthGreen.svg'))
                    psth_uv.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthUV.svg'))


            if psth == 'heatmap':
                psth_color = psth_bars(spike_data=units_df, unit=unit,stim_data = gratings_df,condition='color',
                          title = 'Color Condition')
                psth_green = psth_bars(spike_data=units_df, unit=unit,stim_data = gratings_df,condition='green',
                          title = 'Green')
                psth_uv = psth_bars(spike_data=units_df, unit=unit,stim_data = gratings_df,condition='uv',
                          title = 'UV')
                if save==True:
                    psth_color.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthColor.svg'))
                    psth_green.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthGreen.svg'))
                    psth_uv.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthUV.svg'))                
                
        else:
            if psth == 'lines':
                psth_ori = psth_line_overlay(spike_data = units_df, unit = unit, stim_data = gratings_df, condition='ori',
                                  title = 'Orientation')
                psth_sf = psth_line_overlay(spike_data = units_df, unit = unit, stim_data = gratings_df, condition='SF',
                                  title = 'Spatial Frequency')
                if save==True:
                    psth_ori.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthOri'+k+'.svg'))
                    psth_sf.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthSF'+k+'.svg'))

            if psth == 'heatmap':
                psth_ori = psth_bars(spike_data = units_df, unit = unit, stim_data = gratings_df, condition='ori',
                                  title = 'Orientation')
                psth_sf = psth_bars(spike_data = units_df, unit = unit, stim_data = gratings_df, condition='SF',
                                  title = 'Spatial Frequency')
                if save==True:
                    psth_ori.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthOri'+k+'.svg'))
                    psth_sf.savefig(os.path.join(image_path,'unit'+str(unit)+'_psthSF'+k+'.svg'))
    if save==True:
        print('Plots have been saved to '+image_path)