# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:24:37 2022

@author: bhutch1286
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from os.path import join

os.getcwd()
os.chdir(directory)

biosemi_montage = mne.channels.make_standard_montage('biosemi64') #biosemi montage
eog_ch = ['EXG3','EXG4','EXG5','EXG6', 'EXG7', 'EXG8'] #EOG channel list for data import
ch_dict = {'EXG1': 'misc',
           'EXG2': 'misc'}

directory = 'C:/documents/EEG/raw data/downsample_python'

event_dict = {'related/word1/phase1': 111, 'related/word1/phase2': 112, 'related/word1/phase3': 113,
              'related/word2/phase1': 211, 'related/word2/phase2': 212, 'related/word2/phase3': 213,
              'unrelated/word1/phase1': 121, 'unrelated/word1/phase2': 122, 'unrelated/word1/phase3': 123,
              'unrelated/word2/phase1': 221, 'unrelated/word2/phase2': 222, 'unrelated/word2/phase3': 223}

low_freq = 0.2
high_freq = 30
name = 'sub01'
stim_channel = 'Status'
event_id=event_dict
tmin= -0.1
tmax= 1.0
baseline=(None,0)
reject=dict(eeg=100, eog=150)
bad_channels= #
decim=1

subjects = [
    'sub01',
    'sub02',
    #'sub03',
    #'sub04',    
    #'sub05',
    #'sub06',
    #'sub07',
    #'sub08',
    #'sub09',
    #'sub10',
    #'sub11',
    #'sub11_2',    
    #'sub12',
    #'sub13',
    #'sub14',    
    #'sub15',
    #'sub16',
    #'sub17',
    #'sub18',
    #'sub19',
    #'sub20',
    #'sub21',
    #'sub22',
    #'sub23',
    #'sub24',    
    #'sub25',
    #'sub26',
    #'sub27' 
    ]

bad_channels_dict = dict()
bad_channels_dict[subjects[0]] = ['Cp6']
bad_channels_dict[subjects[1]] = ['Af8']
#bad_channels_dict[subjects[2]] = ['Fpz', 'Fp2', 'Fp4']
#repeat for all subjects


#%%Read functions

#def read_data(name, save_dir): #for raw .bdf
#    raw_name = name + '.bdf'        
#    raw_path = join(save_dir, raw_name)
#    raw = mne.io.read_raw_bdf(input_fname=raw_path,
#                              eog=eog_ch,
#                              stim_channel = stim_channel,
#                              infer_types=True,
#                              preload=True)
#    return raw

def read_data(name, save_dir): #for downsampled .fif
    raw_name = name + '_downsampled.fif'        
    raw_path = join(save_dir, raw_name)
    raw = mne.io.read_raw(fname=raw_path, preload=True)
    return raw

def read_filtered(name, directory, high_freq):  
    raw_name = name + '_' + str(high_freq) + '_Hz' + '-raw.fif'
    raw_path = join(directory, raw_name)
    raw = mne.io.read_raw(raw_path, preload=True)
    return raw

def read_events(name, directory):
    events_name = name + '-eve.bdf'
    events_path = join(directory, events_name)
    events = mne.read_events(events_path, mask=None)
    return events
    
#%%Do functions

#####come back to reference function
#def reference_data(raw):
#    raw_aref = raw.set_eeg_reference(ref_channels='average', #'average' for average
#                                     projection=False,
#                                     ch_type='auto',
#                                     forward=None,
#                                     verbose=None)
#    return raw_aref

def downsample_data(raw, integer_multiple):
    current_sfreq = raw.info['sfreq']
    desired_sfreq = current_sfreq / integer_multiple #integer multiple = 4 downsamples a 2048 Hz srate to 512 Hz 
    data_downsampled = raw.resample(sfreq=desired_sfreq)
    data_downsampled.save(fname= name + '_downsampled.fif')

def filter_data(raw, name, directory, low_freq, high_freq):
    filter_name = name + '_' + str(high_freq) + '_Hz' + '-raw.fif'
    filter_path = join(directory, filter_name)
    raw = raw.filter(l_freq = low_freq,
                     h_freq = high_freq,
                     picks=None,
                     method='fir',
                     phase='zero',
                     fir_window='hamming',
                     fir_design='firwin',
                     pad='reflect_limited')
    raw.save(filter_path)

def event_finder(name, directory, stim_channel, high_freq):
    events_name = name + '-eve.fif'
    events_path = join(directory, events_name)
    raw = read_filtered(name, directory, high_freq)
    events = mne.find_events(raw, stim_channel)
    mne.event.write_events(events_path, events)

def epoching(name, directory, high_freq, event_id, tmin, tmax,
              baseline, reject, bad_channels, decim):
    
    epochs_name = name + '_' + str(high_freq) + '_Hz' + '-epo.fif'
    epochs_path = join(directory, epochs_name)                       
    
    events = read_events(name, directory)
    raw = read_filtered(name, directory, high_freq)
    raw.info['bads'] = bad_channels
    picks = mne.pick_types(raw.info, eeg=True, eog=True, exclude='bads')
            
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline,
                        reject=reject, preload=True, picks=picks,
                        decim=decim)
    
    epochs.save(epochs_path)  

#%%plotting functions



#%% processing pipeline
for subject in subjects:
    name = subject
    bad_channels = bad_channels_dict[subject]
    
    data = read_data(name, directory)
    #downsample_data(data_boy) #only unhash if downsampling is needed first
    filter_data(data, name, directory, low_freq, high_freq)
    event_finder(name, directory, stim_channel, high_freq)
    epoching(name, directory, high_freq, event_id, tmin, tmax,
              baseline, reject, bad_channels, decim)
    
    run_ica()
    plot_ica()
    apply_ica()
    plot_epochs_image()
    get_evokeds()
    plot_evokeds()
    
#end


#SUCCESS SO FAR!

        #run_ica(name, save_dir, lowpass, overwrite=overwrite)
        #plot_ica(name, save_dir, lowpass, subject, save_plots, figures_path)
        #apply_ica(name, save_dir, lowpass, overwrite=overwrite)
        #plot_epochs_image(name, save_dir, lowpass, subject, save_plots, figures_path)
        #get_evokeds(name, save_dir, lowpass, overwrite=overwrite)
        #plot_evokeds(name, save_dir, lowpass, subject, save_plots,figures_path)
#six steps left

