# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 08:08:44 2022

@author: blu12
"""
from __future__ import print_function

import mne
from os.path import join
import pickle


#%%           READ functions

def read_data_raw(name, save_dir): #----------------------------------for raw .bdf
    raw_name = name + '.bdf'        
    raw_path = join(save_dir, raw_name)
    raw = mne.io.read_raw_bdf(input_fname=raw_path,
                              eog=eog_ch,
                              stim_channel = stim_channel,
                              infer_types=True,
                              preload=True)
    return raw

def read_data_downsampled(name, save_dir): #---------------------------for downsampled .fif
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
    events_name = name + '-eve.fif'
    events_path = join(directory, events_name)
    events = mne.read_events(events_path, mask=None)
    return events

def read_epochs(name, directory, high_freq):    
    epochs_name = name + '_' + str(high_freq) + '_Hz' + '-epo.fif'
    epochs_path = join(directory, epochs_name)                       
    epochs = mne.read_epochs(epochs_path)
    return epochs

def read_ica(name, directory, high_freq):
    ica_name = name + '_' + str(high_freq) + '_Hz' + '-ica.fif'
    ica_path = join(directory, ica_name)
    ica = mne.preprocessing.read_ica(ica_path)    
    return ica

def read_ica_epochs(name, directory, high_freq):
    ica_epochs_name = name + '_' + str(high_freq) + '_Hz' + '-ica-epo.fif'
    ica_epochs_path = join(directory, ica_epochs_name)
    ica_epochs = mne.read_epochs(ica_epochs_path)
    return(ica_epochs)

def read_evokeds(name, directory, high_freq):
    evokeds_name = name + '_' + str(high_freq) + '_Hz' + '-ave.fif' 
    evokeds_path = join(directory, evokeds_name)
    evokeds = mne.read_evokeds(evokeds_path)
    return evokeds  