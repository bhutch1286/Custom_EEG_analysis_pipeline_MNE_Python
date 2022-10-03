# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 08:09:59 2022

@author: blu12

"DO" functions, includes functions for processing, analysis, and plotting

"""
from __future__ import print_function

import mne
import numpy as np
from os.path import join, isfile, isdir
import matplotlib.pyplot as plt
import mayavi.mlab
from scipy import stats
from os import makedirs, listdir, environ
import sys
from . import io_functions as io
import pickle
import subprocess



#%%           DO functions

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
    raw.save(filter_path, overwrite=True)

def event_finder(name, directory, stim_channel, high_freq):
    events_name = name + '-eve.fif'
    events_path = join(directory, events_name)
    raw = read_filtered(name, directory, high_freq)
    events = mne.find_events(raw, stim_channel)
    mne.event.write_events(events_path, events, overwrite=True)

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
    
    epochs.save(epochs_path, overwrite=True)  

def run_ica(name, directory, high_freq):
    
    ica_name = name + '_' + str(high_freq) + '_Hz' + '-ica.fif'
    ica_path = join(directory, ica_name)

    raw = read_filtered(name, directory, high_freq) 
    epochs = read_epochs(name, directory, high_freq)
        
    ica = mne.preprocessing.ICA(n_components=0.95, method='fastica')
    ica.fit(epochs)
        
    eog_epochs = mne.preprocessing.create_eog_epochs(raw)
    eog_indices, eog_scores = ica.find_bads_eog(eog_epochs) 
    ica.exclude += eog_indices
    ica.save(ica_path, overwrite=True)

def apply_ica(name, directory, high_freq):
    
    ica_epochs_name = name + '_' + str(high_freq) + '_Hz' + '-ica-epo.fif'     
    ica_epochs_path = join(directory, ica_epochs_name)

    epochs = read_epochs(name, directory, high_freq)
    ica = read_ica(name, directory, high_freq)
    ica_epochs = ica.apply(epochs)
    ica_epochs.save(ica_epochs_path, overwrite=True)


def get_evokeds(name, directory, high_freq):
    evokeds_name = name + '_' + str(high_freq) + '_Hz' + '-ave.fif'     
    evokeds_path = join(directory, evokeds_name)
    epochs = read_ica_epochs(name, directory, high_freq)
    
    evokeds = []
    for trial_type in epochs.event_id:     
        evokeds.append(epochs[trial_type].average())
    
    mne.evoked.write_evokeds(evokeds_path, evokeds, overwrite = True)
        
        
def grand_average_evokeds(evoked_data_all, save_dir_averages, high_freq):

    grand_averages = dict()
    for trial_type in evoked_data_all:
        grand_averages[trial_type] = \
            mne.grand_average(evoked_data_all[trial_type]) #note: changed from mne.evoked.grand_average(evoked_data_all[trial_type])
            
    for trial_type in grand_averages:
        grand_average_path = save_dir_averages + \
            trial_type + '_' + str(high_freq) + '_Hz' + \
            '_grand_average-ave.fif'
        mne.evoked.write_evokeds(grand_average_path,
                                 grand_averages[trial_type])

#%%           PLOT functions
def plot_ica(name, directory, high_freq, subject, save_plots, figures_path):
    ica = read_ica(name, directory, high_freq)
    ica_figure = ica.plot_components(ica.exclude)
    if save_plots:
        save_path = join(figures_path, subject, 'ica', name + \
            '_' + str(high_freq) + '_Hz' + '.jpg')
        ica_figure.savefig(save_path, dpi=600)
        print('figure: ' + save_path + ' has been saved')
    else:
        print('Not saving plots; set "save_plots" to "True" to save')            

def plot_epochs_image(name, directory, high_freq, subject, save_plots, figures_path):
                          
    channel = 'Cz'     #primary channel for N400 data                   
    epochs = read_epochs(name, directory, high_freq)
    picks = mne.pick_channels(epochs.info['ch_names'], [channel])
    for trial_type in epochs.event_id:
        epochs_image = mne.viz.plot_epochs_image(epochs[trial_type], picks)
        plt.title(trial_type)

        if save_plots:
            save_path = join(figures_path, subject, 'epochs',
                             trial_type + '_' + channel + '_' + name + \
                             '_' + str(high_freq) + '_Hz' + '.jpg')          
            epochs_image[0].savefig(save_path, dpi=600)
            print('figure: ' + save_path + ' has been saved')
        else:
            print('Not saving plots; set "save_plots" to "True" to save')            

    
def plot_evokeds(name, directory, high_freq, subject, save_plots, figures_path):

    evokeds = read_evokeds(name, directory, high_freq)
    order = [
        'related_word1_p1', 'related_word1_p2', 'related_word1_p3',
        'related_word2_p1', 'related_word2_p2', 'related_word2_p3',
        'unrelated_word1_p1', 'unrelated_word1_p2', 'unrelated_word1_p3',
        'unrelated_word2_p1', 'unrelated_word2_p2', 'unrelated_word2_p3'
        ]  
    
   
    colours = ['white', 'blue', 'green',
               'purple', 'yellow', 'red', 
               'orange', 'pink',  'grey',
               'black', 'brown', 'magenta']
    
    # sort evokeds
    plot_evokeds = []
    plot_related = []
    plot_unrelated = []
   # for evoked_type in order:
   #     for evoked in evokeds:
   #         if evoked.comment == evoked_type:
   #             plot_evokeds.append(evoked)
   #         if evoked.comment == evoked_type and \
   #                 (evoked.comment[0] == 'r'):
   #             plot_related.append(evoked)            
   #         if evoked.comment == evoked_type and \
   #                 (evoked.comment[0] == 'u'):
   #             plot_unrelated.append(evoked)     #here is possibly where the issue is arising             
    
    plt.close('all')
    
    evoked_figure = mne.viz.plot_evoked_topo(plot_evokeds, color=colours)
    evoked_figure.comment = 'all_evokeds_'     
                                      
    related_figure = mne.viz.plot_evoked_topo(plot_related,
                                                color=colours[:6])                                         
    related_figure.comment = 'related_evokeds_'              
                                  
    unrelated_figure = mne.viz.plot_evoked_topo(plot_unrelated,
                                                color=colours[6:12])
    unrelated_figure.comment = 'unrelated_evokeds_'                                                
                                                
    figures = [evoked_figure, related_figure, unrelated_figure]
    
    if save_plots:
        for figure in figures:
            save_path = join(figures_path, subject, 'evokeds', 
                             figure.comment + name + \
                             '_' + str(high_freq) + '_Hz' + '.jpg')
            figure.savefig(save_path, dpi=600)
            print('figure: ' + save_path + ' has been saved')
    else:
        print('Not saving plots; set "save_plots" to "True" to save')

