# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:24:37 2022

@author: bhutch1286
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from os.path import join

directory = 'F:/Phd year 3/n400 study/reanalysis_june_2021/OSF/EEG raw data/raw data/downsample_python'
save_directory_averages = directory + '/grand_averages'
figures_path = directory + '/figures'
os.chdir(directory)
os.getcwd()

biosemi_montage = mne.channels.make_standard_montage('biosemi64') #biosemi montage
eog_ch = ['EXG3','EXG4','EXG5','EXG6', 'EXG7', 'EXG8'] #EOG channel list for data import
ch_dict = {'EXG1': 'misc',
           'EXG2': 'misc'}

#N400_events = {'related/word1/phase1': 111, 'related/word1/phase2': 112, 'related/word1/phase3': 113,
#              'related/word2/phase1': 211, 'related/word2/phase2': 212, 'related/word2/phase3': 213,
#              'unrelated/word1/phase1': 121, 'unrelated/word1/phase2': 122, 'unrelated/word1/phase3': 123,
#              'unrelated/word2/phase1': 221, 'unrelated/word2/phase2': 222, 'unrelated/word2/phase3': 223}

N400_events = dict(related_w1_p1=111, related_w1_p2=112,related_w1_p3=113,
                related_w2_p1=211, related_w2_p2=212, related_w2_p3=213,
                unrelated_w1_p1=121, unrelated_w1_p2=122, unrelated_w1_p3=123,
                unrelated_w2_p1=221, unrelated_w2_p2=222, unrelated_w2_p3=223) #use this one instead (i think?)

low_freq = 0.2
high_freq = 30
name = 'sub01'
stim_channel = 'Status'
event_id=N400_events
tmin= -0.1
tmax= 1.0
baseline=(None,0)
reject=dict(eeg=150, eog=150)
decim=1 #integer division for downsampling
save_plots = True ## should plots be saved

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
bad_channels_dict[subjects[0]] = []
bad_channels_dict[subjects[1]] = []
#bad_channels_dict[subjects[2]] = ['Fpz', 'Fp2', 'Fp4']
#repeat for all subjects
evoked_data_all = dict(related_w1_p1=[], related_w1_p2=[], related_w1_p3=[],
                       related_w2_p1=[], related_w2_p2=[], related_w2_p3=[],
                       unrelated_w1_p1=[], unrelated_w1_p2=[], unrelated_w1_p3=[],
                       unrelated_w2_p1=[], unrelated_w2_p2=[], unrelated_w2_p3=[])


#%%           READ functions

#def read_data(name, save_dir): #----------------------------------for raw .bdf
#    raw_name = name + '.bdf'        
#    raw_path = join(save_dir, raw_name)
#    raw = mne.io.read_raw_bdf(input_fname=raw_path,
#                              eog=eog_ch,
#                              stim_channel = stim_channel,
#                              infer_types=True,
#                              preload=True)
#    return raw

def read_data(name, save_dir): #---------------------------for downsampled .fif
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

    
def plot_evokeds(name, directory, high_freq, subject, save_plots, figures_path): #STILL NEEDS WORK

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


#%%           PROCESSING PIPELINE
for subject in subjects:
    name = subject
    bad_channels = bad_channels_dict[subject]
    
    data = read_data(name, directory)
    #downsample_data(data_boy) #only unhash if downsampling is needed first
    filter_data(data, name, directory, low_freq, high_freq)
    event_finder(name, directory, stim_channel, high_freq)
    epoching(name, directory, high_freq, event_id, tmin, tmax,
              baseline, reject, bad_channels, decim)
    
    run_ica(name, directory, high_freq)
    #plot_ica(name, directory, high_freq, subject, save_plots, figures_path)
    apply_ica(name, directory, high_freq)
    plot_epochs_image(name, directory, high_freq, subject, save_plots, figures_path) #plot cleaned epoch [this step works!!!]
    get_evokeds(name, directory, high_freq)
    plot_evokeds(name, directory, high_freq, subject, save_plots,figures_path)
    
#end

evoked_data = read_evokeds(name, directory, high_freq)
for evoked in evoked_data:
    trial_type = evoked.comment
    evoked_data_all[trial_type].append(evoked)
    
grand_average_evokeds(evoked_data_all, save_directory_averages, high_freq)

#SUCCESS SO FAR!

#to do list:
    # reference function (goal is to implement PREP robust median reference here)
    # fix the issue with plot commands
    # add in overwrite feature to all functions
    #ICA fit suggests high pass filtering but no baseline correction; fix
