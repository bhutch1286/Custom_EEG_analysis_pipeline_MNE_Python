# -*- coding: utf-8 -*-
"""
Created on Wed Jul 6 15:24:37 2022

@author: bhutch1286
"""
import os
from os import __file__

import numpy as np
import matplotlib.pyplot as plt
import mne
from os.path import join

from pipe_functions import read_functions as rd
from pipe_functions import do_functions as do

#%%           DIRECTORY AND PROJECT ENVIRONMENT SET UP

#-----------------------------Directory---------------------------------------#
"""
Do you want to use a custom directory (e.g., on an external HD) or just create a directory on C drive?
    1 = former
    0 = latter
"""
directory_wanted = 1
directory = 'F:/Phd year 3/n400 study/reanalysis_june_2021/OSF/EEG raw data/raw data/downsample_python' #put your custom directory here

#-------------------------Project Environment---------------------------------#

"""
First need to reset the working directory to base C drive/ documents path incase anything weird is going on with your directory.
This assumes C drive path can be derived from using the path to where you have Python installed, as this is what it uses.
So if that isn't the case, will need to be modified.
"""

python_here = os.path.realpath(__file__) #identify where python is installed

# function to get parent folders
def getParent(path, levels = 1):
    common = path
 
    # Using for loop for getting starting point required for os.path.relpath()
    for i in range(levels + 1):
         # Starting point
        common = os.path.dirname(common)
 
        # Parent directory upto specified level
    return os.path.relpath(path, common)
 
path = python_here
remove_folders = getParent(path, 3) #change if need to change the number of subfolders to remove
wd = python_here.replace(remove_these, '')
os.chdir(wd)

#Either change to specified custom directory or stay in base path
if directory_wanted == 0:
    wd = os.getcwd()
else:
    os.chdir(directory)
    wd = os.getcwd()

#make project folder and make this our new directory
mkdir EEG_Project_MNE_python
os.chdir(wd + "\\EEG_Project_MNE_python")
wd = os.getcwd()

#sub folders for storing grand averages and figures
save_directory_averages = wd + '\\grand_averages'
figures_path = wd + '\\figures'

#%%           PARAMETERS AND DATA SET UP

"""Most of this will need to be customised depending on what your data looks like."""

biosemi_montage = mne.channels.make_standard_montage('biosemi64') #what scalp montage is the data?
eog_ch = ['EXG3','EXG4','EXG5','EXG6', 'EXG7', 'EXG8'] #what external electrodes if any were used?
ch_dict = {'EXG1': 'misc',
           'EXG2': 'misc'}

N400_events = dict(related_w1_p1=111, related_w1_p2=112,related_w1_p3=113,
                   related_w2_p1=211, related_w2_p2=212, related_w2_p3=213,
                   unrelated_w1_p1=121, unrelated_w1_p2=122, unrelated_w1_p3=123,
                   unrelated_w2_p1=221, unrelated_w2_p2=222, unrelated_w2_p3=223)

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


#Subjects
subject_number = 27 #how many subjects are there?
subjects = []

for i in range(1, subject_number+1): #this will need to be  changed depending on what your subject files are called
    if i <= 9:
        subject = "sub0" + str(i)
    else:
        subject = "sub" + str(i)
    subjects.append(subject)

#Bad channels
bad_channels_dict = dict()
bad_channels_dict[subjects[0]] = []
bad_channels_dict[subjects[1]] = []
#bad_channels_dict[subjects[2]] = ['Fpz', 'Fp2', 'Fp4']
#repeat for all subjects

evoked_data_all = dict(related_w1_p1=[], related_w1_p2=[], related_w1_p3=[],
                       related_w2_p1=[], related_w2_p2=[], related_w2_p3=[],
                       unrelated_w1_p1=[], unrelated_w1_p2=[], unrelated_w1_p3=[],
                       unrelated_w2_p1=[], unrelated_w2_p2=[], unrelated_w2_p3=[])


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
    

#grand_average_evokeds() #outside of participant for loop
grand_average_evokeds(evoked_data_all, save_directory_averages, high_freq)

#SUCCESS SO FAR!

        #run_ica(name, directory, high_freq)
        #plot_ica(name, directory, high_freq, subject, save_plots, figures_path)
        #apply_ica(name, directory, high_freq)
        #plot_epochs_image(name, directory, high_freq, subject, save_plots, figures_path)
        #get_evokeds(name, directory, high_freq)
        #plot_evokeds(name, directory, high_freq, subject, save_plots,figures_path)
#six steps left

#to do list:
    # fix the thingy with plot evoked
    # add in overwrite feature to all functions
    #RuntimeWarning: The epochs you passed to ICA.fit() were baseline-corrected. However, we suggest to fit ICA only on data that has been high-pass filtered, but NOT baseline-corrected.
    #

