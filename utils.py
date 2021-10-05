# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:52:26 2021
Utility functions that can be used for the tutorial.
@author: ihoxha
"""
import numpy as np
import mne
from collections import Counter

#%% import and preprocessing
def import_raw(datapath):
    '''Imports raw data as an MNE structure.
    Input: datapath (str) path to reach data files.
    Ouput: eegdata (MNE.Raw) relevant data'''
    return

def filter_raw(eegdata):
    return

def get_bad_impedance(eegdata,thresh=50):
    '''Adds a "bad" flag to channels whose impedance is beyond the specified threshold.
    Acts in place (ie modifies the input)'''
    impedances=eegdata.impedances.copy()
    del impedances['Ref']
    for i in impedances:
        if np.isnan(impedances[i]['imp']):
            eegdata.info['bads'].extend([i])
        elif impedances[i]['imp']>thresh:
            eegdata.info['bads'].extend([i])
    return
def get_rej_trials(drop_log):
    '''From a drop log, get the index of trials that have been rejected'''
    ix_list=[]
    for i,t in enumerate(drop_log):
        if len(t)!=0:
            ix_list.append(i)
    return ix_list

def get_bad_stats(drop_log):
    scores = Counter([ch for d in drop_log for ch in d])
    return scores

def get_bad_channels_trials(eegdata,events,thresh_trial=110e-6,thresh_chans=0.15,tmin=-1,tmax=3,reject_tmin=-0.2,reject_tmax=1):
    '''Returns bad channels and trials given the specified thresholds.
    Inputs:
        thresh_trial: (default 100e-6) threshold value from which an epoch should be rejected (in V)
        thresh_chans: (default 0.15) proportion of trials rejected to decide to reject channels. Should be between 0 and 1
    Outputs:
        rej_trials: list of trial indices that should be rejected
        rej_channels: list of str of the rejected channels'''
    rej_dict=dict(eeg=thresh_trial)
    ep = mne.Epochs(eegdata, events, baseline=None,tmin=tmin, tmax=tmax,reject=rej_dict,reject_tmin=reject_tmin,reject_tmax=reject_tmax, preload=True)
    drop_log=list(ep.drop_log)
    stats=len(get_rej_trials(drop_log))/len(events) #ep.drop_log_stats()
    stat_details=get_bad_stats(drop_log)
    rej_trials=get_rej_trials(drop_log)
    rej_channels=[]
    ch=0
    while stats>thresh_chans and ch<eegdata.info['nchan']:
        chan_to_rej=stat_details.most_common()[0][0]
        rej_channels.extend([chan_to_rej])
        for i in range(len(drop_log)):
            if chan_to_rej in drop_log[i]:
                new_log=list(drop_log[i]).remove(chan_to_rej)
                if new_log is None:
                    drop_log[i]=()
                else:
                    drop_log[i]=tuple(new_log)
        stats=len(get_rej_trials(drop_log))/len(events)
        stat_details=get_bad_stats(drop_log)
        ch+=1
    rej_trials=get_rej_trials(drop_log)
    return rej_trials,rej_channels

#%% 
def get_epochs(eegdata,class_labels,tmin,tmax):
    return 

def get_labels(epochs,class_labels):
    '''Extract labels of epochs'''
    return

def get_RT(eegdata, stim_labels, resp_labels):
    return

def get_errors(eegdata, stim_labels, resp_labels):
    return



