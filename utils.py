# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 10:52:26 2021
Utility functions that can be used for the tutorial.
@author: ihoxha
"""
import numpy as np
import mne
from collections import Counter
import pandas as pd

#%% import and preprocessing
def import_raw(datapath,nblock):
    '''Imports raw data as an MNE structure.
    Input: datapath (str) path to reach data files.
    Ouput: eegdata (MNE.Raw) relevant data'''
    eegdatas=[0 for i in range(nblock)]
    for b in range (nblock):
        filename=datapath.format(b+1)
        vhdr_fname=filename+'.vhdr';
        eegdatas[b]=mne.io.read_raw_brainvision(vhdr_fname,preload=True)
        
    eegdata=mne.concatenate_raws(eegdatas,preload=True)
    montage1020 = mne.channels.make_standard_montage('standard_1020')
    eegdata = eegdata.set_montage(montage1020)
    return eegdata

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

def get_bad_channels_trials(eegdata,event_id,thresh_trial=110e-6,thresh_chans=0.15,tmin=-1,tmax=3,reject_tmin=-0.9,reject_tmax=0.5):
    '''Returns bad channels and trials given the specified thresholds.
    Inputs:
        thresh_trial: (default 100e-6) threshold value from which an epoch should be rejected (in V)
        thresh_chans: (default 0.15) proportion of trials rejected to decide to reject channels. Should be between 0 and 1
    Outputs:
        rej_trials: list of trial indices that should be rejected
        rej_channels: list of str of the rejected channels'''
        
    all_evt,evt_dict=mne.events_from_annotations(eegdata)
    stim_ix=np.where((all_evt[:,2]==event_id[0]) | (all_evt[:,2]==event_id[1]))[0]
    events=all_evt[stim_ix,:]
    
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
                new_log=list(drop_log[i])
                new_log.remove(chan_to_rej)
                if new_log is None:
                    drop_log[i]=()
                else:
                    drop_log[i]=tuple(new_log)
        stats=len(get_rej_trials(drop_log))/len(events)
        stat_details=get_bad_stats(drop_log)
        ch+=1
    rej_trials=get_rej_trials(drop_log)
    return rej_trials,rej_channels

#%% get functions to extract information from data
def get_epochs(eegdata,event_id,tmin,tmax):
    events, event_dict = mne.events_from_annotations(eegdata)
    evt_ix=np.where(events[:,2]==event_id[0])
    for ix in range(1,len(event_id)):
        evt_ix=np.hstack((evt_ix, np.where(events[:,2]==event_id[ix])))
    stim_events=np.squeeze(events[evt_ix])
    stim_events=np.sort(stim_events.view('int,int,int'), order=['f1'], axis=0).view(np.int)

    metadata = {'event_time': stim_events[:, 0],
                'trial_number': range(len(stim_events))}#
    metadata = pd.DataFrame(metadata)
    
    epochs = mne.Epochs(eegdata, stim_events, event_id, tmin, tmax, proj=True, baseline=None, metadata=metadata, detrend=0, preload=True,event_repeated='merge')
    return epochs

def get_labels(epochs):
    '''Extract labels of epochs'''
    evt_sorted=np.sort(epochs.events.view('int,int,int'), order=['f1'], axis=0).view(np.int)
    labels = evt_sorted[:, -1]
    return labels

def get_RT(eegdata, stim_labels, resp_labels, rejected=[]):
    '''Extract response times from markers in EEG recordings, given the labels of the stimulus and the labels of the response.
    Inputs:
        - eegdata: mne.io.Raw instance
        - stim_labels: list of stimulus labels
        - resp_labels: list of response labels
        - rejected (default []): list of trials to be rejected
    Output:
        - RT: list of response times'''
    events, event_dict= mne.events_from_annotations(eegdata)
    subevt_ix=np.where((events[:,2]==stim_labels[0]) | (events[:,2]==stim_labels[1])| (events[:,2]==resp_labels[0])| (events[:,2]==resp_labels[1])| (events[:,2]==resp_labels[2]))[0]
    stim_ix=np.where((events[:,2]==stim_labels[0]) | (events[:,2]==stim_labels[1]))[0]
    if len(rejected)!=0:
        stim_ix=np.delete(stim_ix,rejected)
    _,which_timings,_=np.intersect1d(subevt_ix, stim_ix, return_indices=True)
    stim_and_resp_events=events[subevt_ix,0]
    RT=np.diff(stim_and_resp_events)
    RT=RT[which_timings]
    return RT

def get_errors(eegdata, stim_labels, resp_labels, rejected=[]):
    '''Returns a list of binary labels, where 1 indicates that the participant did not provide the right response
    Inputs:
        - eegdata: mne.io.Raw instance
        - stim_labels: list of stimulus labels
        - resp_labels: list of response labels
        - rejected (default []): list of trials to reject
    Output:
        -errors: binary list (n_trials) indicating error trials'''
    events, event_dict= mne.events_from_annotations(eegdata)
    responses=events[np.where((events[:,2]==resp_labels[0])| (events[:,2]==resp_labels[1])| (events[:,2]==resp_labels[2]))[0],2]-1000
    stimuli=events[np.where((events[:,2]==stim_labels[0]) | (events[:,2]==stim_labels[1]))[0],2]-10
    errors=(stimuli!=responses).astype(int)
    return errors

#%% utility functions for the ITPC and wITPC
def compute_itc_from_phase(phases):
    n=np.shape(phases)[0]
    itc=np.abs(np.sum(np.exp(1j*phases),axis=0)/n)
    return itc

def compute_witpc_from_phase(phases,weights):
    n=np.shape(phases)[0]
    witpc=np.abs(np.sum(np.multiply(weights[:, np.newaxis, np.newaxis],np.exp(1j*phases)),axis=0)/n)
    return witpc

#%% plotting functions
def plot_ERP(epochs,events):
    for e in events:
        ev=epochs[e].average()
        ev.plot_joint(title=e)
    all_ev=epochs.average()
    all_ev.plot_joint(title='All conditions')
    return

