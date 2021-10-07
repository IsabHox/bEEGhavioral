# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:10:50 2021
SCript for the tutorial (split things into sections)
@author: ihoxha
"""

#%% imports
from utils import import_raw, get_epochs, plot_ERP
import matplotlib.pyplot as plt
import numpy as np

#%% load data (test which subject is best)
subject='756'
datapath='D:\\PhD\\my_data\\{}\\{}_calib_000{}'.format(subject,subject,{})
datapath_test='D:\\PhD\\my_data\\{}\\{}_test_000{}'.format(subject,subject,{})
nblocks=4

eegdata=import_raw(datapath,nblocks)
eegdata.plot_sensors(show_names=True)

#create and plot ERP
events=[11,12]
tmin=-1
tmax=0.5

#get epochs and plot ERP
epochs=get_epochs(eegdata, events,tmin=tmin, tmax=tmax)
plot_ERP(epochs,['11','12'])
#%%artifact rejection and filtering
from utils import get_bad_impedance, get_bad_channels_trials
fmin=None
fmax=30
eegdata_filtered=eegdata.filter(fmin, fmax, fir_design='firwin')

impedances=eegdata_filtered.impedances
get_bad_impedance(eegdata_filtered)
print(eegdata_filtered.info['bads'])
rejt,rejc=get_bad_channels_trials(eegdata_filtered,events)
eegdata_filtered.info['bads'].extend(rejc)
eegdata_filtered = eegdata_filtered.copy().interpolate_bads(reset_bads=False)
print(eegdata_filtered.info['bads'])

#%%
clean_epochs=get_epochs(eegdata_filtered, events,tmin=tmin, tmax=tmax)
clean_epochs.drop(rejt)
plot_ERP(clean_epochs,['11','12'])

#%% behavior
#extract RTs
from utils import get_RT, get_labels, get_errors

responses=[1001,1002,1003]
RT=get_RT(eegdata_filtered, events, responses, rejt)

# labels
labels=get_labels(clean_epochs)

# plot histograms
plt.figure()
plt.hist(RT[labels==11],alpha=0.2,label='Visual, mean={:.0f} ms'.format(np.mean(RT[labels==11])))
plt.hist(RT[labels==12],alpha=0.2,label='Auditory, mean={:.0f} ms'.format(np.mean(RT[labels==12])))
plt.title('Histogramm of response times, mean={:.0f} ms'.format(np.mean(RT)))
plt.legend()

# get errors
errors=get_errors(eegdata_filtered, events,[1001,1002,1003],rejt)
print('This participant committed {} errors'.format(np.sum(errors)))

#%% itpc