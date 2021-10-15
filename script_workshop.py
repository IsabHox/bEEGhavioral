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
subject='722'
subjects=['pilot3','136','358','916','347','766','661','959','205','400','756','897','420','804','207','196','295','722']


datapath='D:\\PhD\\my_data\\{}\\{}_calib_000{}'.format(subject,subject,{})
datapath_test='D:\\PhD\\my_data\\{}\\{}_test_000{}'.format(subject,subject,{})
nblocks=4

eegdata=import_raw(datapath,nblocks)
eegdata.plot_sensors(show_names=True)

#create and plot ERP
events=[11,12]
tmin=-0.5
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
from utils import compute_itc_from_phase, compute_witpc_from_phase

freqs={'delta':[1,4],
       'theta':[4,8],
       'alpha':[8,12],
       'beta':[12,25]}

pows={}
phases={}
itc1={}
itc2={}
itc_all={}
witc1={}
witc2={}
witc_all={}

for f in freqs:
    #first, filter raw data in relevant frequency band
    temp_data=eegdata.copy()
    temp_data.filter(freqs[f][0], freqs[f][1], fir_design='firwin')
    #then, apply hilbert transform
    temp_data.apply_hilbert()
    #now, epoch data in the relevant period and extract data
    ep=get_epochs(temp_data,events,tmin,tmax)
    ep.drop(rejt)
    my_data=ep.get_data()
    
    pows[f]=my_data.real**2+my_data.imag**2
    phases[f]=np.arctan(my_data.imag/my_data.real)
    itc1[f]=compute_itc_from_phase(phases[f][labels==11,:,:])#,RT[labels==11])
    itc2[f]=compute_itc_from_phase(phases[f][labels==12,:,:])#,RT[labels==12])
    itc_all[f]=compute_itc_from_phase(phases[f])#,RT)
    witc1[f]=compute_witpc_from_phase(phases[f][labels==11,:,:],RT[labels==11])
    witc2[f]=compute_witpc_from_phase(phases[f][labels==12,:,:],RT[labels==12])
    witc_all[f]=compute_witpc_from_phase(phases[f],RT)
    
#%%
for f in freqs:
    plt.figure()
    plt.pcolormesh(epochs.times,epochs.info['ch_names'],itc1[f])
    plt.title('ITC, Visual, {}, participant {}'.format(f, subject))
    plt.savefig('C:/Users/ihoxha/Desktop/PhD/Neurosceptic/presentations/figures/expe1/itc_poststim_{}_{}_visual'.format(f, subject), dpi=400)
    plt.figure()
    plt.pcolormesh(epochs.times,epochs.info['ch_names'],itc2[f])
    plt.title('ITC, Auditory, {}, participant {}'.format(f, subject))
    plt.savefig('C:/Users/ihoxha/Desktop/PhD/Neurosceptic/presentations/figures/expe1/itc_poststim_{}_{}_audio'.format(f, subject), dpi=400)
    plt.figure()
    plt.pcolormesh(epochs.times,epochs.info['ch_names'],itc_all[f])
    plt.title('ITC, All conditions, {}, participant {}'.format(f, subject))
    plt.savefig('C:/Users/ihoxha/Desktop/PhD/Neurosceptic/presentations/figures/expe1/itc_poststim_{}_{}_all'.format(f, subject), dpi=400)

#%% great, just need the normalization step now!
# permuted_itc1={}
# permuted_itc2={}
# permuted_itc_all={}

# niter=200
# RTcopy=RT.copy()
# RT1copy=RT[labels==11].copy()
# RT2copy=RT[labels==12].copy()

# for f in freqs:
#     perms=np.zeros((niter,clean_epochs.info['nchan'],len(clean_epochs.times)))
#     perms1=np.zeros((niter,clean_epochs.info['nchan'],len(clean_epochs.times)))
#     perms2=np.zeros((niter,clean_epochs.info['nchan'],len(clean_epochs.times)))
#     for n in range(niter):
#         np.random.shuffle(RTcopy)
#         np.random.shuffle(RT1copy)
#         np.random.shuffle(RT2copy)
#         perms[n,:,:]=compute_witpc_from_phase(phases[f],RTcopy)
#         perms1[n,:,:]=compute_witpc_from_phase(phases[f][labels==11,:,:],RT1copy)
#         perms2[n,:,:]=compute_witpc_from_phase(phases[f][labels==12,:,:],RT2copy)
#     permuted_itc_all[f]=perms
#     permuted_itc1[f]=perms1
#     permuted_itc2[f]=perms2