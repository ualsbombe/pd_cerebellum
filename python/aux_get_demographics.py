#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:20:27 2023

@author: lau
"""

#%% IMPORTS

from config import (recordings, fname, split_recording_subjects, bad_subjects,
                    subjects_no_watershed)
import mne
from datetime import date as get_date ## to not get conflict
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np

#%% FIND AGES

age = dict(patient=list(), control=list())
hand = dict(patient=dict(), control=dict())
sex = dict(patient=dict(), control=dict())

for recording in recordings:
    if recording['subject'] == 'fsaverage':
        continue
    subject = recording['subject']
    if subject in bad_subjects:
        continue
    if subject in subjects_no_watershed['1_layer']:
        continue
    if recording['mr_date'] is None:
        continue
    date = recording['date']
    patient = recording['patient']
    
    print(subject)
    
    if subject in split_recording_subjects:
        raw_name = fname.split_raw_file_1(subject=subject, date=date)
    else:
        raw_name = fname.raw_file(subject=subject, date=date)
        
    info = mne.io.read_info(raw_name)
    meas_date = info['meas_date'].date()
    birthday = info['subject_info']['birthday']
    birthday = get_date(birthday[0], birthday[1], birthday[2])
    diff_years = relativedelta(meas_date, birthday).years
    if patient:
        age['patient'].append(diff_years)
        hand['patient'][subject] = info['subject_info']['hand']
        sex['patient'][subject] = info['subject_info']['sex']
    else:
        age['control'].append(diff_years)
        hand['control'][subject] = info['subject_info']['hand']
        sex['control'][subject] = info['subject_info']['sex']



#%% PLOT

means = (np.mean(age['patient']), np.mean(age['control']))
stds = (np.std(age['patient']), np.std(age['control']))
medians = (np.median(age['patient']), np.median(age['control']))


plt.close('all')
plt.figure()
plt.hist([age['patient'], age['control']])
plt.legend(['Patients', 'Controls'])
plt.xlim(30, 100)
plt.xlabel('Age')
plt.ylabel('#')
plt.text(80, 2.0, 'Median age:\nPatients: ' + str(medians[0]) + ' y\n' + \
         'Controls: ' + str(medians[1]) + ' y\n')
plt.show()

#%% HOW MANY?

n_men   = dict(patient=0, control=0)
n_women = dict(patient=0, control=0)

for kind in sex:
    if kind == 'patient':
        for subject in sex[kind]:
            if sex[kind][subject] == 1:
                n_men['patient'] += 1
            else:
                n_women['patient'] += 1
            
    elif kind == 'control':
        for subject in sex[kind]:
            if sex[kind][subject] == 1:
                n_men['control'] += 1
            else:
                n_women['control'] += 1
                
                
#%% UPDRS

updrs = list()
for recording in recordings:
    if recording['subject'] == 'fsaverage':
        continue
    if recording['subject'] in bad_subjects:
        continue
    if not recording['patient']:
        continue

    updrs.append(recording['updrs'])
    
print('n patients: ' + str(len(updrs)))
print('Mean: ' + str(round(np.mean(updrs), 1)))
print('Std: ' + str(round(np.std(updrs), 2)))