#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:55:17 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_fmin, evoked_fmax,
                    evoked_tmin, evoked_tmax, hyades_parameters)

from sys import argv
from helper_functions import should_we_run, set_hyades_parameters

import mne

def this_function(subject, date, overwrite):
    
    output_names = list()
    output_names.append(fname.evoked_average_no_proj(subject=subject,
                                                    date=date,
                                                    fmin=evoked_fmin,
                                                    fmax=evoked_fmax,
                                                    tmin=evoked_tmin,
                                                    tmax=evoked_tmax))
                        
    output_names.append(fname.evoked_average_proj(subject=subject,
                                                    date=date,
                                                    fmin=evoked_fmin,
                                                    fmax=evoked_fmax,
                                                    tmin=evoked_tmin,
                                                    tmax=evoked_tmax))
    
    for output_name in output_names:
        if should_we_run(output_name, overwrite):
            if 'no_proj' in output_name:
                epochs = mne.read_epochs(fname.evoked_epochs(
                    subject=subject, date=date, fmin=evoked_fmin,
                     fmax=evoked_fmax, tmin=evoked_tmin, tmax=evoked_tmax),
                    proj=False)
            else:
                epochs = mne.read_epochs(fname.evoked_epochs(
                    subject=subject, date=date, fmin=evoked_fmin,
                     fmax=evoked_fmax, tmin=evoked_tmin, tmax=evoked_tmax),
                    proj=True)
            evokeds = list()
            
            ## collapse n events
            old_event_ids = list()
            new_event_id = dict(n=40)
            for event in epochs.event_id:
                if 'n' in event:
                    old_event_ids.append(event)
            
            epochs = mne.epochs.combine_event_ids(epochs, old_event_ids,
                                                  new_event_id)
            
            for event in epochs.event_id:
                evokeds.append(epochs[event].average())
                    
                
            mne.write_evokeds(output_name, evokeds, overwrite=overwrite)
            
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)       