#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:14:49 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_fmin, evoked_fmax,
                    evoked_tmin, evoked_tmax, evoked_baseline, evoked_decim,
                    evoked_event_id, hyades_parameters)

from sys import argv
from helper_functions import (should_we_run, check_if_all_events_present,
                              set_hyades_parameters)

import mne

def this_function(subject, date, overwrite):
    
                        
    output_name = fname.evoked_epochs(subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax)
    
    
    if should_we_run(output_name, overwrite):
        raw = mne.io.read_raw_fif(fname.evoked_filter(subject=subject,
                                                      date=date,
                                                      fmin=evoked_fmin,
                                                      fmax=evoked_fmax))
        events = mne.read_events(fname.events(subject=subject, date=date))
        event_id = check_if_all_events_present(events, evoked_event_id)
      
        epochs = mne.Epochs(raw, events, event_id,
                            evoked_tmin, evoked_tmax, evoked_baseline,
                            decim=evoked_decim,
                            proj=False)
        
        epochs.save(output_name, overwrite=overwrite)
            
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)     