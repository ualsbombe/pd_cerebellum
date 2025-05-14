#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:06:36 2021

@author: lau
"""

from config import (fname, submitting_method, hilbert_fmins, hilbert_fmaxs,
                    hilbert_tmin, hilbert_tmax, hilbert_baseline,
                    hilbert_event_id, hyades_parameters)

from sys import argv
from helper_functions import (should_we_run, check_if_all_events_present,
                              set_hyades_parameters)

import mne

def this_function(subject, date, overwrite):
    
    for hilbert_fmin, hilbert_fmax in zip(hilbert_fmins, hilbert_fmaxs):

    
       output_name = fname.hilbert_epochs(   
        subject=subject,
        date=date,
        fmin=hilbert_fmin,
        fmax=hilbert_fmax,
        tmin=hilbert_tmin,
        tmax=hilbert_tmax)
        
        
       
       if should_we_run(output_name, overwrite):
           raw = mne.io.read_raw_fif(fname.hilbert_filter(subject=subject,
                                           date=date,
                                         fmin=hilbert_fmin,
                                         fmax=hilbert_fmax),
                                     preload=True)
           raw.apply_hilbert(envelope=False)
           events = mne.read_events(fname.events(subject=subject,
                                               date=date))
           event_id = check_if_all_events_present(events,
                                                   hilbert_event_id)
           proj = False
           reject = None
          
             
           epochs = mne.Epochs(raw, events, event_id,
                             hilbert_tmin, hilbert_tmax,
                             hilbert_baseline,
                             reject=reject,
                             proj=proj)
 
           epochs.save(output_name, overwrite=overwrite)
           epochs = None
           raw = None
            
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)