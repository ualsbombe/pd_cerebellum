#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:37:48 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_fmin, evoked_fmax,
                    split_recording_subjects, bad_channels, hyades_parameters)
from sys import argv
from helper_functions import (should_we_run, read_split_raw,
                              set_hyades_parameters)

import mne

def this_function(subject, date, overwrite):

    
    output_name = fname.evoked_filter(subject=subject, date=date,
                                      fmin=evoked_fmin, fmax=evoked_fmax)
   
    if should_we_run(output_name, overwrite):
        if subject in split_recording_subjects:
            raw = read_split_raw(subject, date)
        else:
            raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date), preload=True)
        
        raw.info['bads'] = bad_channels[subject]
        
        if evoked_fmin is None:
            l_freq = None
            h_freq = evoked_fmax
        elif evoked_fmax is None:
            l_freq = evoked_fmin
            h_freq = None
        else:
            l_freq = evoked_fmin
            h_freq = evoked_fmax
        
        raw.filter(l_freq, h_freq)
        raw.save(output_name, overwrite=overwrite)

set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)       