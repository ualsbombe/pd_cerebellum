#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:34:28 2021

@author: lau
"""

from config import (fname, submitting_method, split_recording_subjects,
                    hyades_parameters, events_with_256_added,
                    evoked_event_id)
from helper_functions import set_hyades_parameters
from sys import argv
from helper_functions import (should_we_run, read_split_raw,
                              fix_events_with_256_added)

import mne


def this_function(subject, date, overwrite):
    output_name = fname.events(subject=subject, date=date)
    figure_name = fname.events_plot(subject=subject, date=date)
    if should_we_run(output_name, overwrite):

        if subject in split_recording_subjects:
            raw = read_split_raw(subject, date)
        else:
            raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date))
            
        events = mne.find_events(raw, min_duration=0.002)
        
        if subject in events_with_256_added:
            events = fix_events_with_256_added(events, evoked_event_id)
        
        
        
        mne.write_events(output_name, events, overwrite=overwrite)
        fig = mne.viz.plot_events(events)
        fig.savefig(figure_name)
        
        
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)