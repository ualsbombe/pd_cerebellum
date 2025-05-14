#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:24:43 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_fmin, evoked_fmax,
                    evoked_tmin, evoked_tmax, recordings, bad_subjects,
                    grand_average_types, hyades_parameters)
from sys import argv
from helper_functions import (should_we_run, does_it_go_in_grand_average,
                              set_hyades_parameters)
import mne

def this_function(subject, date, overwrite):
    for grand_average_type in grand_average_types:
        output_name = fname.evoked_grand_average_proj_interpolated(
                                            subject=subject,
                                            date=date,
                                            fmin=evoked_fmin,
                                            fmax=evoked_fmax,
                                            tmin=evoked_tmin,
                                            tmax=evoked_tmax,
                                            Type=grand_average_type)
    
        if should_we_run(output_name, overwrite):
            grand_average_evokeds = dict()
            ## sort files
            subject_counter = 0
            for recording in recordings:
                subject = recording['subject']
                if subject == 'fsaverage':
                    continue
                include_in_grand_average = \
                    does_it_go_in_grand_average(recording,
                                                grand_average_type)
                if subject in bad_subjects:# or \
                    continue ## skip the subject
                if not include_in_grand_average:
                    continue
                
                subject_counter += 1 
    
                subject_date = recording['date']
                evokeds = mne.read_evokeds(fname.evoked_average_proj(
                                    subject=subject,
                                    date=subject_date,
                                    fmin=evoked_fmin,
                                    fmax=evoked_fmax,
                                    tmin=evoked_tmin,
                                    tmax=evoked_tmax))
                
                for evoked in evokeds:
                    event = evoked.comment
                    if subject_counter == 1:
                        grand_average_evokeds[event] = [evoked]
                    else:
                        grand_average_evokeds[event].append(evoked)
            
            ## calculate grand averages
            grand_averages = list()
            for grand_average_evoked in grand_average_evokeds:
                these_evokeds = grand_average_evokeds[grand_average_evoked]    
                grand_average = mne.grand_average(these_evokeds,
                                                      interpolate_bads=True)
                grand_average.comment = grand_average.comment + ': ' + \
                    grand_average_evoked
                grand_averages.append(grand_average)
                
            mne.write_evokeds(output_name, grand_averages, overwrite=overwrite)
        
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                         submitting_method)   
