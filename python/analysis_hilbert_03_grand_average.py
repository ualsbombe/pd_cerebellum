#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:50:16 2023

@author: lau
"""

from config import (fname, submitting_method, hilbert_fmins, hilbert_fmaxs,
                    hilbert_tmin, hilbert_tmax, recordings, bad_subjects,
                    grand_average_types, hyades_parameters)
from sys import argv
from helper_functions import (should_we_run, does_it_go_in_grand_average,
                              set_hyades_parameters)

import mne

def this_function(subject, date, overwrite):
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        for grand_average_type in grand_average_types:
            output_name = fname.hilbert_grand_average_proj_interpolated(
                                subject=subject,
                                date=date,
                                fmin=fmin,
                                fmax=fmax,
                                tmin=hilbert_tmin,
                                tmax=hilbert_tmax,
                                Type=grand_average_type)
            
            if should_we_run(output_name, overwrite):
                grand_average_hilberts = dict()
                ## sort files
                subject_counter = 0
                for recording in recordings:
                    subject = recording['subject']
                    if subject == 'fsaverage':
                        continue
                    include_in_grand_average = \
                        does_it_go_in_grand_average(recording,
                                                    grand_average_type)
                        
                    if subject in bad_subjects:
                        continue
                    if not include_in_grand_average:
                        continue
                    
                    subject_counter += 1
                    
                    subject_date = recording['date']
                    hilberts = mne.read_evokeds(fname.hilbert_average_proj(
                                    subject=subject,
                                    date=subject_date,
                                    fmin=fmin,
                                    fmax=fmax,
                                    tmin=hilbert_tmin,
                                    tmax=hilbert_tmax))
                    
                    for hilbert in hilberts:
                        event = hilbert.comment
                        if subject_counter == 1:
                            grand_average_hilberts[event] = [hilbert]
                        else:
                            grand_average_hilberts[event].append(hilbert)
                            
                ## calculate grand averages
                grand_averages = list()
                for grand_average_hilbert in grand_average_hilberts:
                    these_hilberts = \
                        grand_average_hilberts[grand_average_hilbert]
                    grand_average = mne.grand_average(these_hilberts,
                                                      interpolate_bads=True)
                    grand_average.comment = grand_average.comment + ': ' + \
                        grand_average_hilbert
                    grand_averages.append(grand_average)
                    
                    
                mne.write_evokeds(output_name, grand_averages,
                                  overwrite=overwrite)
                
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)                