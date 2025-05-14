#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:13:32 2022

@author: lau
"""

from config import (fname, submitting_method, hilbert_lcmv_contrasts,
                    hilbert_lcmv_regularization, hilbert_lcmv_weight_norms,
                    hilbert_tmin, hilbert_tmax,
                    hilbert_fmins, hilbert_fmaxs, 
                    recordings, bad_subjects,
                    bem_conductivities, grand_average_types,
                    subjects_no_watershed, hyades_parameters)
from sys import argv
from helper_functions import (should_we_run, does_it_go_in_grand_average,
                              set_hyades_parameters)
import mne

def this_function(subject, date, overwrite):        
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        for hilbert_lcmv_weight_norm in hilbert_lcmv_weight_norms:
            for bem_conductivity in bem_conductivities:
                n_layers = len(bem_conductivity)
                for this_contrast in hilbert_lcmv_contrasts:
                    print(this_contrast)
                    first_event = this_contrast[0]
                    second_event = this_contrast[1]
                    for event in this_contrast:
                        for grand_average_type in grand_average_types:
                   
 
                            output_name = \
                            fname.source_hilbert_beamformer_grand_average(
                                subject=subject, date=date,
                                Type=grand_average_type,
                                fmin=fmin, fmax=fmax,
                                tmin=hilbert_tmin, tmax=hilbert_tmax,
                                reg=hilbert_lcmv_regularization,
                                event=event,
                                first_event=first_event, 
                                second_event=second_event,
                                weight_norm=hilbert_lcmv_weight_norm,
                                n_layers=n_layers
                                )    

                   
                            if should_we_run(output_name, overwrite):
                                subject_counter = 0
                                for recording in recordings:
                                    subject_name = recording['subject']
                                    if subject_name == 'fsaverage':
                                        continue
                                    include_in_grand_average = \
                                        does_it_go_in_grand_average(recording,
                                                        grand_average_type)
                                    if subject_name in bad_subjects:# or \
                                        continue # skip the subject
                                    if subject_name in \
                                    subjects_no_watershed[str(n_layers) + \
                                                          '_layer']:
                                            continue # skip the subject
                                    if not include_in_grand_average:
                                            continue
                                    if recording['mr_date'] is None:
                                            continue
                                    subject_counter += 1
                                    subject_date = recording['date']
                                    
                                    lcmv = mne.read_source_estimate(
                                    fname.source_hilbert_beamformer_morph(
                                    subject=subject_name, 
                                    date=subject_date,
                                    fmin=fmin, fmax=fmax,
                                    tmin=hilbert_tmin,
                                    tmax=hilbert_tmax,
                                    reg=hilbert_lcmv_regularization, 
                                    event=event,
                                    first_event=first_event,
                                    second_event=second_event,
                                    weight_norm=hilbert_lcmv_weight_norm,
                                    n_layers=1))
                            
                                    ## single grand averages
                                    if subject_counter == 1:
                                        grand_average = lcmv.copy()
                                    else:
                                        grand_average._data += lcmv.data
                                      
                                # get the mean        
                                grand_average._data /= subject_counter 
                                grand_average.save(output_name, ftype='h5')
                    
                
     
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                          submitting_method)    
                

    