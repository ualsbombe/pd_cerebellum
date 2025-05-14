#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:59:55 2022

@author: lau
"""


from config import (fname, submitting_method, evoked_lcmv_contrasts,
                    evoked_lcmv_regularization, evoked_lcmv_weight_norms,
                    evoked_tmin, evoked_tmax,
                    evoked_fmin, evoked_fmax, 
                    recordings, bad_subjects,
                    subjects_no_watershed, grand_average_types,
                    bem_conductivities, hyades_parameters)
from sys import argv
from helper_functions import (should_we_run, set_hyades_parameters,
                              does_it_go_in_grand_average)
import mne

def this_function(subject, date, overwrite):
    for this_contrast in evoked_lcmv_contrasts:
        print(this_contrast)
        for event in this_contrast:
            for evoked_lcmv_weight_norm in evoked_lcmv_weight_norms:
                for bem_conductivity in bem_conductivities:
                    n_layers = len(bem_conductivity)
                    for grand_average_type in grand_average_types: 
                        
                        output_name = \
                            fname.source_evoked_beamformer_grand_average(
                            subject=subject, date=date,
                            Type=grand_average_type,
                            fmin=evoked_fmin, fmax=evoked_fmax,
                            tmin=evoked_tmin, tmax=evoked_tmax,
                            reg=evoked_lcmv_regularization,
                            event=event,
                            first_event=this_contrast[0],
                            second_event=this_contrast[1],
                            weight_norm=evoked_lcmv_weight_norm,
                            n_layers=n_layers)
                    
                        if should_we_run(output_name, overwrite):
                            subject_counter = 0
                            for recording in recordings:
                                subject_name = recording['subject']
                                if subject_name == 'fsaverage':
                                    continue
                                include_in_grand_average = \
                                    does_it_go_in_grand_average(recording,
                                                            grand_average_type)
                                if subject_name in bad_subjects:
                                    continue
                                if subject_name in \
                            subjects_no_watershed[str(n_layers) + '_layer']:
                                    continue # skip the subject
                                if not include_in_grand_average:
                                    continue
                                if recording['mr_date'] is None:
                                    continue

                                subject_counter += 1
                                date_name = recording['date']
                                
                                lcmv = mne.read_source_estimate(
                                    fname.source_evoked_beamformer_morph(
                                subject=subject_name,date=date_name,
                                fmin=evoked_fmin, fmax=evoked_fmax,
                                tmin=evoked_tmin,
                                tmax=evoked_tmax,
                                reg=evoked_lcmv_regularization, event=event,
                                first_event=this_contrast[0],
                                second_event=this_contrast[1],
                                weight_norm=evoked_lcmv_weight_norm,
                                n_layers=n_layers))
                        
                                if subject_counter == 1:
                                    grand_average = lcmv.copy()
                                    grand_average._data = abs(lcmv.data)
                                else:
                                    grand_average._data += abs(lcmv.data)
                            grand_average._data /= subject_counter # mean
                            grand_average.save(output_name, ftype='h5',
                                               overwrite=overwrite)
                        
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                         submitting_method)      