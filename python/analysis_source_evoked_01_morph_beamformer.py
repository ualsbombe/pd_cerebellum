#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:34:07 2022

@author: lau
"""


from config import (fname, submitting_method, src_spacing,
                    evoked_lcmv_contrasts, evoked_lcmv_regularization,
                    evoked_tmin, evoked_tmax, evoked_fmin, evoked_fmax,
                    evoked_lcmv_weight_norms, bem_conductivities,
                    hyades_parameters, subjects_no_watershed)
from sys import argv
from helper_functions import should_we_run, set_hyades_parameters

import mne

def this_function(subject, date, overwrite):
    morph_name = fname.anatomy_freesurfer_morph_volume(subject=subject,
                                            spacing=src_spacing)
    morph = mne.read_source_morph(morph_name)
    for this_contrast in evoked_lcmv_contrasts:
    
        for evoked_lcmv_weight_norm in evoked_lcmv_weight_norms:
            
            for bem_conductivity in bem_conductivities:
                n_layers = len(bem_conductivity)
                if subject in subjects_no_watershed[str(n_layers) + '_layer']:
                    continue
                
                input_names = list()
                output_names = list()
                for event in this_contrast:
                    input_names.append(fname.source_evoked_beamformer(
                                            subject=subject,
                                            date=date,
                                            fmin=evoked_fmin,
                                            fmax=evoked_fmax,
                                            tmin=evoked_tmin,
                                            tmax=evoked_tmax,
                                            event=event,
                                            first_event=this_contrast[0],
                                            second_event=this_contrast[1],
                                            reg=evoked_lcmv_regularization,
                                            weight_norm=evoked_lcmv_weight_norm,
                                            n_layers=n_layers))
                

                    
                    output_names.append(fname.source_evoked_beamformer_morph(
                                            subject=subject,
                                            date=date,
                                            fmin=evoked_fmin,
                                            fmax=evoked_fmax,
                                            tmin=evoked_tmin,
                                            tmax=evoked_tmax,
                                            event=event, # first event
                                            reg=evoked_lcmv_regularization,
                                            first_event=this_contrast[0],
                                            second_event=this_contrast[1],
                                            weight_norm=evoked_lcmv_weight_norm,
                                            n_layers=n_layers))
             
            
                for name_index, output_name in enumerate(output_names):
                    if should_we_run(output_name, overwrite):
                        if morph.vol_morph_mat is None: # only compute once
                            morph.compute_vol_morph_mat()
                        print(output_name)
                        stc = mne.read_source_estimate(input_names[name_index])
                        stc_morph = morph.apply(stc)
                        stc_morph.save(output_name, ftype='h5',
                                       overwrite=overwrite)
                        
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                         submitting_method)              