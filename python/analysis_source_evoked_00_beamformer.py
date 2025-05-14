#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:28:54 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_lcmv_contrasts,
                    evoked_lcmv_weight_norms, evoked_lcmv_regularization,
                    evoked_lcmv_picks,
                    evoked_tmin, evoked_tmax,
                    evoked_fmin, evoked_fmax,
                    bad_channels, src_spacing,
                    bem_conductivities, hyades_parameters,
                    subjects_no_watershed)
from sys import argv
from helper_functions import should_we_run, set_hyades_parameters

import mne


def this_function(subject, date, overwrite):
    raw_loaded = False
    for this_contrast in evoked_lcmv_contrasts:
        for evoked_lcmv_weight_norm in evoked_lcmv_weight_norms:
            if evoked_lcmv_weight_norm == 'unit-gain':
                weight_norm = None
            else:
                weight_norm = evoked_lcmv_weight_norm
            for bem_conductivity in bem_conductivities:
                n_layers = len(bem_conductivity)
                if subject in subjects_no_watershed[str(n_layers) + '_layer']:
                    continue
                output_names = list()

                for event in this_contrast:
                    output_names.append(fname.source_evoked_beamformer(
                                        subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax,
                                        event=event,
                                        reg=evoked_lcmv_regularization,
                                        first_event=this_contrast[0],
                                        second_event=this_contrast[1],
                                        weight_norm=evoked_lcmv_weight_norm,
                                        n_layers=n_layers))
                
                if should_we_run(output_names[0], overwrite) or \
                   should_we_run(output_names[1], overwrite):
                    if not raw_loaded:
                        raw = \
                            mne.io.read_raw_fif(fname.evoked_filter(
                                subject=subject,
                                date=date,
                                fmin=evoked_fmin,
                                fmax=evoked_fmax),
                                preload=False)
                        raw_loaded = True

                    epochs = mne.read_epochs(fname.evoked_epochs(
                                    subject=subject,
                                    date=date,
                                    fmin=evoked_fmin,
                                    fmax=evoked_fmax,
                                    tmin=evoked_tmin,
                                    tmax=evoked_tmax),
                        proj=False, preload=False)
        
                    
                    ## apply bads
                    raw.info['bads'] = bad_channels[subject]
                    epochs.info['bads'] = bad_channels[subject]
                    
                    picks = mne.pick_types(epochs.info, 
                                           meg=evoked_lcmv_picks)
                    
                       
                    events = epochs.events
                    baseline = epochs.baseline
                    event_ids = epochs.event_id
                    
                    ## only look at contrast
                    new_event_id = dict()
                    for event in event_ids:
                        if event in this_contrast:
                            new_event_id[event] = event_ids[event]
                            
                    epochs_cov = mne.Epochs(raw, events, new_event_id,
                                            evoked_tmin, evoked_tmax,
                                            baseline,
                                            proj=False,
                                            preload=True,
                                            picks=picks)
                    
                    ## remove projs
                    epochs.del_proj()
                    epochs_cov.del_proj()
                    rank = None ## for computing covariance
                    
                    ## make forward model on the fly 
                    trans = fname.anatomy_transformation(
                        subject=subject,
                        date=date)
                    src = fname.anatomy_freesurfer_volumetric_source_space(
                                                subject=subject,
                                                spacing=src_spacing)
    
                    bem = fname.anatomy_freesurfer_bem_solutions(
                                                subject=subject,
                                                n_layers=n_layers)
                    
                    fwd = mne.make_forward_solution(epochs_cov.info,
                                                    trans,
                                                    src, bem)
                        
                    
                    data_cov = mne.compute_covariance(epochs_cov, tmin=0,
                                                      tmax=evoked_tmax,
                                                      rank=rank)
                    filters = mne.beamformer.make_lcmv(epochs_cov.info, fwd,
                                                       data_cov,
                                                       pick_ori='max-power',
                                           weight_norm=weight_norm,
                                           reg=evoked_lcmv_regularization)
                    del epochs_cov ## release memory
                    
                    print('Reconstructing events in contrast: ' + \
                          str(this_contrast))
                        
                    for event in this_contrast:
                        # these_epochs = epochs[event]
                        # these_epochs.load_data()
                        # these_epochs.pick(picks)
                        evoked = mne.read_evokeds(
                            fname.evoked_average_no_proj(
                            subject=subject, date=date,
                            fmin=evoked_fmin, fmax=evoked_fmax,
                            tmin=evoked_tmin, tmax=evoked_tmax),
                            proj=False,
                            condition=event)
                        # can't add eog and ecg - so remove these
                        bad_channels_evoked = bad_channels[subject].copy()
                        remove_these = ['EOG001', 'EOG002', 'ECG003',
                                        'EMG004', 'EMG005']
                        for remove_this in remove_these:
                            if remove_this in bad_channels_evoked:
                                bad_channels_evoked.remove(remove_this)
                        ## ended
                        evoked.info['bads'] = bad_channels_evoked
                        evoked.del_proj()
                        evoked.pick_types(meg=evoked_lcmv_picks)
                        
                        stc = mne.beamformer.apply_lcmv(evoked, filters)
                            
            
                        stc.save(fname.source_evoked_beamformer(
                                subject=subject,
                                date=date,
                                fmin=evoked_fmin,
                                fmax=evoked_fmax,
                                tmin=evoked_tmin,
                                tmax=evoked_tmax,
                                event=event,
                                reg=evoked_lcmv_regularization,
                                first_event=this_contrast[0],
                                second_event=this_contrast[1],
                                weight_norm=evoked_lcmv_weight_norm,
                                n_layers=n_layers),
                        overwrite=True)
                        
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                         submitting_method)                                                  