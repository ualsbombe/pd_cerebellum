#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:02:13 2021

@author: lau
"""

from config import (fname, submitting_method, hilbert_lcmv_contrasts,
                    hilbert_lcmv_weight_norms, hilbert_lcmv_regularization,
                    hilbert_lcmv_picks, hilbert_fmins, hilbert_fmaxs,
                    hilbert_tmin, hilbert_tmax, bad_channels,
                    src_spacing, bem_conductivities, hyades_parameters,
                    subjects_no_watershed)
from sys import argv
from helper_functions import should_we_run, set_hyades_parameters

import mne
import numpy as np

# hilbert_lcmv_contrasts = hilbert_lcmv_contrasts[-1:]

def this_function(subject, date, overwrite):
    
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        raw_loaded = False
        for this_contrast in hilbert_lcmv_contrasts:
            for hilbert_lcmv_weight_norm in hilbert_lcmv_weight_norms:
                if hilbert_lcmv_weight_norm == 'unit-gain':
                    weight_norm = None
                else:
                    weight_norm = hilbert_lcmv_weight_norm
                for bem_conductivity in bem_conductivities:
                    n_layers = len(bem_conductivity)
                    if subject in \
                        subjects_no_watershed[str(n_layers) + '_layer']:
                        continue
                    output_names = list()
                    for event in this_contrast:
                        output_names.append(
                    fname.source_hilbert_beamformer(
                                            subject=subject,
                                            date=date,
                                            fmin=fmin,
                                            fmax=fmax,
                                            tmin=hilbert_tmin,
                                            tmax=hilbert_tmax,
                                            event=event,
                                            first_event=this_contrast[0],
                                            second_event=this_contrast[1],
                                            reg=hilbert_lcmv_regularization,
                                        weight_norm=hilbert_lcmv_weight_norm,
                                            n_layers=n_layers))
              
     
                    if should_we_run(output_names[0], overwrite) or \
                       should_we_run(output_names[1], overwrite):
                        if not raw_loaded:
                            raw = \
                            mne.io.read_raw_fif(fname.hilbert_filter(
                                subject=subject,
                                date=date,
                                fmin=fmin,
                                fmax=fmax),
                                preload=False)
                            raw_loaded = True
                        epochs_hilbert = \
                        mne.read_epochs(fname.hilbert_epochs(
                            subject=subject,
                            date=date,
                            fmin=fmin,
                            fmax=fmax,
                            tmin=hilbert_tmin,
                            tmax=hilbert_tmax),
                                            proj=False, preload=False)
                        ## apply bads
                        raw.info['bads'] = bad_channels[subject]
                        epochs_hilbert.info['bads'] = bad_channels[subject]
                        
                        picks = mne.pick_types(epochs_hilbert.info,
                                               meg=hilbert_lcmv_picks)
                        
                            
                        events = epochs_hilbert.events
                        baseline = epochs_hilbert.baseline
                        event_ids = epochs_hilbert.event_id
                        
                        ## only look at contrast
                        new_event_id = dict()
                        for event in event_ids:
                            if event in this_contrast:
                                new_event_id[event] = event_ids[event]
                                
                        epochs_cov = mne.Epochs(raw, events, new_event_id,
                                                hilbert_tmin, hilbert_tmax,
                                                baseline,
                                                proj=False,
                                                preload=True,
                                                picks=picks)
                        
                        ## remove projs
                        epochs_hilbert.del_proj()
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
                        
                        fwd = mne.make_forward_solution(
                            epochs_cov.info, trans,
                            src, bem)
                            
                        
                        if baseline is None:
                            data_cov = mne.compute_covariance(epochs_cov,
                                                              tmin=None,
                                                              tmax=None,
                                                              rank=rank)
                        else:
                            raise RuntimeError('"baseline" ' + \
                                               str(baseline) + \
                                               ' not implemented')    
                            
                        filters = mne.beamformer.make_lcmv(epochs_cov.info, 
                                         fwd,
                                         data_cov,
                                         pick_ori='max-power',
                                        weight_norm=weight_norm,
                                        reg=hilbert_lcmv_regularization)
                        del epochs_cov ## release memory
                        
                        print('Reconstructing events in contrast: ' + \
                              str(this_contrast))
                                                
                        for event in this_contrast:
                            these_epochs = epochs_hilbert[event]
                            these_epochs.load_data()
                            these_epochs.pick(picks)
                            
                            stcs = mne.beamformer.apply_lcmv_epochs(
                                these_epochs,
                                filters)
                            for stc in stcs:
                                stc._data = np.array(np.abs(stc.data),
                                                     dtype='float64')
                            
                            stc_mean = stcs[0].copy()
                            mean_data = np.mean([stc.data for stc in stcs],
                                                axis=0)
                            stc_mean._data = mean_data
        
                            stc_mean.save(
                            fname.source_hilbert_beamformer(
                                      subject=subject,
                                      date=date,
                                      fmin=fmin,
                                      fmax=fmax,
                                      tmin=hilbert_tmin,
                                      tmax=hilbert_tmax,
                                      event=event,
                                      reg=hilbert_lcmv_regularization,
                                      first_event=this_contrast[0],
                                      second_event=this_contrast[1],
                                weight_norm=hilbert_lcmv_weight_norm,
                                      n_layers=n_layers),
                            overwrite=True)
        
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                         submitting_method)      