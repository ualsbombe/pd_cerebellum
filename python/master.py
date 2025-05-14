#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:37:13 2021

@author: lau
"""

#%% IMPORTS

from helper_functions import submit_job
from config import recordings

#%% RECORDINGS

subjects_to_run = [
    # '0002',
    # '0003',
    # '0004',
    # '0005',
    # '0006',
    # '0007',
    # '0008',
    # '0009',
    # '0010',
    # '0011',
    # '0012',
    # '0013',
    # '0014',
    # '0015',
    # '0016',
    # '0017',
    # '0018',
    # '0019',
    # '0020',
    # '0021',
    # '0022',
    # '0023',
    # '0024',
    # '0025',
    # '0026',
    # '0027',
    # '0028',
    # '0029',
    # '0030',
    # '0031',
    # '0032',
    # '0033',
    # '0034',
    # '0035',
    # '0036',
    # '0037',
    # '0038',
    # '0039',
    # '0040',
    # '0041',
    # '0042',
    # '0043',
    # '0044',
    # '0045',
    # '0046',
    # '0047',
    # '0048',
    # '0049',
    # '0050',
    # '0051',
    # '0052',
    # '0053',
    # '0054',
    # '0055',
    # '0056',
    # '0057',
    
    # 'fsaverage'
# 
    ]

functions = [
               ## GENERAL
                # 'analysis_00_create_folders',
                # 'analysis_01_find_events',
    
               ## GENERAL PLOTTING
                # 'analysis_plot_00_power_spectra',
               
               # ## EVOKED ANALYSIS
                # 'analysis_evoked_00_filter',
                # 'analysis_evoked_01_epochs',
                # 'analysis_evoked_02_average',
                # 'analysis_evoked_03_grand_average',
            
              
               ## HILBERT ANALYSIS
                # 'analysis_hilbert_00_filter',
                # 'analysis_hilbert_01_epochs',
                # 'analysis_hilbert_02_average',
                # 'analysis_hilbert_03_grand_average',
                
                
               ## ANATOMY PROCESSING - FREESURFER
                # 'analysis_anatomy_freesurfer_00_segmentation',
                # 'analysis_anatomy_freesurfer_01_bem',
                # 'analysis_anatomy_freesurfer_02_forward_model',
            
                  
               ## SOURCE EVOKED
                # 'analysis_source_evoked_00_beamformer',
                # 'analysis_source_evoked_01_morph_beamformer',
                # 'analysis_source_evoked_02_beamformer_grand_average',
                

               ## SOURCE HILBERT
                # 'analysis_source_hilbert_00_beamformer',
                # 'analysis_source_hilbert_01_morph_beamformer',
                # 'analysis_source_hilbert_02_beamformer_grand_average',
                # 'analysis_source_hilbert_03_envelope_correlation',
                
            ]


#%% RUN ALL JOBS

for function in functions:
    for recording in recordings:
        if recording['subject'] in subjects_to_run:
            submit_job(recording, function, overwrite=False)
