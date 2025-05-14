#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:00:03 2021

@author: lau
"""

from config import (fname, submitting_method, bem_conductivities,
                    src_spacing, split_recording_subjects,
                    morph_subject_to, hyades_parameters)
from sys import argv
from helper_functions import (should_we_run, read_split_raw_info,
                              set_hyades_parameters)
from os.path import exists

import mne

def this_function(subject, date, overwrite):
    
    for bem_conductivity in bem_conductivities:
        n_layers = len(bem_conductivity)
        output_names = list()
       
        output_names.append(fname.anatomy_freesurfer_forward_model(
                        subject=subject, date=date, spacing=src_spacing,
                        n_layers=n_layers))
    
        for output_name in output_names:
            if should_we_run(output_name, overwrite):
                subjects_dir = fname.freesurfer_subjects_dir

                if subject in split_recording_subjects:
                    info = read_split_raw_info(subject, date)
                else:
                    info = mne.io.read_info(fname.raw_file(subject=subject,
                                                           date=date))
                trans = fname.anatomy_transformation(subject=subject,
                                                     date=date)
                
                bem = fname.anatomy_freesurfer_bem_solutions(subject=subject,
                                                          n_layers=n_layers)
                
                src_path = fname.anatomy_freesurfer_volumetric_source_space(
                            subject=subject, spacing=src_spacing)
                if not exists(src_path) or overwrite:
                    src = mne.source_space.setup_volume_source_space(subject,
                                                             pos=src_spacing,
                                                                 bem=bem,
                                                    subjects_dir=subjects_dir)
                    mne.write_source_spaces(src_path, src, overwrite=overwrite)
                else:
                    src = src_path
                
                        
                fwd = mne.make_forward_solution(info, trans, src, bem)
                mne.write_forward_solution(output_name, fwd, overwrite)

                ## morph to fsaverage
                output_name = fname.anatomy_freesurfer_morph_volume(
                    subject=subject,
                    spacing=src_spacing)
                if should_we_run(output_name, overwrite):
                    if subject != 'fsaverage':
                        fwd = mne.read_forward_solution(
                            fname.anatomy_freesurfer_forward_model(
                                subject=subject,
                                date=date,
                                spacing=src_spacing,
                                n_layers=n_layers))
                        # note: freesurfer is correct two lines below
                        src_to_path = \
                            fname.anatomy_freesurfer_volumetric_source_space(
                                subject=morph_subject_to,
                                spacing=src_spacing)
                        src_to = mne.read_source_spaces(src_to_path)
                        morph = mne.compute_source_morph(fwd['src'],
                                                subject,
                                                subjects_dir=subjects_dir,
                                                src_to=src_to)
                        morph.save(output_name, overwrite=overwrite)
        
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                         submitting_method)     