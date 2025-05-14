#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:19:53 2023

@author: lau
"""


from config import (fname, submitting_method, bem_conductivities, bem_ico,
                    hyades_parameters)
from sys import argv
import mne
from helper_functions import should_we_run, set_hyades_parameters

## simnibs/freesurfer commands
# meshfix
# mris_transform

def this_function(subject, date, overwrite):
    
    subjects_dir = fname.freesurfer_subjects_dir


    ## create bem_model
    
    
    for bem_conductivity in bem_conductivities:
        n_layers = len(bem_conductivity)
        output_name = fname.anatomy_freesurfer_bem_surfaces(subject=subject,
                                                         n_layers=n_layers)

        if should_we_run(output_name, overwrite):
            bem_surfaces = mne.bem.make_bem_model(
                        subject, ico=bem_ico,
                        conductivity=bem_conductivity,
                        subjects_dir=subjects_dir)
            mne.bem.write_bem_surfaces(output_name, bem_surfaces, overwrite)
        
        ## bem_solution 
        output_name = fname.anatomy_freesurfer_bem_solutions(subject=subject,
                                                          n_layers=n_layers)
        if should_we_run(output_name, overwrite):
            input_name = fname.anatomy_freesurfer_bem_surfaces(subject=subject,
                                                            n_layers=n_layers)
            bem_surfaces = mne.bem.read_bem_surfaces(input_name)
            bem_solution = mne.bem.make_bem_solution(bem_surfaces)
            mne.bem.write_bem_solution(output_name, bem_solution, overwrite)
            
            ## bem_solution 
        output_name = fname.anatomy_freesurfer_bem_solutions(subject=subject,
                                                          n_layers=n_layers)

            

set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)