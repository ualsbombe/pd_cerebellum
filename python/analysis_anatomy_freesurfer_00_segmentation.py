#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:26:46 2021

@author: lau
"""

from config import (fname, submitting_method, t1_file_ending,
                    subjects_with_MRs_from_elsewhere, hyades_parameters)
from os import listdir
from os.path import join, isdir
from sys import argv
import mne

from helper_functions import (run_process_and_write_output,
                              set_hyades_parameters)

def this_function(subject, date, overwrite):
    if subject in subjects_with_MRs_from_elsewhere:
        mr_path = fname.subject_MR_elsewhere_path(subject=subject, date=date)
    else:
        mr_path = fname.subject_MR_path(subject=subject, date=date)
    directories = listdir(mr_path)
    subjects_dir = fname.freesurfer_subjects_dir
    
    for directory in directories:
        if t1_file_ending in directory:
            break ## directory becomes the wanted directory
            
    image_path = join(mr_path, directory, 'files')
    image_filename = listdir(image_path)[0]
    full_path = join(image_path, image_filename)
    
    freesurfer_path = fname.subject_freesurfer_path(subject=subject)
    
    ## IMPORT MRI
    if not isdir(freesurfer_path) or overwrite:
       command = [
                   'recon-all',
                   '-subjid', subject,
                   '-i', full_path
                   ]
       run_process_and_write_output(command, subjects_dir)
       
       ## RECONSTRUCT
       command = [
                  'recon-all',
                  '-subjid', subject,
                  '-all'
                 ]
       
       run_process_and_write_output(command, subjects_dir)
       
       ## WATERSHED
       mne.bem.make_watershed_bem(subject, subjects_dir, overwrite=overwrite)
       
       ## MAKE DENSE SCALP SURFACES
       if overwrite:
           overwrite_string = '--overwrite'
       else:
           overwrite_string = ''
           
       command = [
                   'mne_make_scalp_surfaces',
                   '--subject', subject,
                   overwrite_string
                   ]
       run_process_and_write_output(command, subjects_dir)
      
       
set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method) 
    