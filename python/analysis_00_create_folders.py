#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:53:33 2021

@author: lau
"""

from config import fname, submitting_method, hyades_parameters
from helper_functions import set_hyades_parameters
from os import makedirs
from sys import argv


def this_function(subject, date, overwrite):
    exist_ok=True

    makedirs(fname.subject_path(subject=subject,
                                date=date), exist_ok=exist_ok)
    makedirs(fname.subject_figure_path(subject=subject,
                                date=date), exist_ok=exist_ok)
    makedirs(fname.subject_beamformer_hilbert_path(subject=subject, date=date),
                                                    exist_ok=exist_ok)
    makedirs(fname.subject_beamformer_evoked_path(subject=subject, date=date),
                                                     exist_ok=exist_ok)
    makedirs(fname.subject_beamformer_hilbert_labels_path(subject=subject,
                                                          date=date),
                                                          exist_ok=exist_ok)
    makedirs(fname.subject_simnibs_path(subject=subject, date=date),
             exist_ok=exist_ok)
    makedirs(fname.subject_envelope_path(subject=subject, date=date),
             exist_ok=exist_ok)
    


set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)