#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:25:09 2021

@author: lau
"""

from config import (fname, submitting_method,  split_recording_subjects,
                    n_jobs_power_spectra, hyades_parameters)
from sys import argv
from helper_functions import (should_we_run, read_split_raw,
                              set_hyades_parameters)

import mne


def this_function(subject, date, overwrite):
    output_name = fname.power_spectra_plot(subject=subject, date=date)
    
    if should_we_run(output_name, overwrite):
        if subject in split_recording_subjects:
            read_split_raw(subject, date)
        else:
            raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date))
        
        fig = raw.compute_psd(n_jobs=n_jobs_power_spectra).plot()
        fig.savefig(output_name)

set_hyades_parameters(hyades_parameters, this_function, argv, __file__,
                      submitting_method)