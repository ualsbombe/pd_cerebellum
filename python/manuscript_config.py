#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:35:03 2025

@author: lau
"""

#%% IMPORTS

from config import fname
from os.path import join
import numpy as np

#%% GENERAL

fsaverage = 'fsaverage'
fsaverage_date = '20230629_000000'

figure_path = fname.subject_figure_path(subject=fsaverage, date=fsaverage_date) + '/playaround'

default_rc_params = dict()
    
default_rc_params['font.size'] = 14
default_rc_params['font.weight'] = 'bold'
default_rc_params['lines.linewidth'] = 1.5
default_rc_params['axes.facecolor'] = 'black'

p_col = 'red'
c_col = 'yellow'
coll_col = 'green'
fit_col = 'orange'
if default_rc_params['axes.facecolor'] == 'white':
    zero_col = 'black'
elif default_rc_params['axes.facecolor'] == 'black':
        zero_col = 'white'
        
colormap ='YlOrRd'        


dpi = 300

tmin = -0.100
tmax = 0.100

t_alpha = 0.05

## general
o_combination = dict(contrast=[
    dict(event='o0', first_event='o0', second_event='o15'),
    dict(event='o15', first_event='o0', second_event='o15'),
        ])

vertex_input_name = fname.source_hilbert_beamformer_grand_average(
    subject=fsaverage, date=fsaverage_date, Type='collapsed',
    fmin=14, fmax=30,
    tmin=-0.750, tmax=0.750, reg=0.000, event='o0',
    first_event='o0', second_event='o15',
    weight_norm='unit-noise-gain-invariant', n_layers=1)

src_name = fname.anatomy_freesurfer_volumetric_source_space(
                                                   subject=fsaverage,
                                                   spacing=7.5)


#%% FIGURE - EXPERIMENTAL PARADIGM

""" 
created as illustation outside python 
"""

#%% FIGURE - SENSOR SPACE RESULTS

"""
    A: butterfly plots of magnetometers for HC
    B: butterfly plots of magnetometers for PD
    C: SI and SII topographies for HC
    D: SI and SII topographies for PD
"""

evoked_tmin = -0.100
evoked_tmax =  0.200
picks = 'mag'
butterfly_size = (7, 3)
topoplot_size  = (4, 4)
topomap_times = (0.053, 0.132)
topomap_vlim =  (-40, 40) ## fT
topomap_text_coordinates = (-0.60, 0.45)

evoked_panel_A = join(figure_path, 'evoked_panel_A_butterfly_controls.png')
evoked_panel_B = join(figure_path, 'evoked_panel_B_butterfly_patients.png')
evoked_panel_C = join(figure_path, 'evoked_panel_C_topomap_controls.png')
evoked_panel_D = join(figure_path, 'evoked_panel_D_topomap_patients.png')

#%% FIGURE - PRESENT F AND t RESULTS

"""
    A: basal ganglia (-14 ms) (F)
    B: cerebellum (-23 ms) (F)
    C: basal ganglia PD (t)
    D: cerebellum PD (t)
    E: basal ganglia HC (t)
    F: cerebellum HC (t)
"""


## panel A
rc_params_F_t = dict()    
rc_params_F_t['font.size'] = 4


F_alpha = 0.005
caudate_time = -0.014 # s
caudate_pos  = (-0.013, 0.015, 0.010) # m
cb_time      = -0.023 # s
cb_pos       = (-0.052, -0.066, -0.034) #

F_t_panel_A = join(figure_path,'F_t_panel_A_Caudate_L_Cerebellum_Crus1_L.png')

## panel B

rc_params_t = dict()    
rc_params_t['font.size'] = 14

F_t_panel_B = join(figure_path,'F_t_panel_B_Caudate_L.png')


## panel C

F_t_panel_C = join(figure_path,'F_t_panel_C_Cerebellum_Crus1_L.png')

## panel D

F_t_panel_D = join(figure_path,'F_t_panel_D_Cerebellum_Crus1_L_F.png')

## panel E

F_t_panel_E = join(figure_path,'F_t_panel_E_Caudate_L_F.png')




#%% FIGURE - REPLICATION OF 44 MS RESULTS (t)

"""
    cerebellum (44 ms) (t)
"""
rc_params_cb_replication = dict()    
rc_params_cb_replication['font.size'] = 4
cb_replication_time = 0.044
cb_replication_pos = (-0.022, -0.047, -0.028)
cb_replication_panel_A = join(figure_path, 'cb_rep_panel_A_glass_brain.png')

cb_replication_panel_B = join(figure_path, 'cb_rep_panel_B_Cerebellum_4_5.png')


#%% FIGURE - CORRELATION SIGNIFICANT F AND UPDRS

"""
    A: cerebellum correlation
    B: basal ganglia correlation
    
"""

rc_params_corr = dict()    
rc_params_corr['font.size'] = 20
rc_params_corr['lines.linewidth'] = 5


# caudate = 6633, CB_early = 3270, peak = -23 ms, CB_late = 3964
caudate_vertex_indices = [6633] # gotten from F images below
cb_vertex_indices = [3270, 2649]  # early peak 3293, 2672, # on peak 3270, 2649: later peak, 3891
# source_indices = np.array(vertex_indices['Caudate_L'])
#-55 - 2 ms
cb_time_indices = np.arange(727, 728 ) # 727 is the peak for vertex 3270
caudate_time_indices = np.arange(736, 737)

corr_panel_A = join(figure_path, 'corr_panel_A_Cerebellum_Crus1_L.png')
corr_panel_B = join(figure_path, 'corr_panel_B_Caudate_L.png')


#%% FIGURE - ENVELOPE ANALYSIS

"""
    cerebello-thalamo-caudate correlation
"""

env_tmin = -0.100
env_tmax =  0.100
env_fmin = 14
env_fmax = 30
n_lines = 7


env_panel_A = join(figure_path, 'env_panel_A_conn_circle.png')

env_panel_B = join(figure_path, 'env_panel_B_anova.png')

env_supp_panel_A = join(figure_path, 'env_supp_panel_A_conn_circle.png')
