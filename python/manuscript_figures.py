#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:18:21 2025

@author: lau
"""

#%% IMPORTS


#%% THE ONE CLASS TO RULE THEM ALL

class manuscript_figure(object):
    
    ## general
    
    def __init__(self, o_beta=None, src=None, envelope=None, ga_evoked=None):
        
        self.o_beta    = o_beta
        self.src       = src
        self.envelope  = envelope
        self.ga_evoked = ga_evoked
        
    def load_ga_evoked(self):
        print('Loading Grand Averages')
        
        if self.ga_evoked is None:
            
            from manuscript_helper_functions import load_data_evoked
            
            evokeds_control, evokeds_patient = load_data_evoked()
            
            self.ga_evoked = dict()
            self.ga_evoked['control'] = evokeds_control
            self.ga_evoked['patient'] = evokeds_patient
    
    def load_o_beta(self):
        print('Loading hilbert data')
        from config import recordings, bad_subjects

        if self.o_beta is None: ## load if not supplied as argument
            ## imports
            
            from manuscript_helper_functions import load_data_hilbert
            
            ## load omission (14-30 Hz)
            
            from manuscript_config import o_combination
            self.o_beta = load_data_hilbert(o_combination, recordings, bad_subjects,
                                       fmin=14, fmax=30)
    
    def load_src(self):
        print('Loading source space')
        if self.src is None: ## load if not supplied as argument
            ## load src
            from manuscript_config import src_name
            from mne import read_source_spaces
            self.src = read_source_spaces(src_name)
            
    
    def load_envelope(self):
        print('Getting envelopes')

        from manuscript_config import env_tmin, env_tmax, env_fmin, env_fmax
        from config import recordings,bad_subjects
        import numpy as np
        from manuscript_helper_functions import get_envelope_data
        
        n_sources = len(self.src[0]['vertno'])
        n_subjects = len(recordings)
    
        data = np.zeros(shape=(n_subjects, n_sources, n_sources))
    
    
        events = ['o0', 'o15']
        self.envelope = dict()
        for event in events: 
    
            this_data, these_indices = get_envelope_data(recordings,
                                                         bad_subjects,
                                                       env_fmin,
                                                       env_fmax,
                                                       env_tmin,
                                                       env_tmax,
                                                       event, data)
            
            self.envelope[event] = this_data[these_indices, :, :]            

    
    def load_stc_vertex(self):
        print('Loading stc container')
        from mne import read_source_estimate
        from manuscript_config import vertex_input_name
        self.stc_vertex = read_source_estimate(vertex_input_name)
        
    
    def get_subject_indices(self):
        print('Getting indices for subjects')
        from config import recordings, bad_subjects
        from manuscript_helper_functions import get_group_indices

        self.patient_indices, self.control_indices = get_group_indices(recordings, 
                                                             bad_subjects)
        
    
    
    def get_label_vertices(self, labels):
        print('Getting vertices for labels')
        from manuscript_helper_functions import get_label_vertices_AAL
        self.stc_indices, self.src_indices = get_label_vertices_AAL(labels,
                                                                    self.src)
        

        
    ## Evoked figure
    
    
    def plot_butterfly_and_topomap(self):
        from manuscript_config import (default_rc_params, evoked_tmin,
                                       evoked_tmax, picks, butterfly_size,
                                       topomap_times, topomap_vlim,
                                       topomap_text_coordinates,
                                       topoplot_size,
                                       evoked_panel_A, evoked_panel_B,
                                       evoked_panel_C, evoked_panel_D,
                                       dpi)
        from manuscript_helper_functions import set_rc_params
        
        import matplotlib.pyplot as plt
        
        set_rc_params(default_rc_params=default_rc_params)
        
        
        
        """butterfly control"""
        fig_c = self.ga_evoked['control'][0].plot(picks=picks,
                                xlim=(evoked_tmin, evoked_tmax),
                                ylim=dict(mag=(-40, 40)),
                                titles=dict(mag='Healthy Controls:\n magnetometers'))
        
        fig_c.set_size_inches(butterfly_size)
        
        fig_c.savefig(evoked_panel_A, dpi=dpi)
        
        """butterfly patient"""
        fig_p = self.ga_evoked['patient'][0].plot(picks=picks,
                                xlim=(evoked_tmin, evoked_tmax),
                                ylim=dict(mag=(-70, 70)),
                                titles=dict(mag="Parkinson's Disease:\n magnetometers"))
        
        fig_p.set_size_inches(butterfly_size)
        
        fig_p.savefig(evoked_panel_B, dpi=dpi)
        
        
        """topomap control"""
        fig_ctopo = self.ga_evoked['control'][0].plot_topomap(
            times=topomap_times, vlim=topomap_vlim, colorbar=True)
          
        
        plt.text(topomap_text_coordinates[0], topomap_text_coordinates[1],
                 'Healthy Controls', fontsize=default_rc_params['font.size'])
        
        fig_ctopo.set_size_inches(topoplot_size)

        
        fig_ctopo.savefig(evoked_panel_C, dpi=dpi)
        
        
        """topomap patient"""
        
        fig_ptopo = self.ga_evoked['patient'][0].plot_topomap(
            times=topomap_times, vlim=topomap_vlim)
        
        plt.text(topomap_text_coordinates[0], topomap_text_coordinates[1],
                         "Parkinson's Disease", fontsize=default_rc_params['font.size'])
        
        fig_ptopo.set_size_inches(topoplot_size)

        
        fig_ptopo.savefig(evoked_panel_D, dpi=dpi)
    
    
    ## F_t figure
        
    
    def get_Fs(self):
        print('Getting F-values')
        from manuscript_helper_functions import get_F
        from manuscript_config import F_alpha, tmin, tmax
        from scipy.stats.distributions import f
        
        F, df_num, df_denom = get_F(self.o_beta, self.control_indices,
                                    self.patient_indices)
        
        F_stc = self.stc_vertex.copy()
        F_stc._data = F
        F_stc.crop(tmin, tmax)
        F_crit = f.ppf(1-F_alpha, df_num, df_denom)
        
        self.F_stc = F_stc
        self.F_crit = F_crit
        
    
    def plot_F_t_glass_brain(self):
        ## make plots
        
        from manuscript_helper_functions import plot_F, set_rc_params
        from manuscript_config import (caudate_time, caudate_pos,
                                       rc_params_F_t, default_rc_params)
        
        set_rc_params(font_size=rc_params_F_t['font.size'],
                      default_rc_params=default_rc_params)
        
        self.caudate_cb = plot_F(self.F_stc, self.src, caudate_time,
                                 caudate_pos, self.F_crit)
        
    
    def plot_F_vertices(self, label):
        
        from manuscript_helper_functions import plot_F_vertices, set_rc_params
        from manuscript_config import default_rc_params, tmin, tmax
        
        set_rc_params(default_rc_params=default_rc_params)

        F_vertices = plot_F_vertices(label, self.F_stc,
                                                self.stc_indices, tmin, tmax,
                                                self.F_crit)
        
        ## save panel
        if label == 'Cerebelum_Crus1_L':
            from manuscript_config import F_t_panel_D, dpi
            F_vertices.savefig(F_t_panel_D, dpi=dpi)
        
        elif label == 'Caudate_L':
            from manuscript_config import F_t_panel_E, dpi
            F_vertices.savefig(F_t_panel_E, dpi=dpi)
    
    
    def save_F_t_glass_brain(self):
        from manuscript_config import F_t_panel_A
        from manuscript_helper_functions import save_T1_plot_only_hilbert
        
        save_T1_plot_only_hilbert(self.caudate_cb, F_t_panel_A)
        
    
    def get_t_by_indices(self):
        
        print('Getting t by group indices')
        from manuscript_helper_functions import get_t, get_t_stc
        from manuscript_config import tmin, tmax
       
        self.t_patient, self.df_patient = \
            get_t(self.o_beta[self.patient_indices, :, :])
        self.patient_t_stc = get_t_stc(self.t_patient, self.stc_vertex, tmin,
                                       tmax)
        
        self.t_control, self.df_control = \
            get_t(self.o_beta[self.control_indices, :, :])
        self.control_t_stc = get_t_stc(self.t_control, self.stc_vertex, tmin,
                                       tmax)
    

    def plot_t_vertices_by_index_and_save(self, label):
        print('Plotting and saving t by group indices')
        from manuscript_helper_functions import set_rc_params, plot_t_vertices
        from manuscript_config import default_rc_params, tmin, tmax, dpi
        set_rc_params(default_rc_params=default_rc_params)

        t_plot = plot_t_vertices(label, self.patient_t_stc,
                                    self.control_t_stc,
                           self.stc_indices, tmin, tmax)
        ## save panel
        if label == 'Cerebelum_Crus1_L':
            from manuscript_config import F_t_panel_C
            t_plot.savefig(F_t_panel_C, dpi=dpi)
        
        elif label == 'Caudate_L':
            from manuscript_config import F_t_panel_B, dpi
            t_plot.savefig(F_t_panel_B, dpi=dpi)
            
        return t_plot
    
    
## REPLICATION FIGURE

    def get_t_collapsed(self):
        print('Getting t collapsed')

        from manuscript_helper_functions import get_t
        from manuscript_config import tmin, tmax, t_alpha
        from scipy import stats

        self.t, self.df = get_t(self.o_beta)
        self.t_stc = self.stc_vertex.copy()
        self.t_stc._data = self.t
        
        self.t_crit = stats.t.ppf(1 - (t_alpha/2), self.df)


        self.t_stc.crop(tmin, tmax)
        

    def plot_t_mni_and_save(self):        
        from manuscript_config import (default_rc_params,
                                       rc_params_cb_replication,
                                       cb_replication_pos, cb_replication_time)
        from manuscript_helper_functions import (set_rc_params,
                                                 plot_t_glass_brain,
                                                 save_T1_plot_only_hilbert)
        
        set_rc_params(font_size=rc_params_cb_replication['font.size'],
                      default_rc_params=default_rc_params)
        
        from manuscript_config import cb_replication_panel_A

        
        cb_replication = plot_t_glass_brain(self.t_stc, self.src,
                                            cb_replication_time,
                                            cb_replication_pos, self.t_crit)
        
        save_T1_plot_only_hilbert(cb_replication, cb_replication_panel_A)
        
    def plot_t_vertices_and_save(self, label):
        print('Plotting and saving t collapsed')

        
        from manuscript_config import default_rc_params, tmin, tmax
        from manuscript_helper_functions import (plot_t_vertices_collapsed,
                                                 set_rc_params)
        set_rc_params(default_rc_params=default_rc_params)

        cb_rep_t = plot_t_vertices_collapsed(label, self.t_stc,
                           self.stc_indices, tmin, tmax, self.t_crit)
        
        from manuscript_config import cb_replication_panel_B, dpi
        cb_rep_t.savefig(cb_replication_panel_B, dpi=dpi)
        
        
## CORR FIGURE

    def get_data_and_updrs(self):
        
        print('Getting corr data and UPDRS')

        
        from config import recordings, bad_subjects
        from manuscript_config import (cb_vertex_indices,
                                       caudate_vertex_indices,
                                       cb_time_indices, caudate_time_indices)
        from manuscript_helper_functions import get_corr_data, get_updrs
        
        self.cb_corr = get_corr_data(self.o_beta, cb_vertex_indices,
                                cb_time_indices,
                                self.patient_indices, self.stc_vertex)
        self.caudate_corr = get_corr_data(self.o_beta, caudate_vertex_indices,
                                     caudate_time_indices, self.patient_indices,
                                     self.stc_vertex)
        self.updrs = get_updrs(recordings, bad_subjects)
        

    def plot_corr_and_save(self, label):
        
        print('Plotting and saving correlations')

        
        from manuscript_helper_functions import plot_corr, set_rc_params
        from manuscript_config import default_rc_params, rc_params_corr
        from manuscript_config import dpi, corr_panel_A, corr_panel_B

        
        set_rc_params(font_size=rc_params_corr['font.size'],
                      line_width=rc_params_corr['lines.linewidth'],
                      default_rc_params=default_rc_params)
        if label == 'Cerebelum_Crus1_L':
            corr = plot_corr(self.cb_corr, self.updrs, label)
            corr.savefig(corr_panel_A, dpi=dpi)
        elif label == 'Caudate_L':
            corr = plot_corr(self.caudate_corr, self.updrs, label)
            corr.savefig(corr_panel_B, dpi=dpi)
            
            
### ENVELOPE FIGURE

    def get_label_envelopes(self, seeds):
        
        print('Get envelopes per group per ROI')
        
        from manuscript_helper_functions import get_label_envelope
        self.envelope_by_indices = dict()
        for seed in seeds:
            print('Getting seed: ' + seed)
            self.envelope_by_indices[seed] = dict()
            self.envelope_by_indices[seed]['control'] = get_label_envelope(
                self.envelope['o0'][self.control_indices, :, :],
                self.envelope['o15'][self.control_indices, :, :],
                seed, self.stc_indices)
            self.envelope_by_indices[seed]['patient'] = get_label_envelope(
                self.envelope['o0'][self.patient_indices, :, :],
                self.envelope['o15'][self.patient_indices, :, :],
                seed, self.stc_indices)
            
    def plot_conn_circle(self, node_names, supplementary=False):
        
        from manuscript_helper_functions import plot_conn
        from manuscript_config import (default_rc_params, dpi, env_panel_A,
                                       env_supp_panel_A)
        from manuscript_helper_functions import set_rc_params
        
        set_rc_params(default_rc_params=default_rc_params)
        
        if not supplementary:
            
            from manuscript_config import n_lines
        
            self.env_circle, ax = plot_conn(self.envelope_by_indices, 
                                                 node_names, n_lines)
    
            
            self.env_circle.savefig(env_panel_A, dpi=dpi)
        else:
            self.supp_circle, ax = plot_conn(self.envelope_by_indices,
                                             node_names, n_lines=None)
            
            self.supp_circle.savefig(env_supp_panel_A, dpi=dpi)
        
    
    def do_and_plot_anova(self, seed, labels):
        
        print('Plot and do ANOVA')
        
        from manuscript_config import default_rc_params, dpi, env_panel_B
        from manuscript_helper_functions import do_anova, set_rc_params
        
        set_rc_params(default_rc_params=default_rc_params)
        
        env_plot = do_anova(self.envelope_by_indices[seed]['control'],
                            self.envelope_by_indices[seed]['patient'], labels)
        
        env_plot.savefig(env_panel_B, dpi=dpi)
        
        
#%% LOAD HEAVY STUFF

# Only load if not already done

if 'loader' not in globals():
    loader = manuscript_figure()
    loader.load_ga_evoked()
    loader.load_src()
    loader.load_envelope()
    loader.load_o_beta()
    o_beta_copy = loader.o_beta
    src_copy = loader.src
    envelope_copy = loader.envelope
    
# else:
#     loader = manuscript_figure(o_beta=o_beta, src=src)

        

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


evoked = manuscript_figure(ga_evoked=loader.ga_evoked)
evoked.plot_butterfly_and_topomap()
        
# %% FIGURE - PRESENT F AND t RESULTS

"""
    A: basal ganglia (-14 ms) (F) and cerebellum (-23 ms) (F)
    B: basal ganglia PD (t) and basal ganglia HC (t)
    C: cerebellum PD (t) and cerebellum HC (t)
    D: basal ganglia F time course
    E: cerebellar F time course
"""

F_t = manuscript_figure(loader.o_beta, loader.src)
F_t.load_stc_vertex()
F_t.get_subject_indices()
F_t.get_label_vertices(labels=['Caudate_L', 'Cerebelum_Crus1_L'])
F_t.get_Fs()
F_t.plot_F_t_glass_brain()
# panel A
F_t.save_F_t_glass_brain()
F_t.get_t_by_indices()
# panel B
F_t.plot_t_vertices_by_index_and_save('Cerebelum_Crus1_L')
# panel C
F_t.plot_t_vertices_by_index_and_save('Caudate_L')
# panel D
F_t.plot_F_vertices('Cerebelum_Crus1_L')
# panel E
F_t.plot_F_vertices('Caudate_L')


#%% FIGURE - REPLICATION OF 44 MS RESULTS (t)

"""
    cerebellum (44 ms) (t)
    A: glass brain
    B_ time courses
"""

cb_replication = manuscript_figure(loader.o_beta, loader.src)
cb_replication.load_stc_vertex()
cb_replication.get_label_vertices(['Cerebelum_4_5_L'])
cb_replication.get_t_collapsed()
cb_replication.plot_t_mni_and_save()
cb_replication.plot_t_vertices_and_save(label='Cerebelum_4_5_L')

#%% FIGURE - CORRELATION SIGNIFICANT F AND UPDRS

"""
    A: cerebellum correlation
    B: basal ganglia correlation
"""

corr = manuscript_figure(loader.o_beta, loader.src)
corr.load_stc_vertex()
corr.get_subject_indices()
corr.get_data_and_updrs()
corr.plot_corr_and_save(label='Cerebelum_Crus1_L')
corr.plot_corr_and_save(label='Caudate_L')

#%% FIGURE - ENVELOPE ANALYSIS

"""
    cerebello-thalamo-caudate correlation
"""

labels = [
    'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R',
    'Caudate_L', 'Caudate_R',
    'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
    'Precentral_L', 'Precentral_R',
    'Thalamus_L', 'Thalamus_R',
                              ]


envelope = manuscript_figure(src=loader.src, envelope=loader.envelope)
envelope.get_label_vertices(labels)
envelope.get_subject_indices()
envelope.get_label_envelopes(seeds=labels)
envelope.plot_conn_circle(labels)
envelope.plot_conn_circle(labels, supplementary=True)
envelope.do_and_plot_anova('Caudate_L', labels)
