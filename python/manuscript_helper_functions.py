#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:34:48 2025

@author: lau
"""

#%% IMPORTS

from config import fname
from manuscript_config import default_rc_params
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


#%% GENERAL

def set_rc_params(font_size=None, font_weight=None, line_width=None,
                  background_color=None,
                  default_rc_params=default_rc_params):
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['interactive'] = True # otherwise mpl blocks the figures
    if font_size is None:
        font_size = default_rc_params['font.size']
    if font_weight is None:
        font_weight = default_rc_params['font.weight']
    if line_width is None:
        line_width = default_rc_params['lines.linewidth']
    if background_color is None:
        background_color = default_rc_params['axes.facecolor']
        
            
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.weight'] = font_weight
    mpl.rcParams['lines.linewidth'] = line_width
    mpl.rcParams['axes.facecolor'] = background_color
    mpl.rcParams['figure.facecolor'] = background_color
    if background_color == 'black':
       
        mpl.rcParams['axes.edgecolor'] = 'white'     # Axes border color
        mpl.rcParams['axes.labelcolor'] = 'white'    # Axis labels
        mpl.rcParams['xtick.color'] = 'white'        # X-axis tick labels
        mpl.rcParams['ytick.color'] = 'white'        # Y-axis tick labels
        mpl.rcParams['text.color'] = 'white'         # General text color
        mpl.rcParams['legend.edgecolor'] = 'white'  # Legend border
        mpl.rcParams['savefig.facecolor'] = 'black' # Background color for saved figures
        mpl.rcParams['savefig.edgecolor'] = 'black' 

def plot_legend(col_0, col_1, label_0, label_1):
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    
    # Create custom line objects for the legend
    red_line = mlines.Line2D([], [], color=col_0, label=label_0)
    blue_line = mlines.Line2D([], [], color=col_1, label=label_1)
    
    plt.legend(handles=[red_line, blue_line])
    
    
def load_data_evoked():
    
    
    from manuscript_config import fsaverage, fsaverage_date
    full_path_c = fname.evoked_grand_average_proj_interpolated(
                  subject=fsaverage, date=fsaverage_date, Type='control',
                  fmin=None, fmax=40, tmin=-0.200, tmax=0.400)
    full_path_p = fname.evoked_grand_average_proj_interpolated(
                  subject=fsaverage, date=fsaverage_date, Type='patient',
                  fmin=None, fmax=40, tmin=-0.200, tmax=0.400)
    
    evokeds_c = mne.read_evokeds(full_path_c)
    evokeds_p = mne.read_evokeds(full_path_p)
    
    return evokeds_c, evokeds_p
             

def load_data_hilbert(combination, recordings, bad_subjects, fmin, fmax):
    
    for recording_index, recording in enumerate(recordings):
        subject = recording['subject']
        date = recording['date']
        if subject in bad_subjects:
            continue
        if subject == 'fsaverage':
            continue
        print(subject)
        full_path = fname.source_hilbert_beamformer_morph(
                subject=subject,
                date=date,
                fmin=fmin, fmax=fmax,
                tmin=-0.750, tmax=0.750,
                event=combination['contrast'][0]['event'],
                first_event=combination['contrast'][0]['first_event'],
                second_event=combination['contrast'][0]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=1)
        full_path2 = fname.source_hilbert_beamformer_morph(
                subject=subject,
                date=date,
                fmin=fmin, fmax=fmax,
                tmin=-0.750, tmax=0.750,
                event=combination['contrast'][1]['event'],
                first_event=combination['contrast'][1]['first_event'],
                second_event=combination['contrast'][1]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=1)
        
        stc = mne.read_source_estimate(full_path)
        stc2 = mne.read_source_estimate(full_path2)
        temp = np.expand_dims(stc.data.copy(), 0)
        temp2 = np.expand_dims(stc2.data.copy(), 0)
        
        ratio = np.array((temp - temp2) / (temp + temp2))
        ratio[np.isnan(ratio)] = 0
        
        if recording_index == 0:
            data = ratio
        else:
            data = np.concatenate((data, ratio), axis=0)
            
    return data

def get_t(array):

    n_subjects = array.shape[0]
    
    mu    = np.mean(array, axis=0)
    SEM = np.std(array, axis=0) / np.sqrt(n_subjects)
    
    t = mu / SEM
    df = n_subjects - 1
    
    return t, df

def get_t_stc(t, stc_vertex, tmin, tmax):
    t_stc = stc_vertex.copy()
    t_stc._data = t
    t_stc.crop(tmin, tmax)
    
    return t_stc
    

def get_F(array, indices_0, indices_1):
    
    from mne.stats import f_oneway
    n_subjects = len(indices_0) + len(indices_1)
    
    F = f_oneway(array[indices_0, :, :],
                 array[indices_1, :, :])
    
    df_num   = 1
    df_denom = n_subjects - 1
    
    return F, df_num, df_denom


def save_T1_plot_only_hilbert(fig, T1_filename):
    axis_0 = fig.axes[0]
    extent = \
        axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(T1_filename, dpi=300, bbox_inches=extent.expanded(1.00, 0.95))


def get_label_vertices_AAL(labels, src):
    
    ## save nifti version for atlas lookup
    from manuscript_config import vertex_input_name
    from os.path import exists
    from nilearn import image
    stc_vertex = mne.read_source_estimate(vertex_input_name)
    src_vertices = src[0]['vertno']
    nifti_path = vertex_input_name[:-2] + 'nii'
    
    if not exists(nifti_path):
        stc_vertex.save_as_volume(nifti_path, src)
        
    img = image.load_img(nifti_path)

    ## transform to MNI152 (from MNI305)
    def MNI_305_to_152(img, nifti_path):
        import nibabel
        scale_factor = 1.002
        data = img.get_fdata()
        affine_305 = img.affine
        affine_transform = np.array([
            [scale_factor, 0, 0, 0], # scale X
            [0, scale_factor, 0, 0], # scale y
            [0, 0, scale_factor, 0], # scale z
            [0, 0, 0, 1]
            ])
        
        affine_152 = affine_305 @ affine_transform
        img_152 = nibabel.Nifti1Image(data, affine_152, img.header)
        nifti_152_path = nifti_path[:-3] + '_MNI_152.nii' 
        nibabel.save(img_152, nifti_152_path)
        print('MNE305 converted to MNI152')
        
        return nifti_152_path
        
    nifti_152_path = MNI_305_to_152(img, nifti_path)
    
    if not exists(nifti_152_path):
        stc_vertex.save_as_volume(nifti_152_path, src)
        
        
    ## atlas interpolation label-by-label
    from nilearn import datasets
    img = image.load_img(nifti_152_path)
    data = np.asanyarray(img.dataobj)
    
    stc_indices = dict()
    src_indices = dict()
    
    for label in labels:
        print('Finding vertices for label: ' + label)
        
        ## what is to be returned
        these_src_vertices = list()
        these_stc_vertices = list()
        
        
        atlas = datasets.fetch_atlas_aal()
        
        atlas_img = image.load_img(atlas['maps'])
        atlas_interpolated = image.resample_to_img(atlas_img, img, 'nearest')
        atlas_interpolated_data = np.asanyarray(atlas_interpolated.dataobj)
        
        ## find the relevant label
        for atlas_label in atlas['labels']:
            if label == atlas_label:
                break
        else:
            raise NameError('label: ' + label + ' not found')
        label_index = int(atlas['indices'][atlas['labels'].index(label)])
        
        ## make the mask - that only has (arbitrary and positive) data
        mask = atlas_interpolated_data == label_index
        opposite_mask = ~mask
        this_data = data.copy()
        label_data = np.abs(this_data)
        label_data[opposite_mask, :] = 0
        
        x, y, z = np.where(label_data[:, :, :, 0] > 0)
        
        ## create stc coordinates (removing time dimension)
        
        ## all coordinates
        all_coordinates = np.array(
            np.unravel_index(src_vertices, img.shape[:3], order='F')).T
        
        label_coordinates = np.concatenate((np.expand_dims(x, 1),
                                     np.expand_dims(y, 1),
                                     np.expand_dims(z, 1)), axis=1)
        
        ## compare label coordinates to all coordinates
        for coordinate_index, coordinate in enumerate(all_coordinates):
            for label_coordinate in label_coordinates:
                if np.all(coordinate == label_coordinate):
                    src_vertex = src_vertices[coordinate_index]
                    these_src_vertices.append(src_vertex)
                    break
        for src_vertex in these_src_vertices:
            these_stc_vertices.append(np.where(src_vertices == \
                                               src_vertex)[0][0])
                
        src_indices[label] = these_src_vertices
        stc_indices[label] = these_stc_vertices
        
    return stc_indices, src_indices
        

    
def prettify_labels(label):
    if   label == 'Caudate_L':
        pretty_label = 'Caudate-L'
    elif label == 'Caudate_R':
        pretty_label = 'Caudate-R'
    elif label == 'Cerebelum_Crus1_L':
        pretty_label = 'Cerebellum-L Crus I'
    elif label == 'Cerebelum_Crus1_R':
        pretty_label = 'Cerebellum-R Crus I'
    elif label == 'Putamen_L':
        pretty_label = 'Putamen-L'
    elif label == 'Cerebelum_6_L':
        pretty_label = 'Cerebellum-L VI'
    elif label == 'Cerebelum_4_5_L':
        pretty_label = 'Cerebellum-L IV-V'
    elif label == 'Cerebelum_4_5_R':
        pretty_label = 'Cerebellum-R IV-V'
    elif label == 'Precentral_L':
        pretty_label = 'Motor Cortex-L'
    elif label == 'Precentral_R':
        pretty_label = 'Motor Cortex-R'
    elif label == 'Thalamus_L':
        pretty_label = 'Thalamus-L'
    elif label == 'Thalamus_R':
        pretty_label = 'Thalamus-R'        
        
    else:
        print('No pretty version of: ' + label + ' exists (yet)')
        pretty_label = label
    return pretty_label

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

#%% FIGURE - PRESENT F AND t RESULTS

"""
    A: basal ganglia (-14 ms) (F) and cerebellum (-23 ms) (F)
    B: basal ganglia PD (t) and basal ganglia HC (t)
    C: cerebellum PD (t) and cerebellum HC (t)
"""

def get_group_indices(recordings, bad_subjects):
    patient_indices = list()
    control_indices = list()

    subject_counter = 0

    for recording in recordings:
        subject = recording['subject']
        if subject in bad_subjects:
            continue
        if subject == 'fsaverage':
            continue
        if recording['patient']:
            patient_indices.append(subject_counter)
        else:
            control_indices.append(subject_counter)
            
        subject_counter += 1
        
    return patient_indices, control_indices

def plot_F(F_stc, src, initial_time, initial_pos, F_crit):
    from manuscript_config import colormap
    clim=dict(kind='value', lims=(F_crit, 1.1*F_crit, 1.2*F_crit))
    plot = F_stc.plot(src, initial_time=initial_time, initial_pos=initial_pos,
               clim=clim, mode='glass_brain', colormap=colormap)
    
    return plot

def plot_F_vertices(label, F_stc, stc_indices, tmin, tmax, F_crit):
    
    from manuscript_config import coll_col, zero_col
    
    pretty_label = prettify_labels(label)
    
    fig = plt.figure()
    plt.plot(F_stc.times * 1e3, 
             F_stc.data[stc_indices[label], :].T, coll_col)

    plt.xlabel('Time (ms)')
    plt.ylabel('F-values')
    plt.title(pretty_label + ' sources:\n ratios (non-jittered vs jittered)')
    plt.ylim(0, 22)
    plt.vlines(0, 0, 22,
               linestyles='--', lw=3, colors=zero_col)
    plt.hlines(0, tmin * 1e3, tmax * 1e3, linestyles='--', lw=3,
               colors=zero_col)
    
    plt.hlines(F_crit, tmin * 1e3, tmax * 1e3, linestyles='--', lw=1,
               colors=coll_col)
    
    plt.show()
    
    return fig


def plot_t_glass_brain(t_stc, src, initial_time, initial_pos, t_crit):
    from manuscript_config import colormap
    clim=dict(kind='value', lims=(t_crit, 1.1*t_crit, 1.2*t_crit))
    plot = t_stc.plot(src, initial_time=initial_time, initial_pos=initial_pos,
               clim=clim, mode='glass_brain', colormap=colormap)
    
    return plot

def plot_t_vertices(label, patient_t_stc, control_t_stc, stc_indices, 
                    tmin, tmax):
    
    from manuscript_config import p_col, c_col, zero_col
    import numpy as np
    
    pretty_label = prettify_labels(label)
    
    fig = plt.figure()
    plt.plot(patient_t_stc.times * 1e3, 
             patient_t_stc.data[stc_indices[label], :].T, p_col)
    plt.plot(control_t_stc.times * 1e3, 
             control_t_stc.data[stc_indices[label], :].T, c_col)

    plot_legend(p_col, c_col, 'Patient', 'Control')
    plt.xlabel('Time (ms)')
    plt.ylabel('t-values')
    plt.title(pretty_label + ' sources:\n ratios (non-jittered vs jittered)')
    plt.ylim(-5, 5)
    plt.vlines(0, np.min(patient_t_stc.data), np.max(patient_t_stc.data),
               linestyles='--', lw=3, colors=zero_col)
    plt.hlines(0, tmin * 1e3, tmax * 1e3, linestyles='--', lw=3,
               colors=zero_col)
    plt.show()
    
    return fig


#%% FIGURE - REPLICATION OF 44 MS RESULTS (t)

"""
    cerebellum (44 ms) (t)
"""

def plot_t_vertices_collapsed(label, collapsed_t_stc, stc_indices, 
                    tmin, tmax, t_crit):
    
    from manuscript_config import coll_col, zero_col
    import numpy as np
    
    pretty_label = prettify_labels(label)
    
    fig = plt.figure()
    plt.plot(collapsed_t_stc.times * 1e3, 
             collapsed_t_stc.data[stc_indices[label], :].T, coll_col)

    plt.xlabel('Time (ms)')
    plt.ylabel('t-values')
    plt.title(pretty_label + ' sources:\n ratios (non-jittered vs jittered)')
    plt.ylim(-4, 4)
    plt.vlines(0, np.min(collapsed_t_stc.data), np.max(collapsed_t_stc.data),
               linestyles='--', lw=3, colors=zero_col)
    plt.hlines(0, tmin * 1e3, tmax * 1e3, linestyles='--', lw=3,
               colors=zero_col)
    
    plt.hlines(t_crit, tmin * 1e3, tmax * 1e3,
               linestyles='--', lw=1, colors=coll_col)
    plt.hlines(-t_crit, tmin * 1e3, tmax * 1e3,
               linestyles='--', lw=1, colors=coll_col)
    plt.show()
    
    return fig

#%% FIGURE - CORRELATION SIGNIFICANT F AND UPDRS

"""
    A: cerebellum correlation
    B: basal ganglia correlation
    
"""

def get_corr_data(o_beta, vertex_indices, time_indices, patient_indices,
                  stc_vertex):
    source_indices = list()
    for vertex_index in vertex_indices:
        source_indices.append(np.where(stc_vertex.vertices[0] == \
                                                   vertex_index)[0][0])


    corr_data = o_beta[patient_indices, :, :]
    corr_data = corr_data[:, :,  time_indices]
    corr_data = np.mean(corr_data, axis=2)
    corr_data = corr_data[:, source_indices]
    absmax_index = np.argmax(corr_data, axis=1)

    this_corr = list()
    for index in range(len(patient_indices)):
        this_corr.append(corr_data[index, absmax_index[index]])
    corr_data = this_corr

    return corr_data


def get_updrs(recordings, bad_subjects):
    updrs = list()
    for recording in recordings:
        if recording['subject'] == 'fsaverage':
            continue
        if recording['subject'] in bad_subjects:
            continue
        if not recording['patient']:
            continue

        updrs.append(recording['updrs'])
        
    return updrs


def plot_corr(corr_data, updrs, label):
    
    from manuscript_config import p_col, fit_col
        
    pretty_label = prettify_labels(label)
    pretty_label_bf = pretty_label.replace(' ', '\ ')
    
    fig = plt.figure(figsize=(12, 9))
    plt.plot(updrs, corr_data, 'o', color=p_col)
    plt.xlabel('UPDRS-III scores')
    plt.ylabel('Jitter ratio (patients)')
    plt.title('Pearson correlation - UPDRS-III / Jitter ratio\n' + r'$\bf{' + \
              pretty_label_bf + '}$')

    plt.show()

    ## standard lin
    X = np.ones(shape=(len(corr_data), 2))
    X[:, 1] = updrs

    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ corr_data
    

    
    
    # print(beta_hat)
    ## regression line
    plt.plot((np.min(updrs), np.max(updrs)), 
             (np.min(updrs) * beta_hat[1] + beta_hat[0],
              np.max(updrs) * beta_hat[1] + beta_hat[0]), color=fit_col)
        
    ## confidence fit
    import statsmodels.api as sm
    
    model = sm.OLS(corr_data, X)
    results = model.fit()
    
    predictions = results.get_prediction(X)
    pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
    X_sort = np.argsort(X[:,1])
    ci_lower = pred_summary['mean_ci_lower'][X_sort]
    ci_upper = pred_summary['mean_ci_upper'][X_sort]

    
    plt.fill_between(X[:, 1][X_sort], ci_lower, ci_upper, color=fit_col,
                     alpha=0.2,
                     label="95% Confidence Interval")
    
    ## get r and p
    
    from scipy.stats import pearsonr
    r, p = pearsonr(corr_data, updrs)
        
    text = '$\it{r}$ = ' + str(np.round(r, 3)) + '\n$\it{p}$ = ' + \
            str(np.round(p, 4))
    
    if label == 'Cerebelum_Crus1_L'       :
        x_text = 12
        y_text = 0.05
    elif label == 'Caudate_L':
        x_text = 12
        y_text = 0.00
    else:
        raise NameError('Correlation plot not made for this label:' + label)
    plt.text(x_text, y_text, text, horizontalalignment='left', color='w')
    
    return fig

#%% FIGURE - ENVELOPE ANALYSIS

"""
    cerebello-thalamo-caudate correlation
"""

    
def get_envelope_data(recordings, bad_subjects, fmin, fmax, tmin, tmax, event,
                    data):
    
    print('Loading for range: ' + str(fmin) + '-' + str(fmax) + \
          ' Hz for event: ' + event)
    indices = list()
    for recording_index, recording in enumerate(recordings[:]):
        subject = recording['subject']
        if subject in bad_subjects:
            continue
        if subject == 'fsaverage':
            continue
        date    = recording['date']
        first_event = event[0] + '0'
        second_event = event[0] + '15'
        
        filename = fname.envelope_correlation_morph_data(
                    subject=subject, date=date, fmin=fmin, fmax=fmax,
                    tmin=tmin, tmax=tmax, reg=0.00,
                    weight_norm='unit-noise-gain-invariant',
                    event=event, n_layers=1,
                    first_event=first_event, second_event=second_event)
        
        print('Loading subject: ' + subject)
        this_data = np.load(filename)
        data[recording_index, :, :] = this_data
        indices.append(recording_index)

        
    return data, indices

def get_label_envelope(cond_0, cond_1, seed, stc_indices):
    
    values = dict()
     
    for label in stc_indices:
        values[label] = dict()
        values[label]['no-jitter'] = \
np.mean(np.median(cond_0[:, stc_indices[seed], :][:, :, stc_indices[label]],
                                        axis=2), axis=1)
        values[label]['jitter'] = \
np.mean(np.median(cond_1[:, stc_indices[seed], :][:, :, stc_indices[label]],
                                        axis=2), axis=1)
    

    return values


def plot_conn(label_data, node_names, n_lines):
    
    import mne_connectivity
    
    n_nodes = len(node_names)
    conn_array = np.full((n_nodes, n_nodes), np.nan)
    
    row_index = 0
    for node_name in node_names:
        column_index = 0
        for node_name in node_names:
            conn_array[row_index, column_index] = \
np.mean(label_data[node_names[row_index]]['patient'][node_name]['no-jitter']) - \
np.mean(label_data[node_names[row_index]]['patient'][node_name]['jitter']) - \
(np.mean(label_data[node_names[row_index]]['control'][node_name]['no-jitter']) - \
np.mean(label_data[node_names[row_index]]['control'][node_name]['jitter']))

        
            column_index +=1
        row_index += 1
        
    ## only show positive differences ...?   
    # conn_array[conn_array < 0] = 0
    
    pretty_node_names = [prettify_labels(label) for label in node_names]
        
    fig, ax = mne_connectivity.viz.plot_connectivity_circle(conn_array,
                                                        pretty_node_names, 
        vmin=np.nanmin(conn_array),
        vmax=np.nanmax(conn_array),
        node_height=3.0,
        fontsize_names=7,
        fontsize_title=18,
        colormap='GnBu',
        n_lines=n_lines,
        linewidth=5,
        title='Regularity differences (no-jitter minus jitter)\n' + \
            'between Groups (PD minus HC)'
        )    
    
            
    return fig, ax
    

def do_anova(values_c, values_p, labels):
    
    import pandas as pd
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    from manuscript_config import c_col, p_col

    def standardise_data(array):
        stan_array = (array - np.min(array)) / (np.max(array) - np.min(array))
        # stan_array = array
        return stan_array
    
    n_groups = 2
    dv = np.array(list())
    iv_group = np.array(list())
    iv_jitter = np.array(list())
    iv_roi = np.array(list())
    for label in labels:
    
        this_o_no_jitter_control = values_c[label]['no-jitter']
        this_o_jitter_control    = values_c[label]['jitter']
        this_o_no_jitter_patient = values_p[label]['no-jitter']
        this_o_jitter_patient    = values_p[label]['jitter']
        
        
        n_c = len(this_o_jitter_control)
        n_p = len(this_o_jitter_patient)
        
        dv = np.concatenate((dv, this_o_no_jitter_control,
                             this_o_jitter_control,
                             this_o_no_jitter_patient, this_o_jitter_patient))
        
        
        
        iv_group  = np.concatenate((iv_group,
                                    np.repeat('control', n_groups*n_c),
                                    np.repeat('patient', n_groups*n_p)))
        iv_jitter = np.concatenate((iv_jitter,
                    np.repeat('no-jitter', n_c), np.repeat('jitter', n_c),
                    np.repeat('no-jitter', n_p), np.repeat('jitter', n_p)
                    ))
    iv_roi = np.repeat(labels, n_groups  * (n_c + n_p))
    
    
    dt = pd.DataFrame(
                      {
                          'envelope_correlation': dv,
                          'group': iv_group,
                           'roi': iv_roi,
                          'regularity': iv_jitter
                          }
                      )
    
    dt['envelope_correlation'][dt['group'] == 'control'] = \
        standardise_data(dt['envelope_correlation'][dt['group'] == 'control'])
    dt['envelope_correlation'][dt['group'] == 'patient'] = \
        standardise_data(dt['envelope_correlation'][dt['group'] == 'patient'])
    
    if len(labels) == 1:
        model = ols('envelope_correlation ~ group * regularity',
                   data=dt)
    else:
        model = ols('envelope_correlation ~ group * regularity + roi',                  
                   data=dt)
    model = model.fit()
    print(sm.stats.anova_lm(model, typ=2))
    
    
    ## plotting
        
        
    ## FIXME: ugly
    
    this_o_no_jitter_control = \
        dt['envelope_correlation'][dt['group'] == \
                               'control'][dt['regularity'] == 'no-jitter']
    this_o_jitter_control = \
        dt['envelope_correlation'][dt['group'] == \
                               'control'][dt['regularity'] == 'jitter']
    this_o_no_jitter_patient = \
        dt['envelope_correlation'][dt['group'] == \
                               'patient'][dt['regularity'] == 'no-jitter']
    this_o_jitter_patient = \
        dt['envelope_correlation'][dt['group'] == \
                               'patient'][dt['regularity'] == 'jitter']
    
    
    means = np.array([np.mean(this_o_no_jitter_control),
                      np.mean(this_o_jitter_control),
                      np.mean(this_o_no_jitter_patient),
                      np.mean(this_o_jitter_patient)])
    
    sems = np.array(
            [
            np.std(this_o_no_jitter_control)  / \
                np.sqrt(len(this_o_no_jitter_control)),
            np.std(this_o_jitter_control)  / \
                np.sqrt(len(this_o_jitter_control)),
            np.std(this_o_no_jitter_patient)  / \
                np.sqrt(len(this_o_no_jitter_patient)),
            np.std(this_o_jitter_patient)  / \
                np.sqrt(len(this_o_jitter_patient))
            ]
        )
    
    
    def add_jitter(x, jitter=0.05, seed=7):
        np.random.seed(seed)
        x = x + np.random.uniform(-jitter, jitter, x.shape)
        return x
    
            
    fig = plt.figure(figsize=(8, 6))
    x_jittered_0 = add_jitter(np.array((0, 1)), seed=7)
    x_jittered_1 = add_jitter(np.array((0, 1)), seed=14)
    plt.plot(x_jittered_0, means[:2],'-', color=c_col)
 
    plt.xlabel('Regularity')
    plt.ylabel('Normalised Envelope correlation')
    axis = fig.axes[0]
    axis.set_xticks(x_jittered_0)
    axis.set_xticklabels(['No-jitter', 'Jitter'])
    plt.plot(x_jittered_1, means[2:], '-', color=p_col)

    plt.errorbar(x_jittered_0[0], means[0], sems[0], ecolor=c_col,
                 capsize=5)
    plt.errorbar(x_jittered_0[1], means[1], sems[1], ecolor=c_col,
                 capsize=5)
    plt.errorbar(x_jittered_1[0], means[2], sems[2], ecolor=p_col,
                 capsize=5)
    plt.errorbar(x_jittered_1[1], means[3], sems[3], ecolor=p_col,
                 capsize=5)
    
    plt.legend(['Control', 'Patient'])
    plt.title('Caudate connectivity')
    plt.show()
    
    return fig
