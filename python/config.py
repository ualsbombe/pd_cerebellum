#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:54:18 2021

@author: lau

"""

#%% IMPORTS

from os import getlogin
from os.path import join
from socket import getfqdn
from fnames import FileNames


import warnings ## to ignore future warning from nilearn
warnings.simplefilter(action='ignore', category=FutureWarning)

# from nilearn import datasets

#%% GET USER AND HOST AND SET PROJECT PATH
try:
    user = getlogin()
except OSError: # on hyades
    user = None
host = getfqdn()

project_name = 'MINDLAB2021-MEG-Cerebellum-PD'


if user == 'lau' and host == 'lau':
    ## my laptop
    project_path = '/home/lau/projects/pd_cerebellum'
    submitting_method = 'local'
elif (user is None or user == 'lau') and host[:6] == 'hyades':
    hyades_core = int(host[6:8])
    project_path = join('/projects/', project_name)
    if hyades_core < 4:
        ## CFIN server frontend
        submitting_method = 'hyades_frontend'
    else:
        ## CFIN server backend
        submitting_method = 'hyades_backend'
    
else:
    raise RuntimeError('Please edit config.py to include this "user" and '
                       '"host"')

#%% RECORDINGS

# recordings = [
#     dict(subject='0001', date='20210810_000000', mr_date='20191015_121553')
#              ]

recordings = [
    dict(subject='0002', date='20230213_000000', mr_date='20230306_081437'
         , patient=True, updrs=22), # split file
    dict(subject='0003', date='20230213_000000', mr_date='20230306_113109',
         patient=False, updrs=None),
    dict(subject='0004', date='20230213_000000', mr_date='20230306_141259',
         patient=True, updrs=25),
    dict(subject='0005', date='20230214_000000', mr_date='20230307_090426',
         patient=False, updrs=None),
    dict(subject='0006', date='20230214_000000', mr_date='20230307_124424',
         patient=True, updrs=27),
    dict(subject='0007', date='20230214_000000', mr_date='20230306_103544',
         patient=True, updrs=39),
    dict(subject='0008', date='20230215_000000', mr_date='20230308_100536',
         patient=True, updrs=28),
    dict(subject='0009', date='20230215_000000', mr_date='20230309_103706',
         patient=True, updrs=39),
    dict(subject='0010', date='20230215_000000', mr_date='20230309_100659',
         patient=True, updrs=38),
    dict(subject='0011', date='20230216_000000', mr_date='20230309_094428', 
         patient=True, updrs=40),
    dict(subject='0012', date='20230216_000000', mr_date='20230309_091224',
         patient=True, updrs=23),
    dict(subject='0013', date='20230216_000000', mr_date='20230307_131624',
         patient=True, updrs=31),
    dict(subject='0014', date='20230217_000000', mr_date='20230307_081623',
         patient=True, updrs=38),
    dict(subject='0015', date='20230217_000000', mr_date='20230308_110541',
         patient=True, updrs=38),
    dict(subject='0016', date='20230217_000000', mr_date='20230308_113527',
         patient=False, updrs=None),
    dict(subject='0017', date='20230227_000000', mr_date='20230306_125930',
         patient=False, updrs=None),
    dict(subject='0018', date='20230227_000000', mr_date='20230307_093707',
         patient=True, updrs=24),
    dict(subject='0019', date='20230227_000000', mr_date='20230307_120353',
         patient=False, updrs=None),
    dict(subject='0020', date='20230228_000000', mr_date='20230307_101110',
         patient=False, updrs=None),
    dict(subject='0021', date='20230303_000000', mr_date='20230307_140427',
         patient=True, updrs=39),
    dict(subject='0022', date='20230303_000000', mr_date='20230303_131516',
         patient=False, updrs=None),
    dict(subject='0023', date='20230303_000000', mr_date='20230306_163310',
         patient=False, updrs=None),
    dict(subject='0024', date='20230426_000000', mr_date='20230504_115008',
         patient=True, updrs=35),
    dict(subject='0025', date='20230411_000000', mr_date='20230623_090057',
         patient=True, updrs=26),
    dict(subject='0026', date='20230411_000000', mr_date='20230504_093452',
         patient=False, updrs=None),
    dict(subject='0027', date='20230412_000000', mr_date='20230502_085545',
         patient=True, updrs=13),
    dict(subject='0028', date='20230412_000000', mr_date='20230504_093452',
         patient=True, updrs=26),
    dict(subject='0029', date='20230412_000000', mr_date='20230412_161308',
         patient=False, updrs=None),
    dict(subject='0030', date='20230413_000000', mr_date='20230413_094457',
         patient=False, updrs=None),
    dict(subject='0031', date='20230413_000000', mr_date='20230413_131434',
         patient=False, updrs=None),
    dict(subject='0032', date='20230414_000000', mr_date='20230414_102031',
         patient=False, updrs=None),
    dict(subject='0033', date='20230414_000000', mr_date='20230504_110520',
         patient=False, updrs=None),
    dict(subject='0034', date='20230414_000000', mr_date='20230414_160411',
         patient=False, updrs=None),
    dict(subject='0035', date='20230424_000000', mr_date='20230424_095042',
         patient=True, updrs=23),
    dict(subject='0036', date='20230424_000000', mr_date='20230502_102220',
         patient=True, updrs=37),
    dict(subject='0037', date='20230424_000000', mr_date='20230502_162200',
         patient=False, updrs=None),
    dict(subject='0038', date='20230424_000000', mr_date='20230424_184931',
         patient=False, updrs=None),
    dict(subject='0039', date='20230425_000000', mr_date='20230504_080955',
         patient=False, updrs=None),
    dict(subject='0040', date='20230425_000000', mr_date='20230425_152807',
         patient=True, updrs=41),
    dict(subject='0041', date='20230425_000000', mr_date='20230425_155043',
         patient=False, updrs=None),
    dict(subject='0042', date='20230426_000000', mr_date='20230502_093751',
         patient=True, updrs=30),
    dict(subject='0043', date='20230426_000000', mr_date='20230502_110906',
         patient=True, updrs=47),
    dict(subject='0044', date='20230501_000000', mr_date='20230501_105026',
         patient=True, updrs=31),
    dict(subject='0045', date='20230619_000000', mr_date='20230502_120836',
         patient=False, updrs=None),
    dict(subject='0046', date='20230621_000000', mr_date='20230504_124528',
         patient=True, updrs=34),
    dict(subject='0047', date='20230619_000000', mr_date='20230623_094134',
         patient=True, updrs=29),
    dict(subject='0048', date='20230619_000000', mr_date='20230623_110530',
         patient=False, updrs=None),
    dict(subject='0049', date='20230620_000000', mr_date='20230623_123632',
         patient=True, updrs=43),
    dict(subject='0050', date='20230620_000000', mr_date='20230623_141757',
         patient=True, updrs=26),
    dict(subject='0051', date='20230620_000000', mr_date='20230623_115414', 
         patient=True, updrs=25),
    dict(subject='0052', date='20230621_000000', mr_date='20230623_080639',
         patient=False, updrs=None),
    dict(subject='0053', date='20230621_000000', mr_date='20230623_154026', 
         patient=False, updrs=None),
    dict(subject='0054', date='20230622_000000', mr_date='20230622_103548'
         , patient=True, updrs=43),
    dict(subject='0055', date='20230622_000000', mr_date='20230622_133019',
         patient=False, updrs=None),
    dict(subject='0056', date='20230622_000000', mr_date='20230622_162042',
         patient=False, updrs=None),
    dict(subject='0057', date='20230629_000000', mr_date='20230629_164800',
         patient=False, updrs=None),
    
    dict(subject='fsaverage', date='20230629_000000', mr_date=None)

    
    ]
#%% SUBJECT SPECIFIC
## file in bad ones from "subject_notes.txt"
bad_channels = dict()
behavioural_data_time_stamps = dict()

bad_channels['0001'] = ['MEG0142']
bad_channels['0002'] = ['MEG0422']
bad_channels['0003'] = ['MEG0422']
bad_channels['0004'] = ['MEG0422']
bad_channels['0005'] = ['MEG0422', 'MEG1712', 'MEG1221', 'MEG1411', 'MEG1441',
                        'MEG1321']
bad_channels['0006'] = ['MEG0422', 'MEG2131']
bad_channels['0007'] = ['MEG0422', 'MEG1712', 'MEG2413']
bad_channels['0008'] = ['MEG0422']
bad_channels['0009'] = ['MEG0422', 'MEG2342', 'MEG2542', 'MEG2131', 'MEG2141']
bad_channels['0010'] = ['MEG0422']
bad_channels['0011'] = ['MEG0422']
bad_channels['0012'] = ['MEG0422']
bad_channels['0013'] = ['MEG0422']
bad_channels['0014'] = ['MEG0422', 'MEG0433']
bad_channels['0015'] = ['MEG0422']
bad_channels['0016'] = ['MEG0422']
bad_channels['0017'] = ['MEG0422', 'MEG0742']
bad_channels['0018'] = ['MEG0422']
bad_channels['0019'] = ['MEG0422', 'MEG2443']
bad_channels['0020'] = ['MEG0422']
bad_channels['0021'] = ['MEG0422', 'MEG0343']
bad_channels['0022'] = ['MEG0422', 'MEG1742', 'MEG2141', 'MEG1531', 'MEG0343']
bad_channels['0023'] = ['MEG0422', 'MEG0343']
bad_channels['0024'] = ['MEG0422']
bad_channels['0025'] = ['MEG0422', 'MEG0633']
bad_channels['0026'] = ['MEG0412', 'MEG0422']
bad_channels['0027'] = ['MEG0422']
bad_channels['0028'] = ['MEG0422', 'MEG2223', 'MEG1133']
bad_channels['0029'] = ['MEG0422', 'MEG1312']
bad_channels['0030'] = ['MEG0422']
bad_channels['0031'] = ['MEG0422', 'MEG1412', 'MEG1442']
bad_channels['0032'] = ['MEG0422', 'MEG1132']
bad_channels['0033'] = ['MEG0422']
bad_channels['0034'] = ['MEG0422', 'MEG2542']
bad_channels['0035'] = ['MEG0422', 'MEG1312', 'MEG0823']
bad_channels['0036'] = ['MEG0422', 'MEG2423', 'MEG1312']
bad_channels['0037'] = ['MEG0422', 'MEG0823', 'MEG1312']
bad_channels['0038'] = ['MEG0422', 'MEG2443']
bad_channels['0039'] = ['MEG0422', 'MEG2422']
bad_channels['0040'] = ['MEG0422']
bad_channels['0041'] = ['MEG0422', 'MEG1312']
bad_channels['0042'] = ['MEG0422']
bad_channels['0043'] = ['MEG0422', 'MEG1731']
bad_channels['0044'] = ['MEG0422', 'MEG1312']
bad_channels['0045'] = ['MEG0422', 'MEG0933']
bad_channels['0046'] = ['MEG0422']
bad_channels['0047'] = ['MEG0422', 'MEG2342']
bad_channels['0048'] = ['MEG0422', 'MEG1343']
bad_channels['0049'] = ['MEG0422']
bad_channels['0050'] = ['MEG0422', 'MEG1312']
bad_channels['0051'] = ['MEG0422']
bad_channels['0052'] = ['MEG0422', 'MEG0821', 'MEG2533', 'MEG1312']
bad_channels['0053'] = ['MEG0422']
bad_channels['0054'] = ['MEG0422']
bad_channels['0055'] = ['MEG0422', 'MEG2523']
bad_channels['0056'] = ['MEG0422', 'MEG1443', 'MEG0113']
bad_channels['0057'] = ['MEG0422', 'MEG0423']



#%% GENERAL


split_recording_subjects = ['0002']
events_with_256_added = ['0017']


bad_subjects = ['0008', '0014', '0019', # left-handers
                '0024', '0025', '0040', # metal plate
                '0036' # atrophy
                ] 
# noisy profiles, power spectra:  0010, 0013 , 0018, 0033


#%% GENERAL PLOTTING

n_jobs_power_spectra = 3

#%% EVOKED ANALYSIS

## filtering

evoked_fmin = None
evoked_fmax = 40 # Hz

## epoching

evoked_tmin = -0.200 # s
evoked_tmax =  0.400 # s
evoked_baseline = (None, 0) # s
evoked_decim = 1
evoked_event_id = dict(s1=1, s2=3, s3=5,
                       s4_0=23, s5_0=25, s6_0=27,
                       s4_15=33, s5_15=35, s6_15=37,
                       o0=18, o15=28,
                       n1_0=48, n2_0=50, n3_0=52, n4_0=54, n5_0=56,
                       n1_15=58, n2_15=60, n3_15=62, n4_15_0=64, n5_15=66)
# evoked_reject = dict(mag=4e-12, grad=4000e-13) # T / T/cm
# evoked_proj = False

## averaging


#%% HILBERT ANALYSIS

## filtering

hilbert_fmins = [14, 
                 # 4,  8
                 ]
hilbert_fmaxs = [30,
                 # 7, 12
                 ]

## transforming

hilbert_tmin = -0.750 # s
hilbert_tmax =  0.750 # s
hilbert_baseline = None
hilbert_decim = 1
hilbert_event_id = evoked_event_id
hilbert_reject = None ## think about this...

## averaging

#%% CREATE FORWARD MODEL

## import mri

t1_file_ending = 't1_mprage_3D_sag_fatsat'
t2_file_ending = 't2_tse_sag_HighBW'

## make scalp surface with fine resolution

## transformation

## volumetric source space

src_spacing = 7.5 # mm

## bem model

bem_ico = 4
bem_conductivities = [
                        [0.3], # single-layer model
                        [0.3, 0.006, 0.3] # three-layer model
                        ]

## bem solution

## morph 

morph_subject_to = 'fsaverage'

## forward solution
subjects_no_watershed = dict()
subjects_no_watershed['1_layer'] = []
subjects_no_watershed['3_layer'] = ['0006', '0013', '0018',
                                    '0019', '0020', '0022',
                                    '0026', '0027', '0035',
                                    '0036', '0042', '0045',
                                    '0050', '0057']

## simnibs


#%% CONTRASTS

contrasts = [
                 # ['s1', 's2'], ['s2', 's3'], ['s3', 's4_0'],
                  # ['s4_0', 's5_0'], ['s5_0', 's6_0'],
                  # ['s4_15', 's5_15'], ['s5_15', 's6_15'],
                  # ['s4_0', 's4_15'], ['s5_0', 's5_15'], ['s6_0', 's6_15'],
                ['o0', 'o15'],     
                    ]

#%% SOURCE ANALYSIS EVOKED

## lcmv contrasts

evoked_lcmv_contrasts = contrasts
evoked_lcmv_weight_norms = ['unit-noise-gain-invariant', 'unit-gain']
evoked_lcmv_regularization = 0.00 # should we regularize?
evoked_lcmv_picks = 'mag' # can they be combined?
evoked_lcmv_proj = False

## morph contrasts

#%% SOURCE ANALYSIS HILBERT

## lcmv contrasts

hilbert_lcmv_contrasts = contrasts
hilbert_lcmv_weight_norms = ['unit-noise-gain-invariant']#, 'unit-gain']
hilbert_lcmv_regularization = 0.05 # should we regularize?
hilbert_lcmv_picks = 'mag' # can they be combined?

## morph contrasts

## labels

#%% ENVELOPE CORRELATIONS

envelope_events = [['o0', 'o15']]
envelope_downsampling = 100 ## ?!
envelope_fmins = [14]
envelope_fmaxs = [30]
envelope_tmin = -0.100
envelope_tmax =  0.100
envelope_weight_norm = 'unit-noise-gain-invariant'
envelope_regularization = 0.00
envelope_picks = 'mag' # can they be combined?

subjects_conn_cannot_be_saved = []


#%% GRAND AVERAGE AND STATISTICS 

## grand average and stat common

grand_average_types = ['collapsed',
                       'patient', 'control'
                       ]
stat_types = [
             'collapsed-within', 'patient-within', 'control-within',
              'control-patient-between'
              ]

#%% SET FILENAMES

fname = FileNames()    

## directories
fname.add('project_path', project_path)
fname.add('raw_path', '{project_path}/raw')
fname.add('scratch_path', '{project_path}/scratch')
fname.add('MEG_path', '{scratch_path}/MEG')
fname.add('simnibs_subjects_dir', '{scratch_path}/simnibs')
fname.add('freesurfer_subjects_dir', '{scratch_path}/freesurfer')
fname.add('figures_path', '{scratch_path}/figures')
fname.add('script_path', '{project_path}/scripts')
fname.add('python_path', '{script_path}/python')
fname.add('python_qsub_path', '{python_path}/qsub')

## FreeSurfer

fname.add('subject_freesurfer_path', '{freesurfer_subjects_dir}/{subject}')

## SimNIBS directories

fname.add('subject_simnibs_path', '{simnibs_subjects_dir}/{subject}')
fname.add('simnibs_freesurfer_subjects_dir',
          '{simnibs_subjects_dir}/freesurfer')
fname.add('subject_fs_path', '{subject_simnibs_path}/fs_{subject}')
fname.add('subject_m2m_path', '{subject_simnibs_path}/m2m_{subject}')
fname.add('subject_headreco_path',
          '{subject_simnibs_path}/m2m_{subject}_headreco')
fname.add('simnibs_bem_path', '{subject_fs_path}/bem')
fname.add('freesurfer_bem_path', '{freesurfer_subjects_dir}/{subject}/bem')


## directories that require input
fname.add('subject_path', '{MEG_path}/{subject}/{date}')
fname.add('subject_figure_path', '{figures_path}/{subject}/{date}')
fname.add('subject_beamformer_evoked_path',
          '{subject_path}/beamformer_evoked')
fname.add('subject_beamformer_hilbert_path',
          '{subject_path}/beamformer_hilbert')
fname.add('subject_beamformer_hilbert_labels_path',
          '{subject_path}/beamformer_hilbert/labels')
fname.add('subject_envelope_path', '{subject_path}/envelopes')
fname.add('subject_MR_path', '{raw_path}/{subject}/{date}/MR')
fname.add('subject_MR_elsewhere_path',
          '{scratch_path}/MRs_from_elsewhere/{subject}/{date}/MR')

## raw filenames
fname.add('raw_file', '{raw_path}/{subject}/{date}/MEG/'
                      '001.pd_cerebellum_raw/files/pd_cerebellum_raw.fif')
fname.add('split_raw_file_1', '{raw_path}/{subject}/{date}/MEG/'
          '001.pd_cerebellum_raw_1/files/pd_cerebellum_raw_1.fif')
fname.add('split_raw_file_2', '{raw_path}/{subject}/{date}/MEG/'
          '002.pd_cerebellum_raw_2/files/pd_cerebellum_raw_2.fif')
fname.add('split_trans', '{subject_path}/pd-split-trans.fif')


## MEG output
fname.add('events', '{subject_path}/pd-eve.fif')

## evoked
fname.add('evoked_filter', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz-raw.fif')
fname.add('evoked_epochs', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-epo.fif')
fname.add('evoked_average_no_proj', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-no_proj-ave.fif')
fname.add('evoked_average_proj', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-proj-ave.fif')
fname.add('evoked_grand_average_proj_interpolated',
              '{subject_path}/pd-filt-{Type}-{fmin}-{fmax}-Hz'
              '-{tmin}-{tmax}-s-proj-interpolated-ave.fif')



## hilbert
fname.add('hilbert_filter', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz-raw.fif')
fname.add('hilbert_epochs', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-epo.fif')

fname.add('hilbert_average_no_proj', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-no_proj-ave.fif')
fname.add('hilbert_average_proj', '{subject_path}/pd-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-proj-ave.fif')

fname.add('hilbert_grand_average_no_proj', '{subject_path}/pd-filt-{fmin}'
                                            '-{fmax}-Hz-{tmin}-{tmax}-s'
                                            '-no_proj-z_contrasts-ave.fif')
fname.add('hilbert_grand_average_proj', '{subject_path}/pd-filt-{fmin}'
                                            '-{fmax}-Hz-{tmin}-{tmax}-s'
                                            '-proj-z_contrasts-ave.fif')

fname.add('hilbert_grand_average_proj_interpolated',
              '{subject_path}/pd-filt-{Type}-{fmin}-{fmax}-Hz'
              '-{tmin}-{tmax}-s-proj-interpolated-ave.fif')


## anatomy

fname.add('anatomy_transformation', '{subject_path}/pd-trans.fif')


# freesufer
fname.add('anatomy_freesurfer_fiducials',
          '{freesurfer_bem_path}/{subject}-fiducials.fif' )

fname.add('anatomy_freesurfer_bem_surfaces',
           '{freesurfer_bem_path}/{n_layers}-layers-bem.fif')
fname.add('anatomy_freesurfer_bem_solutions',
           '{freesurfer_bem_path}/{n_layers}-layers-bem-sol.fif')
fname.add('anatomy_freesurfer_volumetric_source_space',
          '{freesurfer_bem_path}/volume-{spacing}_mm-src.fif')
fname.add('anatomy_freesurfer_morph_volume', '{freesurfer_bem_path}/volume-'
          '{spacing}_mm-morph.h5')
fname.add('anatomy_freesurfer_forward_model', '{subject_path}/'
          'pd-freesurfer-volume-{spacing}_mm-{n_layers}-layers-fwd.fif')


## source evoked
fname.add('source_evoked_beamformer', '{subject_path}'
          '/beamformer_evoked/'
          'pd-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}-'
          'n_layers-{n_layers}-vl.stc')

fname.add('source_evoked_beamformer_morph',
          '{subject_path}/beamformer_evoked/'
          'pd-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}-'
          'n_layers-{n_layers}-morph-vl-stc.h5')


fname.add('source_evoked_beamformer_grand_average', '{subject_path}/'
          'beamformer_evoked/'
          'pd-filt-{Type}-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-{event}-'
          'filter-{first_event}-{second_event}-{weight_norm}-'
          'n_layers-{n_layers}-morph-vl.h5')


## source hilbert

fname.add('source_hilbert_beamformer', '{subject_path}/beamformer_hilbert/'
          'pd-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}'
          '-n_layers-{n_layers}-vl.stc')

fname.add('source_hilbert_beamformer_morph', '{subject_path}'
          '/beamformer_hilbert/'
          'pd-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}'
          '-n_layers-{n_layers}-morph-vl-stc.h5')


fname.add('source_hilbert_beamformer_grand_average', '{subject_path}'
          '/beamformer_hilbert/'
          'pd-filt-{Type}-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}'
          '-n_layers-{n_layers}-morph-vl.h5')

## envelopes

fname.add('envelope_correlation', '{subject_envelope_path}/'
          'pd-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-event-{event}-'
          'filter-{first_event}-{second_event}-{weight_norm}'
          '-n_layers-{n_layers}.nc')
fname.add('envelope_correlation_morph_data', '{subject_envelope_path}/'
          'pd-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-event-{event}-'
          'filter-{first_event}-{second_event}-{weight_norm}'
          '-n_layers-{n_layers}-morph-data.npy')



## figure names
fname.add('power_spectra_plot', '{subject_figure_path}/pd_power_spectra.png')
fname.add('events_plot', '{subject_figure_path}/pd_events.png')

## set the environment for FreeSurfer and MNE-Python
# environ["SUBJECTS_DIR"] = fname.freesufer_subjects_dir

#%% HYADES PARAMETERS
hyades_parameters = dict() # filled out below

## general
hyades_parameters['analysis_00_create_folders'] =  dict(queue='short.q',
                                                        job_name='cf',
                                                        n_jobs=1,
                                                        deps=None)
 
hyades_parameters['analysis_01_find_events'] = dict(queue='highmem.q',
                                                    job_name='eve',
                                                    n_jobs=2, deps=None)

## plot

 
hyades_parameters['analysis_plot_00_power_spectra'] = dict(queue='highmem.q',
                                                           job_name='plotps',
                                                   n_jobs=n_jobs_power_spectra,
                                                   deps=None)


## evokeds
hyades_parameters['analysis_evoked_00_filter'] = dict(queue='highmem.q', 
                                                      job_name='efilt',
                                                      n_jobs=2, deps=None)

hyades_parameters['analysis_evoked_01_epochs'] = dict(queue='highmem.q',
                                                      job_name='eepo',
                                                      n_jobs=2, 
                                                      deps=['eve', 'efilt'])

hyades_parameters['analysis_evoked_02_average'] = dict(queue='all.q',
                                                       job_name='eave',
                                                       n_jobs=3,
                                               deps=['eve', 'efilt', 'eepo'])

hyades_parameters['analysis_evoked_03_grand_average'] = dict(queue='long.q',
                                                       job_name='egave',
                                                       n_jobs=4,
                                               deps=['eve', 'efilt', 'eepo',
                                                     'eave'])


## hilbert

hyades_parameters['analysis_hilbert_00_filter'] = dict(queue='highmem.q',
                                                       job_name='hfilt',
                                                       n_jobs=4, deps=None)

hyades_parameters['analysis_hilbert_01_epochs'] = dict(queue='highmem.q',
                                                       job_name='hepo',
                                                       n_jobs=4, 
                                                       deps=['eve', 'hfilt'])

hyades_parameters['analysis_hilbert_02_average'] = dict(queue='highmem.q',
                                                        job_name='have',
                                                        n_jobs=6, 
                                                deps=['eve', 'hfilt', 'hepo'])

hyades_parameters['analysis_hilbert_03_grand_average'] = dict(queue='long.q',
                                                              job_name='hgave',
                                                              n_jobs=4,
                                        deps=['eve', 'hfilt', 'hepo', 'have'])

## anatomy

hyades_parameters['analysis_anatomy_simnibs_00_segmentation'] = \
    dict(queue='highmem.q', job_name='snibs',
                                 n_jobs=3, deps=None)

hyades_parameters['analysis_anatomy_simnibs_01_bem'] = dict(queue='all.q',
                                                            job_name='snbem',
                                                            n_jobs=1, 
                                                            deps=['snibs'])

hyades_parameters['analysis_anatomy_simnibs_02_forward_model'] = \
        dict(queue='all.q', job_name='snfwd',
                                 n_jobs=1, deps=['snibs', 'snbem'])

## anatomy freesurfer

hyades_parameters['analysis_anatomy_freesurfer_00_segmentation'] = \
        dict(queue='long.q', job_name='fsreco',
                                 n_jobs=1, deps=None)



hyades_parameters['analysis_anatomy_freesurfer_01_bem'] = \
                        dict(queue='long.q',
                                        job_name='fsbem',
                                        n_jobs=1,
                                    deps=['fsreco'])

hyades_parameters['analysis_anatomy_freesurfer_02_forward_model'] = \
        dict(queue='all.q', job_name='fsfwd',
                                 n_jobs=1, deps=['fsreco', 'fsbem'])
        
## anatomy simnibs

hyades_parameters['analysis_anatomy_simnibs_00_segmentation'] = \
            dict(queue='highmem.q', job_name='snreco',
                                     n_jobs=1, deps=None)    
      
        
## source evoked

hyades_parameters['analysis_source_evoked_00_beamformer'] = \
        dict(queue='all.q', job_name='elcmv',
                                 n_jobs=4, deps=['fsceco', 'fsbem', 'fsfwd',
                                             'eve', 'efilt', 'eepo', 'eave'])
        
hyades_parameters['analysis_source_evoked_01_morph_beamformer'] = \
                dict(queue='all.q', job_name='emlcmv',
                     n_jobs=1, deps=['fsceco', 'fsbem', 'fsfwd',
                                'eve', 'efilt', 'eepo', 'eave',
                                'elcmv'])

hyades_parameters['analysis_source_evoked_02_beamformer_grand_average'] = \
                dict(queue='all.q', job_name='egalcmv',
                     n_jobs=8, deps=['fsceco', 'fsbem', 'fsfwd',
                                'eve', 'efilt', 'eepo', 'eave', 'elcmv',
                                        'emlcmv'])       
                
                
## source hilbert

                
hyades_parameters['analysis_source_hilbert_00_beamformer'] = \
        dict(queue='highmem.q', job_name='hlcmv',
                                 n_jobs=6, deps=['fsceco', 'fsbem', 'fsfwd',
                                             'eve', 'hfilt', 'hepo', 'have'])   

hyades_parameters['analysis_source_hilbert_01_morph_beamformer'] = \
                dict(queue='long.q', job_name='hmlcmv',
                     n_jobs=1, deps=['fsceco', 'fsbem', 'fsfwd',
                                'eve', 'hfilt', 'hepo', 'have',
                                'hlcmv'])

hyades_parameters['analysis_source_hilbert_02_beamformer_grand_average']  = \
                dict(queue='highmem.q', job_name='hgalcmv',
                     n_jobs=4, deps=['fsceco', 'fsbem', 'fsfwd',
                                'eve', 'hfilt', 'hepo', 'have',
                                'hlcmv', 'hmlcmv'])
                
                
## envelope

hyades_parameters['analysis_source_hilbert_03_envelope_correlation'] = \
                dict(queue='long.q', job_name='helcmv',
                     n_jobs=8, deps=['fsreco', 'fsbem', 'fsfwd',
                                     'eve', 'hfilt', 'hepo', 'have'])
     