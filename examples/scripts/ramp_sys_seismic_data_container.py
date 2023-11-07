# -*- coding: utf-8 -*-
"""
Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.sep.join(['..', '..', 'src']))
from openiam import SystemModel

from ramp import SeismicSurveyConfiguration, SeismicDataContainer
from ramp.utilities.data_readers import default_bin_file_reader

sys.path.insert(0, os.sep.join(['..', '..', 'misc', 'components_setup']))
from dc_configuration_setup import create_seismic_data_setup


if __name__ == "__main__":

    # Define keyword arguments of the system model
    final_year = 100
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'seismic'
    data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
    output_directory = os.path.join(
        '..', 'user', 'output', 'ramp_sys_seismic_data_container')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (1251, 101, 9),
                          'move_axis_destination': [-1, -2, -3]}
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)

    start_ind = 2
    end_ind = 10
    indices = np.arange(start_ind, end_ind+1)

    # Create setup files instead of a dictionary
    setup_filename = 'seismic_setup_file.csv'
    tp_filename = 'time_points.csv'
    create_seismic_data_setup(indices, filename=setup_filename,
                              t_filename=tp_filename)
    family = 'seismic'
    baseline = True

    # Define coordinates of sources
    num_sources = 9
    sources = np.c_[4000 + np.array([240, 680, 1120, 1600, 2040, 2480, 2920, 3400, 3840]), # x
                    np.zeros(num_sources), # y
                    np.zeros(num_sources)] # z

    # Define coordinates of receivers
    num_receivers = 101
    receivers = np.c_[4000 + np.linspace(0, 4000, num=num_receivers),
                      np.zeros(num_receivers),
                      np.zeros(num_receivers)]

    # Create survey with defined coordinates
    survey_config = SeismicSurveyConfiguration(sources, receivers, name='Survey 1')

    # ------------- Create system model -------------
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # ------------- Add data container -------------
    dc = sm.add_component_model_object(
        SeismicDataContainer(name='dc', parent=sm, survey_config=survey_config,
                             total_duration=2.5, sampling_interval=0.002,
                             family=family, obs_name=obs_name,
                             data_directory=data_directory,
                             data_setup=setup_filename, time_points=tp_filename,
                             baseline=baseline, data_reader=data_reader,
                             data_reader_kwargs=data_reader_kwargs))
    # Add parameters of the container
    dc.add_par('index', value=start_ind, vary=False)
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    print('-----------------------------')
    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()
    print('-----------------------------')
    print('Forward simulation finished.')
    print('-----------------------------')

    # Get saved data from files
    time_ind = final_year//10 - 1
    source_ind = 5
    delta_seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'delta_seismic', output_directory, indices=[time_ind], rlzn_number=0)[0]
    seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'seismic', output_directory, indices=[time_ind], rlzn_number=0)[0]

    # Plot results
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    ax.imshow(seismic[source_ind, :, :].T, cmap='gray', aspect='auto')
    ax.set_title('Seismic data at {} years'.format((time_ind+1)*10))
    y = np.linspace(0, 1200, num=5)
    ylabels = np.linspace(0, 2.5, num=5)
    ax.set_yticks(y, labels=ylabels)
    ax.set_ylabel('Time, [sec]')
    x = np.linspace(0, 100, num=6)
    xlabels = 4000 + np.linspace(0, 4000, num=6)
    ax.set_xticks(x, labels=xlabels)
    ax.set_xlabel('Receiver location, [m]')

    ax = fig.add_subplot(122)
    ax.imshow(delta_seismic[source_ind, :, :].T, cmap='gray', aspect='auto')
    ax.set_title('Seismic data difference at {} years'.format((time_ind+1)*10))
    x = np.linspace(0, 100, num=6)
    xlabels = 4000 + np.linspace(0, 4000, num=6)
    ax.set_xticks(x, labels=xlabels)
    ax.set_xlabel('Receiver location, [m]')
    ax.set_yticks(y, labels=ylabels)
