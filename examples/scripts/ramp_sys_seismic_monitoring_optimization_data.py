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
sys.path.insert(0, os.sep.join(['..', '..', 'source']))
from openiam import SystemModel

sys.path.insert(0, os.sep.join(['..', '..', 'ramp']))
from ramp.data_container import default_bin_file_reader
from ramp.seismic_data_container import SeismicDataContainer
from ramp.seismic_configuration import SeismicSurveyConfiguration
from ramp.seismic_monitoring import SeismicMonitoring

if __name__ == "__main__":
    # Define keyword arguments of the system model
    time_points = 10*np.arange(1, 21)
    time_array = 365.25*time_points
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'seismic'
    data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
    output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                    'ramp_sys_seismic_monitoring_optimization_data')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (1251, 101, 9),
                          'move_axis_destination': [-1, -2, -3]}

    num_time_points = len(time_points)
    excluded = [37, 118, 136, 150, 182, 245]  # 6 scenarios
    scenarios = set(range(1, 307)).difference(excluded)
    scenarios = list(scenarios)
    num_scenarios = len(scenarios)
    family = 'seismic'
    data_setup = {}
    for scen in scenarios:
        data_setup[scen] = {'folder': os.path.join('data_sim{:04}'.format(scen),
                                                   'data')}
        for t_ind, tp in enumerate(time_points):
            data_setup[scen]['t{}'.format(t_ind+1)] = 'data_sim{:04}_t{}.bin'.format(scen, tp)
    baseline = True

    # Define coordinates of sources
    num_sources = 9
    sources = np.c_[4000 + np.array([240, 680, 1120, 1600, 2040, 2480, 2920, 3400, 3840]),
                    np.zeros(num_sources),
                    np.zeros(num_sources)]

    # Define coordinates of receivers
    num_receivers = 101
    receivers = np.c_[4000 + np.linspace(0, 4000, num=num_receivers),
                      np.zeros(num_receivers),
                      np.zeros(num_receivers)]

    # Create survey with defined coordinates
    survey_config = SeismicSurveyConfiguration(sources, receivers, name='Test Survey')

    # ------------- Create system model -------------
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # ------------- Add data container -------------
    dc = sm.add_component_model_object(
        SeismicDataContainer(name='dc', parent=sm, survey_config=survey_config,
                             total_duration=2.5,
                             sampling_interval=0.002,
                             family=family, obs_name=obs_name,
                             data_directory=data_directory, data_setup=data_setup,
                             time_points=time_points, baseline=baseline,
                             data_reader=data_reader,
                             data_reader_kwargs=data_reader_kwargs,
                             presetup=True))
    # Add parameters of the container
    dc.add_par('index', value=scenarios[0],
               discrete_vals=[scenarios, num_scenarios*[1/num_scenarios]])
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    # ------------- Add seismic monitoring technology -------------
    smt = sm.add_component_model_object(
        SeismicMonitoring(name='smt', parent=sm, survey_config=survey_config,
                          time_points=time_points))
    # Add keyword arguments linked to the seismic data container outputs
    smt.add_kwarg_linked_to_obs('data', dc.linkobs['seismic'], obs_type='grid')
    smt.add_kwarg_linked_to_obs('baseline', dc.linkobs['baseline_seismic'], obs_type='grid')
    # Add gridded observation
    smt.add_grid_obs('NRMS', constr_type='matrix', output_dir=output_directory)
    # Add scalar observations
    for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
        smt.add_obs(nm)
        smt.add_obs_to_be_linked(nm)

    print('--------------------------------')
    print('Stochastic simulation started...')
    print('--------------------------------')
    print('Number of scenarios: {}'.format(num_scenarios))
    # Create sampleset varying over scenarios: this is not a typical setup
    # We want to make sure scenarios are not repeated.
    samples = np.array(scenarios).reshape(num_scenarios, 1)
    s = sm.create_sampleset(samples)

    results = s.run(cpus=5, verbose=False)
    print('--------------------------------')
    print('Stochastic simulation finished.')
    print('--------------------------------')

    print('--------------------------------')
    print('Collecting results...')
    print('--------------------------------')
    # Get saved gridded observations from files
    nrms = np.zeros((num_scenarios, num_time_points,
                    num_sources, num_receivers))
    time_indices = list(range(num_time_points))

    for rlzn_number in range(1, num_scenarios+1):
        print('Realization {}'.format(rlzn_number))
        data = sm.collect_gridded_observations_as_time_series(
            smt, 'NRMS', output_directory, indices=time_indices,
            rlzn_number=rlzn_number) # data shape (num_time_points, num_sources, num_receivers)
        nrms[rlzn_number-1] = data

    file_to_save = os.path.join(
        output_directory,
        'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
    np.savez_compressed(file_to_save, data=nrms)

    trace_length = 1251
    seismic_data = np.zeros((num_scenarios, num_time_points,
                             num_sources, num_receivers, trace_length))
    for rlzn_number in range(1, num_scenarios+1):
        print('Realization {}'.format(rlzn_number))
        data = sm.collect_gridded_observations_as_time_series(
            dc, 'seismic', output_directory, indices=time_indices,
            rlzn_number=rlzn_number) # data shape (num_time_points, num_sources, num_receivers, trace_length)
        seismic_data[rlzn_number-1] = data

    file_to_save = os.path.join(
        output_directory,
        'seismic_optimization_data_{}_scenarios.npz'.format(num_scenarios))
    np.savez_compressed(file_to_save, data=seismic_data)

    data_check = 0
    if data_check:
        # Read file
        file_to_read = os.path.join(
            output_directory,
            'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
        d = np.load(file_to_read)
        # Determine shape of the data
        data_shape = d['data'].shape
        print(data_shape)  # (36, 20, 9, 101)
        nrms = d['data']

        # Check that the data makes sense
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax_im = ax.imshow(nrms[0, 8, :, :], aspect='auto')

        # Set title
        title = ax.set_title('NRMS at {} years'.format(90))

        # Set x-labels
        x = np.linspace(0, 100, num=11)
        xlabels = np.linspace(1, 101, num=11, dtype=int)
        ax.set_xticks(x, labels=xlabels)
        ax.set_xlabel('Receivers')

        # Set y-labels
        y = np.linspace(0, 8, num=9)
        ylabels = np.linspace(1, 9, num=9, dtype=int)
        ax.set_yticks(y, labels=ylabels)
        ax.set_ylabel('Sources')

        # Add colorbar
        cbar = plt.colorbar(ax_im, label='Percentage, %')
        fig.tight_layout()
