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

sys.path.insert(0, os.sep.join(['..', '..', 'ramp']))
from ramp.utilities.data_readers import default_bin_file_reader
from ramp import (SeismicDataContainer, SeismicSurveyConfiguration,
                  SeismicMonitoring, SeismicEvaluation)


if __name__ == "__main__":
    # Define keyword arguments of the system model
    final_year = 90
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'seismic'
    data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
    output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                    'ramp_sys_seismic_evaluation_lhs')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (1251, 101, 9),
                          'move_axis_destination': [-1, -2, -3]}
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)
    scenarios = list(range(1, 21))
    num_scenarios = len(scenarios)
    family = 'seismic'
    data_setup = {}
    for ind in scenarios:
        data_setup[ind] = {'folder': os.path.join('data_sim{:04}'.format(ind),
                                                  'data')}
        for t_ind in range(1, num_time_points+1):
            data_setup[ind]['t{}'.format(t_ind)] = 'data_sim{:04}_t{}.bin'.format(ind, t_ind*10)
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
    dc.add_par('index', value=1,
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

    # ------------- Add seismic evaluation component -------------
    seval = sm.add_component_model_object(
        SeismicEvaluation(name='seval', parent=sm, time_points=time_points))
    # Add threshold parameter
    threshold = 5.0
    seval.add_par('threshold', value=threshold, vary=False)
    # Add keyword arguments linked to the seismic monitoring metric
    seval.add_kwarg_linked_to_obs('metric', smt.linkobs['ave_NRMS'])
    # Add observations
    for nm in ['leak_detected', 'detection_time']:
        seval.add_obs(nm)        # covers the whole simulation period
        seval.add_obs(nm + '_ts')  # covers how values change in time

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
    # Extract results from stochastic simulations
    out = s.collect_observations_as_time_series()

    # Get observations
    evaluations = {}
    for nm in ['leak_detected', 'detection_time']:
        evaluations[nm] = out['seval.' + nm][:, -1]

    print('Probability of detection:',
          np.sum(evaluations['leak_detected'])/num_scenarios)

    print('--------------------------------')
    print('Plotting results...')
    print('--------------------------------')
    # Plot results
    # Plot leak detected and detection times
    plt.close('all')
    scen_indices = np.array(scenarios)
    fig = plt.figure(figsize=(10, 4))
    # First subplot
    ax = fig.add_subplot(121)
    colors = np.where(evaluations['leak_detected']==1, 'blue', 'red')

    ax.scatter(scen_indices, evaluations['leak_detected'], s=50, c=colors)
    ax.set_xticks(scen_indices)
    ax.set_xlabel('Scenarios')
    ax.set_yticks([0, 1], labels=[0, 1])
    ax.set_ylabel('Leak detected')

    # Second subplot
    ax = fig.add_subplot(122)
    colors = np.where(evaluations['leak_detected']==1, 'blue', 'red')

    ax.scatter(scen_indices, evaluations['detection_time']/365.25, s=50, c='black')
    ax.set_xticks(scen_indices)
    ax.set_xlabel('Scenarios')
    ax.set_ylabel('Detection time, [years]')

    fig.tight_layout()

    # Plot average NRMS
    ave_NRMS = out['smt.ave_NRMS']
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    for sind in range(num_scenarios):
        ax.plot(time_array/365.25, ave_NRMS[sind, :], '-', linewidth=2)
    ax.set_xlabel('Time, [years]')
    ax.set_ylabel('Average NRMS')
