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
    final_year = 200
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'seismic'
    data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
    output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                    'ramp_sys_seismic_evaluation_forward_a')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (1251, 101, 9),
                          'move_axis_destination': [-1, -2, -3]}
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)
    scenarios = list(range(1, 6))
    num_scenarios = len(scenarios)
    family = 'seismic'
    data_setup = {}
    for ind in scenarios:
        data_setup[ind] = {'folder': os.path.join('data_sim{:04}'.format(ind),
                                                  'data')}
        for t_ind in range(1, num_time_points+1):
            data_setup[ind]['t{}'.format(t_ind)] = 'data_sim{:04}_t{}.bin'.format(
                ind, t_ind*10)
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
    scenario_index = scenarios[1]
    dc.add_par('index', value=scenario_index, vary=False)
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
    thresholds = {1: 20, 2: 2, 3: 1, 4: 20, 5: 5}
    threshold = thresholds[scenario_index]
    seval.add_par('threshold', value=threshold, vary=False)
    # Add keyword arguments linked to the seismic monitoring metric
    seval.add_kwarg_linked_to_obs('metric', smt.linkobs['ave_NRMS'])
    # Add observations
    for nm in ['leak_detected', 'detection_time']:
        seval.add_obs(nm)        # covers the whole simulation period
        seval.add_obs(nm + '_ts')  # covers how values change in time

    print('-----------------------------')
    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()
    print('-----------------------------')
    print('Forward simulation finished.')
    print('-----------------------------')
    print('-----------------------------')
    print('Plotting results...')
    print('-----------------------------')

    # Get saved gridded observations from files
    time_ind = final_year//10 - 1
    time_indices = list(range(num_time_points))
    delta_seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'delta_seismic', output_directory, indices=time_indices, rlzn_number=0)
    seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'seismic', output_directory, indices=time_indices, rlzn_number=0)
    nrms = sm.collect_gridded_observations_as_time_series(
        smt, 'NRMS', output_directory, indices=time_indices, rlzn_number=0)

    # Export scalar observations
    metrics = {}
    for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
        metrics[nm] = sm.collect_observations_as_time_series(smt, nm)

    evaluations = {}
    for nm in ['leak_detected', 'detection_time']:
        evaluations[nm] = sm.collect_observations_as_time_series(seval, nm)
        evaluations[nm + '_ts'] = sm.collect_observations_as_time_series(seval, nm + '_ts')

    # Plot results
    plt.close('all')

    # Plot difference in seismic data
    x = np.linspace(0, 100, num=6)
    xlabels = 4000 + np.linspace(0, 4000, num=6)
    y = np.linspace(0, 1200, num=5)
    ylabels = np.linspace(0, 2.5, num=5)
    source_ind = 2
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(18, 10),
                            sharex=True, sharey=True)
    for ind in range(num_time_points):
        row = ind//5
        col = ind%5
        axs[row, col].imshow(delta_seismic[ind][source_ind, :, :].T,
                             cmap='gray', aspect='auto')
        axs[row, col].set_title('t = {} years'.format((ind + 1)*10))

    for row in range(4):
        axs[row, 0].set_yticks(y, labels=ylabels)
        axs[row, 0].set_ylabel('Time, [sec]')
    for col in range(5):
        axs[3, col].set_xticks(x, labels=xlabels)
        axs[3, col].set_xlabel('Receiver location, [m]')

    fig.suptitle('Scenario {}: Seismic data difference (Source {})'.format(
        scenario_index, source_ind+1))
    fig.savefig('output/delta_seismic_scenario_{}_source_{}.png'.format(
        scenario_index, source_ind+1), dpi=150)

    # Plot NRMS as image
    x = np.linspace(0, 100, num=11)
    xlabels = np.linspace(1, 101, num=11, dtype=int)
    y = np.linspace(0, 8, num=9)
    ylabels = np.linspace(1, 9, num=9, dtype=int)

    ax_im = []
    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(18, 10),
                            sharex=True, sharey=True)
    for ind in range(num_time_points):
        row = ind//5
        col = ind%5
        ax_im.append(axs[row, col].imshow(nrms[ind], aspect='auto'))
        # Set title
        axs[row, col].set_title('t = {} years'.format((ind + 1)*10))
        # Add colorbar
        color_bar = fig.colorbar(ax_im[-1], ax=axs[row, col], orientation='vertical')
        color_bar.ax.set_ylabel('Percentage, %', fontsize=8)
        color_bar.ax.tick_params(axis='y', which='major', labelsize=8)

    # Set x-labels
    for col in range(5):
        axs[3, col].set_xticks(x, labels=xlabels)
        axs[3, col].set_xlabel('Receivers')

    # Set y-labels
    for row in range(4):
        axs[row, 0].set_yticks(y, labels=ylabels)
        axs[row, 0].set_ylabel('Sources')

    fig.suptitle('Scenario {}: NRMS'.format(scenario_index))
    fig.tight_layout()
    fig.savefig('output/nrms_scenario_{}.png'.format(scenario_index),
                dpi=150)

    # Plot time changes in NRMS metrics
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    for ind, (nm, obs_label) in enumerate(zip(['ave_NRMS', 'min_NRMS', 'max_NRMS'],
                                              ['Average', 'Minimum', 'Maximum'])):
        axs[ind].plot(time_points, metrics[nm], '-ok')
        axs[ind].set_xlabel('Time, [years]')
        axs[ind].set_ylabel(obs_label + ' NRMS')
    axs[0].plot([time_points[0], time_points[-1]], 2*[threshold], '--r')
    fig.tight_layout()
    fig.savefig('output/nrms_metrics_scenario_{}.png'.format(scenario_index),
                dpi=150)

    # Plot leak_detected metric
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    for ind, nm in enumerate(['leak_detected_ts', 'leak_detected']):
        colors = np.where(evaluations[nm]==1, 'blue', 'red')
        axs[ind].scatter(time_points, evaluations[nm], s=50, c=colors)
        try:
            mark_time_ind = np.where(evaluations[nm]==1)[0][0]
            axs[ind].plot(2*[time_points[mark_time_ind]], [-0.05, 1.1], '--', color='gray')
        except IndexError:
            pass
        axs[ind].set_xlabel('Time, [years]')
    axs[0].set_yticks([0, 1], labels=[0, 1])
    axs[0].set_ylabel('Leak detected (History)')
    axs[1].set_ylabel('Leak detected (Propagated)')
    fig.tight_layout()
    fig.savefig('output/leak_detected_scenario_{}.png'.format(scenario_index),
                dpi=150)

    # Print whether leak was detected and detection time if applicable
    final_obs_ind = final_year//10 - 1
    leak_detected = sm.obs['seval.leak_detected_{}'.format(final_obs_ind)].sim
    print('Is leak detected?', bool(leak_detected))
    if leak_detected:
        detection_time = sm.obs['seval.detection_time_{}'.format(final_obs_ind)].sim
        print('Leak is detected for the first time at t = {} days ({} years)'.format(
            detection_time, detection_time/365.25))
