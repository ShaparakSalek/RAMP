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
from ramp.data_container import DataContainer, default_bin_file_reader
from ramp.plume_estimate import PlumeEstimate

if __name__ == "__main__":

    # Define keyword arguments of the system model
    final_year = 200
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'velocity'
    # Path to the data files
    data_directory = os.path.join('..', '..', 'data', 'user', 'velocity')
    output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                    'ramp_sys_plume_estimate_lhs_criteria2')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    # Data reader to be used to read data files
    data_reader = default_bin_file_reader
    # Keyword arguments for reader specifying expected data shape and reformatting needed
    data_reader_kwargs = {'data_shape': (141, 401),
                          'move_axis_destination': [-1, -2]}
    # Time points at which data is available, in years
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)

    # Select scenarios to link to the data container
    scenarios = list(range(1, 36))
    num_scenarios = len(scenarios)
    # Setup dictionary needed for data container setup
    family = 'velocity'
    data_setup = {}
    for ind in scenarios:
        data_setup[ind] = {'folder': os.path.join('vp_sim{:04}'.format(ind),
                                                  'model')}
        for t_ind in range(1, num_time_points+1):
            data_setup[ind]['t{}'.format(t_ind)] = 'model_sim{:04}_t{}.bin'.format(ind, t_ind*10)
    baseline = True

    # ------------- Create system model -------------
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # ------------- Add data container -------------
    dc = sm.add_component_model_object(
        DataContainer(name='dc', parent=sm, family=family, obs_name=obs_name,
                      data_directory=data_directory, data_setup=data_setup,
                      time_points=time_points, baseline=baseline,
                      data_reader=data_reader,
                      data_reader_kwargs=data_reader_kwargs,
                      presetup=True))
    # Add parameters of the container
    dc.add_par('index', value=1,
               discrete_vals=[scenarios,
                              num_scenarios*[1/num_scenarios]])
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    coordinates = {1: 4000 + 10*np.arange(401),
                   2: 10*np.arange(141)}  # 1 is x, 2 is z
    # ------------- Add plume estimate component -------------
    max_num_plumes = 2
    plest = sm.add_component_model_object(
        PlumeEstimate(name='plest', parent=sm, coordinates=coordinates, size=100,
                      criteria=2, max_num_plumes=max_num_plumes))
    # Add keyword arguments linked to the data container outputs
    plest.add_kwarg_linked_to_obs('data', dc.linkobs['delta_{}'.format(obs_name)],
                                  obs_type='grid')
    # Add threshold parameter
    plest.add_par('threshold', value=100.0, vary=False)
    plume_needed = False
    if plume_needed:
        # Add gridded observation
        plest.add_grid_obs('plume', constr_type='matrix', output_dir=output_directory)
    # Add scalar observation
    plest.add_obs('num_plumes')
    # Add observations related to each dimension
    for nm in ['min1', 'min2', 'max1', 'max2', 'extent1', 'extent2', 'plume_size']:
        plest.add_grid_obs(nm, constr_type='array', output_dir=output_directory)

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

    # Collect gridded observations
    time_indices = list(range(num_time_points))
    plume_metrics = {}
    for nm in ['min1', 'min2', 'max1', 'max2', 'extent1', 'extent2', 'plume_size']:
        plume_metrics[nm] = np.zeros((num_scenarios, num_time_points, max_num_plumes))
        print('Results for plume metric:', nm)
        for rlzn_number in range(num_scenarios):
            print('Realization {}'.format(rlzn_number+1))
            plume_metrics[nm][rlzn_number, :, :] = \
                sm.collect_gridded_observations_as_time_series(
                    plest, nm, output_directory, indices=time_indices,
                    rlzn_number=rlzn_number+1)

    if plume_needed:
        # Collect plume
        plume = np.zeros((num_scenarios, num_time_points, 401, 141))
        for rlzn_number in range(num_scenarios):
            print('Realization {}'.format(rlzn_number))
            data = sm.collect_gridded_observations_as_time_series(
                plest, 'plume', output_directory, indices=time_indices,
                rlzn_number=rlzn_number+1)  # data shape (num_time_points, num_sources, num_receivers)
            plume[rlzn_number-1] = data

    print('--------------------------------')
    print('Plotting results...')
    print('--------------------------------')
    # Plot results
    # Close all open figures (from previous simulations)
    plt.close('all')
    scen_indices = np.arange(1, num_scenarios+1)
    # Plot extents of plumes for all scenarios
    years_ticks = np.linspace(20, 200, num=10)
    for plume_ind in [0, 1]:
        fig = plt.figure(figsize=(10, 8))
        # First subplot
        ax = fig.add_subplot(211)
        for ind in scen_indices:
            ax.plot(time_array/365.25, plume_metrics['extent1'][ind-1][:, plume_ind], 'o')
        ax.set_xlabel('Time, [years]')
        ax.set_xticks(years_ticks)
        ax.set_ylabel('Extent of plume {} in x-direction, [m]'.format(plume_ind+1))

        # Second subplot
        ax = fig.add_subplot(212)

        for ind in scen_indices:
            ax.plot(time_array/365.25, plume_metrics['extent2'][ind-1][:, plume_ind], 'o')
        ax.set_xlabel('Time, [years]')
        ax.set_xticks(years_ticks)
        ax.set_ylabel('Extent of plume {} in z-direction, [m]'.format(plume_ind+1))
        fig.tight_layout()

    # Box plots
    xlabels = np.linspace(10, 200, num=20, dtype=int)
    for plume_ind in [0, 1]:
        fig = plt.figure(figsize=(10, 8))
        # First subplot
        ax = fig.add_subplot(211)
        ax.boxplot(plume_metrics['extent1'][:, :, plume_ind],
                    vert=True,         # vertical box alignment
                    patch_artist=True, # fill with color
                    labels=xlabels)
        ax.set_xlabel('Time, [years]')
        ax.set_ylabel('Extent of plume {} in x-direction, [m]'.format(plume_ind+1))
        ax.yaxis.grid(True)

        # Second subplot
        ax = fig.add_subplot(212)
        ax.boxplot(plume_metrics['extent2'][:, :, plume_ind],
                    vert=True,         # vertical box alignment
                    patch_artist=True, # fill with color
                    labels=xlabels)
        ax.set_xlabel('Time, [years]')
        ax.set_ylabel('Extent of plume {} in z-direction, [m]'.format(plume_ind+1))
        ax.yaxis.grid(True)
        fig.tight_layout()
