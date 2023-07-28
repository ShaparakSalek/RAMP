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
import pandas

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.sep.join(['..', '..', 'source']))
from openiam import SystemModel

sys.path.insert(0, os.sep.join(['..', '..', 'ramp']))
from ramp.data_container import DataContainer, default_bin_file_reader
from ramp.plume_estimate import PlumeEstimate


def process_csv_file():
    # Process data about leaked masses of brine and CO2 to see which scenarios
    # we can consider
    # Scenarios [ 37 118 136 150 182 245 397 449 518 590 598 686 749 863 935 937 970 991] are not complete
    to_be_excluded1 = [37, 118, 136, 150, 182, 245, 397, 449, 518, 590, 598,
                       686, 749, 863, 935, 937, 970, 991]
    # Read csv file
    csv_file = 'co2-brine-leak-mass-kg_prod07-1km.csv'
    path_to_csv_file = os.path.join('..', '..', 'data', 'user', 'mass_leaked')

    p_data = pandas.read_csv(os.path.join(path_to_csv_file, csv_file))
    file_names = p_data['file_name'].to_numpy()
    brine_mass = p_data['brine_leak_mass'].to_numpy()
    co2_mass = p_data['co2_leak_mass'].to_numpy()
    num_rows = len(file_names)  # 19788 rows

    scen_indices = np.zeros(num_rows)
    for ind in range(num_rows):
        scen_indices[ind] = int(file_names[ind][10:14])

    num_scenarios = 991
    num_occurences = np.zeros(num_scenarios)
    for scen in range(1, num_scenarios+1):
        num_occurences[scen-1] = len(np.where(scen_indices==scen)[0])

    # 37, 118, 136, 150, 182, 245, 397, 449, 518, 590, 598, 686, 749, 863, 935, 937, 970
    to_be_excluded2 = (np.where(num_occurences!=20)[0]+1).tolist()

    excluded = set(to_be_excluded1).union(set(to_be_excluded2))

    to_consider = set(range(1, num_scenarios+1)).difference(excluded)
    to_consider = list(to_consider)

    # scenario, time, brine mass, co2 mass, num_plumes, extent1 x, extent1 z, extent2 x, extent2 z,
    # xmin1, xmax1, zmin1, zmax1, xmin2, xmax2, zmin2, zmax2
    num_time_points = 20
    data_to_populate = np.zeros((len(to_consider)*num_time_points, 17))  # 20 time points, 17 columns

    t_indices = list(range(20))

    for ind, scen in enumerate(to_consider):
        data_to_populate[ind*num_time_points:(ind+1)*num_time_points, 0] = scen
        data_to_populate[ind*num_time_points:(ind+1)*num_time_points, 1] = 10*np.arange(1, 21)
        indices = np.where(scen_indices==scen)[0][t_indices]
        data_to_populate[ind*num_time_points:(ind+1)*num_time_points, 2] = brine_mass[indices]
        data_to_populate[ind*num_time_points:(ind+1)*num_time_points, 3] = co2_mass[indices]

    return to_consider, data_to_populate

if __name__ == "__main__":

    all_scenarios, data_to_save = process_csv_file()
    scenarios = all_scenarios
    data_to_populate = data_to_save

    # Time points at which data is available, in years
    time_points = 10*np.arange(1, 21)
    # Define keyword arguments of the system model
    time_array = 365.25*time_points
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    # Observation name
    obs_name = 'velocity'
    # Path to the data files
    data_directory = os.path.join('..', '..', 'data', 'user', 'velocity')
    output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                    'ramp_sys_plume_estimate_optimization_data')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    # Data reader to be used to read data files
    data_reader = default_bin_file_reader
    # Keyword arguments for reader specifying expected data shape and reformatting needed
    data_reader_kwargs = {'data_shape': (141, 401),
                          'move_axis_destination': [-1, -2]}

    num_time_points = len(time_points)

    # Select scenarios to link to the data container
    num_scenarios = len(scenarios)
    # Setup dictionary needed for data container setup
    family = 'velocity'
    data_setup = {}
    for scen in scenarios:
        data_setup[scen] = {'folder': os.path.join('vp_sim{:04}'.format(scen),
                                                    'model')}
        for t_ind, tp in enumerate(time_points):
            data_setup[scen]['t{}'.format(t_ind+1)] = 'model_sim{:04}_t{}.bin'.format(scen, tp)
    # baseline is True if data is supposed to have baseline data
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
    dc.add_par('index', value=scenarios[0],
               discrete_vals=[scenarios,
                              num_scenarios*[1/num_scenarios]])
    # Add gridded observations
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)
    # dc.add_grid_obs('baseline_{}'.format(obs_name), constr_type='matrix',
    #                 output_dir=output_directory)

    # Setup x- and z-coordinates associated with data
    coordinates = {1: 4000 + 10*np.arange(401),
                   2: 10*np.arange(141)}  # 1 is x, 2 is z
    # ------------- Add plume estimate component -------------
    max_num_plumes = 2
    plest = sm.add_component_model_object(
        PlumeEstimate(name='plest', parent=sm, coordinates=coordinates, size=100,
                      criteria=3, max_num_plumes=max_num_plumes))
    # Add keyword arguments linked to the data container outputs
    plest.add_kwarg_linked_to_obs('data', dc.linkobs[obs_name], obs_type='grid')
    plest.add_kwarg_linked_to_obs('baseline', dc.linkobs['baseline_{}'.format(obs_name)],
                                  obs_type='grid')
    # Add threshold parameter
    plest.add_par('threshold', value=5.0, vary=False)  # 5%
    # Add gridded observations
    # "plume" observation is 0's and 1's
    plest.add_grid_obs('plume', constr_type='matrix', output_dir=output_directory)
    # # "plume_data" observation is data that was compared to threshold to produce
    # # "plume" observation. For criteria 3 it's a ratio of change in data over baseline data
    # # multiplied by 100 to get a percentage
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
    # Get scalar observations
    out = s.collect_observations_as_time_series()
    # out.keys() 'plest.num_plumes'

    time_indices = list(range(num_time_points))
    plume_metrics = {}
    for nm in ['min1', 'min2', 'max1', 'max2', 'extent1', 'extent2', 'plume_size']:
        plume_metrics[nm] = np.zeros((num_scenarios, num_time_points, max_num_plumes))
        for rlzn_number in range(num_scenarios):
            print('Realization {}'.format(rlzn_number))
            plume_metrics[nm][rlzn_number, :, :] = \
                sm.collect_gridded_observations_as_time_series(
                    plest, nm, output_directory, indices=time_indices,
                    rlzn_number=rlzn_number+1)

    # Populate numpy matrix with data to be saved
    for scen_ind, scen in enumerate(scenarios):
        for ind in range(num_time_points):
            row_ind = scen_ind*num_time_points + ind
            data_to_populate[row_ind, 4] = out['plest.num_plumes'][scen_ind, ind]
            data_to_populate[row_ind, 5] = plume_metrics['extent1'][scen_ind, ind, 0]
            data_to_populate[row_ind, 6] = plume_metrics['extent2'][scen_ind, ind, 0]
            data_to_populate[row_ind, 7] = plume_metrics['extent1'][scen_ind, ind, 1]
            data_to_populate[row_ind, 8] = plume_metrics['extent2'][scen_ind, ind, 1]
            data_to_populate[row_ind, 9] = plume_metrics['min1'][scen_ind, ind, 0]
            data_to_populate[row_ind, 10] = plume_metrics['max1'][scen_ind, ind, 0]
            data_to_populate[row_ind, 11] = plume_metrics['min2'][scen_ind, ind, 0]
            data_to_populate[row_ind, 12] = plume_metrics['max2'][scen_ind, ind, 0]
            data_to_populate[row_ind, 13] = plume_metrics['min1'][scen_ind, ind, 1]
            data_to_populate[row_ind, 14] = plume_metrics['max1'][scen_ind, ind, 1]
            data_to_populate[row_ind, 15] = plume_metrics['min2'][scen_ind, ind, 1]
            data_to_populate[row_ind, 16] = plume_metrics['max2'][scen_ind, ind, 1]

    header = ''.join([
        'scenario,time,brine_mass,co2_mass,num_plumes,xextent1,zextent1,',
        'xextent2,zextent2,xmin1,xmax1,zmin1,zmax1,xmin2,xmax2,zmin2,zmax2'])
    np.savetxt(os.path.join(output_directory, 'plume_metrics_data.csv'),
                data_to_populate, delimiter=",", header=header, comments='',
                fmt="%d," + "%1.1f,"+ "%1.3f,"*2 + "%d," + "%1.3f,"*11 + "%1.3f")

    # Collect plume
    # plume = np.zeros((num_scenarios, num_time_points, 401, 141))
    # velocity = np.zeros((num_scenarios, num_time_points, 401, 141))
    delta_velocity = np.zeros((num_scenarios, num_time_points, 401, 141))
    for rlzn_number in range(num_scenarios):
        print('Realization {}'.format(rlzn_number))
        # pdata = sm.collect_gridded_observations_as_time_series(
        #     plest, 'plume', output_directory, indices=time_indices,
        #     rlzn_number=rlzn_number+1) # data shape (num_time_points, num_sources, num_receivers)
        # plume[rlzn_number-1] = pdata

        # vdata = sm.collect_gridded_observations_as_time_series(
        #     dc, 'velocity', output_directory, indices=time_indices,
        #     rlzn_number=rlzn_number+1)
        # velocity[rlzn_number-1] = vdata

        dvdata = sm.collect_gridded_observations_as_time_series(
            dc, 'delta_velocity', output_directory, indices=time_indices,
            rlzn_number=rlzn_number+1)
        delta_velocity[rlzn_number-1] = dvdata

    # pfile_to_save = os.path.join(
    #     output_directory,
    #     'plume_data_{}_scenarios.npz'.format(num_scenarios))
    # np.savez_compressed(pfile_to_save, data=plume)

    # vfile_to_save = os.path.join(
    #     output_directory,
    #     'velocity_data_{}_scenarios.npz'.format(num_scenarios))
    # np.savez_compressed(vfile_to_save, data=velocity)

    dvfile_to_save = os.path.join(
        output_directory,
        'delta_velocity_data_{}_scenarios.npz'.format(num_scenarios))
    np.savez_compressed(dvfile_to_save, data=delta_velocity)

    to_plot = 0
    if to_plot:
        # Plot histograms of leaked masses
        # Brine
        labelsize = 12
        indices = num_time_points*np.arange(num_scenarios)
        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(18, 10))
        for ind in range(num_time_points):
            row = ind//5
            col = ind%5
            pic = axs[row, col].hist(np.log10(data_to_populate[indices+ind, 2]), 30, lw=1,
                                      ec="yellow", fc="green", alpha=0.5)
            axs[row, col].set_title('t = {} years'.format(time_points[ind]),
                                    fontsize=labelsize+2)
            axs[row, col].set_xlabel(r'log$_{10}$ mass', fontsize=labelsize+1)
            axs[row, col].set_xlim(2.5, 7)
            axs[row, col].set_ylim(top=40)
        fig.suptitle('Brine mass leaked', fontsize=labelsize+4)
        fig.tight_layout()
        fig.savefig(os.path.join(output_directory, 'brine_mass_leaked_hist.png'),
                    dpi=150)

        # CO2
        fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(18, 10))
        for ind in range(num_time_points):
            row = ind//5
            col = ind%5
            pic = axs[row, col].hist(np.log10(data_to_populate[indices+ind, 3]), 30, lw=1,
                                      ec="yellow", fc="green", alpha=0.5)
            axs[row, col].set_title('t = {} years'.format(time_points[ind]),
                                    fontsize=labelsize+2)
            axs[row, col].set_xlabel(r'log$_{10}$ mass', fontsize=labelsize+1)
            if ind > 0:
                axs[row, col].set_xlim(5.5, 9.5)
                axs[row, col].set_ylim(top=40)
        fig.suptitle(r'CO$_2$ mass leaked', fontsize=labelsize+4)
        fig.tight_layout()
        fig.savefig(os.path.join(output_directory, 'co2_mass_leaked_hist.png'),
                    dpi=150)

        # Print results for the first 5 scenarios
        num_dashes = 88
        print(num_dashes*'-')
        header_to_print = '  |  '.join([
            '|  Scenario', 'time', 'n_plumes', 'xextent1', 'zextent1',
            'xextent2', 'zextent2  |'])
        line_to_print = '  |  '.join(['|  {: >7} ', '{: >4}', '{: >8}', '{: >7} ', '{: >7} ',
                                    '{: >8}', '{: >8}  |'])
        for scen_ind, scen in enumerate(scenarios[0:5]):
            print(header_to_print)
            print(num_dashes*'-')
            for ind in range(num_time_points):
                print(line_to_print.format(
                    scen, time_points[ind],
                    out['plest.num_plumes'][scen_ind, ind],
                    plume_metrics['extent1'][scen_ind, ind, 0],
                    plume_metrics['extent2'][scen_ind, ind, 0],
                    plume_metrics['extent1'][scen_ind, ind, 1],
                    plume_metrics['extent2'][scen_ind, ind, 1]))
            print(num_dashes*'-')
