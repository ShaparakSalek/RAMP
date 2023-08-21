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
from ramp import DataContainer
from ramp.utilities.data_readers import default_bin_file_reader


if __name__ == "__main__":

    # Define keyword arguments of the system model
    t0 = 10.0
    time_point = t0*365.25
    sm_model_kwargs = {'time_point': time_point}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'velocity'
    # Path to the data files
    data_directory = os.path.join('..', '..', 'data', 'user', 'sensitivity')
    output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                    'ramp_sys_sensitivity')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # Data reader to be used to read data files
    data_reader = default_bin_file_reader

    # Time points at which data is available, in years
    time_points = np.array([t0])
    num_time_points = len(time_points)

    # Select scenarios to link to the data container
    scenarios = [1]
    num_scenarios = len(scenarios)
    # Setup dictionary needed for data container setup
    family = 'velocity'
    obs_names = ['sensitivity', 'velocity_p', 'velocity_s', 'density']
    file_names = ['sensitivity_surface_n1550_single_precision_little_endian.bin',
                  'hess_vp.bin', 'hess_vs.bin', 'hess_iso_rho.bin']
    data_setup = {key: {} for key in obs_names}
    for obs_ind, obs_nm in enumerate(obs_names):
        for ind in scenarios:
            data_setup[obs_nm][ind] = {'folder': ''}
            for t_ind in range(1, num_time_points+1):
                data_setup[obs_nm][ind]['t{}'.format(t_ind)] = file_names[obs_ind]
    baseline = False

    # ------------- Create system model -------------
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # ------------- Add data container -------------
    data_reader_kwargs = None
    dc = {}
    for obs_ind, obs_nm in enumerate(obs_names):
        if obs_ind >= 1:
            # Keyword arguments for reader specifying expected data shape and reformatting needed
            data_reader_kwargs = {'data_shape': (250, 600),
                                  'move_axis_destination': [-1, -2]}

        dc[obs_nm] = sm.add_component_model_object(
            DataContainer(name='dc{}'.format(obs_ind+1), parent=sm, family=family,
                          obs_name=obs_nm, data_directory=data_directory,
                          data_setup=data_setup[obs_nm],
                          time_points=time_points, baseline=baseline,
                          data_reader=data_reader,
                          data_reader_kwargs=data_reader_kwargs))

        # Add parameters of the container
        dc[obs_nm].add_par('index', value=1, vary=False)
        # Add gridded observation
        dc[obs_nm].add_grid_obs(obs_nm, constr_type='', output_dir=output_directory)

    coordinates = {'x': 8*np.arange(600), 'z': 8*np.arange(250)}

    print('-----------------------------')
    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()
    print('-----------------------------')
    print('Forward simulation finished.')
    print('-----------------------------')
    print('-----------------------------')
    print('Collecting outputs...')
    print('-----------------------------')

    # Get saved gridded observations from files
    outputs = {}
    for obs_ind, obs_nm in enumerate(obs_names):
        outputs[obs_nm] = sm.collect_gridded_observations_as_time_series(
        dc[obs_nm], obs_nm, output_directory, indices=[0], rlzn_number=0)[0]

    # Check shape of outputs
    for obs_nm in obs_names:
        print(obs_nm, outputs[obs_nm].shape)

    print('--------------------------------')
    print('Plotting results...')
    print('--------------------------------')

    # Plot sensitivity curve
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.plot(4 * np.arange(len(outputs['sensitivity'])), outputs['sensitivity'])
    ax.set_xlabel('x, [m]');
    ax.set_ylabel('Sensitivity')
    ax.grid()

    # Plot the rest of the outputs
    cbars = []
    y = np.array([0, 100, 200])
    ylabels = 8*y
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(5.5, 8), sharex=True)
    for obs_ind, obs_nm in enumerate(obs_names[1:]):
        pic = axs[obs_ind].imshow(outputs[obs_nm].T)
        axs[obs_ind].set_ylabel('depth, [m]')
        axs[obs_ind].set_yticks(y, labels=ylabels)
        cbars.append(fig.colorbar(pic, ax=axs[obs_ind]))
    x = np.linspace(0, 600, num=7, dtype=int)
    xlabels = 8*x
    axs[2].set_xticks(x, labels=xlabels)
    axs[2].set_xlabel('x, [m]')
    cbars[0].set_label(label='velocity, Vp [m/s]')
    cbars[1].set_label(label='velocity, Vs [m/s]')
    cbars[2].set_label(label=r'density, $\rho$ [kg/m$^3$]')
