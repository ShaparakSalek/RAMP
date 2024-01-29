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
from ramp.utilities.data_readers import default_h5_file_reader


if __name__ == "__main__":

    # Define keyword arguments of the system model
    time_points = np.array([5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200])
    time_array = 365.25*time_points
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'gravity'
    # For this scenario pressure and gravity are kept in the same h5 files
    data_directory = os.path.join('..', '..', 'data', 'user', 'pressure')
    output_directory = os.path.join('..', 'user',
                                    'output', 'ramp_sys_data_container_gravity')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_h5_file_reader
    data_reader_kwargs = {'obs_name': obs_name}

    num_time_points = len(time_points)
    # scenarios = list(range(100, 116))
    scenarios = [3, 7, 10, 5, 9, 12, 6, 8]
    num_scenarios = len(scenarios)
    family = 'gravity'
    data_setup = {}
    for scen in scenarios:
        data_setup[scen] = {'folder': 'sim0001_0100'}
        for t_ind in range(1, num_time_points+1):
            data_setup[scen][f't{t_ind}'] = f'sim{scen:04}.h5'
    baseline = False

    # ------------- Create system model -------------
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # ------------- Add data container -------------
    dc = sm.add_component_model_object(
        DataContainer(name='dc', parent=sm, family=family, obs_name=obs_name,
                      data_directory=data_directory, data_setup=data_setup,
                      time_points=time_points, baseline=baseline,
                      data_reader=data_reader,
                      data_reader_kwargs=data_reader_kwargs,
                      data_reader_time_index=True
                      ))
    # Add parameters of the container
    index = scenarios[0]
    dc.add_par('index', value=index, vary=False)
    # Add gridded observation
    for nm in [obs_name]:
        dc.add_grid_obs(nm, constr_type='matrix', output_dir=output_directory)

    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()

    # Get saved gravity anomaly data from files
    gravity_data = sm.collect_gridded_observations_as_time_series(
        dc, 'gravity', output_directory, rlzn_number=0)  # (12, 41, 21)

    print('Gravity data shape:', gravity_data.shape)

    # Number of points in x- and y-directions
    nx = 41
    ny = 21

    # Gravity data associated x- and y-coordinates
    x = np.linspace(4000.0, 8000.0, nx)
    y = np.linspace(1500.0, 3500.0, ny)

    # Plot delta pressure and saturation z-slice plots
    dx = 5
    dy = 3
    cmap = 'jet'
    t_ind = 5
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax_im = ax.pcolormesh(x, y, gravity_data[t_ind, :, :].T, cmap=cmap)
    ax.set_title(f'Scenario {index}: gravity anomaly at t={time_points[t_ind]} years')
    plt.colorbar(ax_im, label='G, [uGal]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_ylabel('y, [m]')

    t_ind = 11
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax_im = ax.pcolormesh(x, y, gravity_data[t_ind, :, :].T, cmap=cmap)
    ax.set_title(f'Scenario {index}: gravity anomaly at t={time_points[t_ind]} years')
    plt.colorbar(ax_im, label='G, [uGal]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_ylabel('y, [m]')
