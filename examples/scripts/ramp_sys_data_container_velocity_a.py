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
    final_year = 90
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'velocity'
    data_directory = os.path.join('..', '..', 'data', 'user', 'velocity')
    output_directory = os.path.join('..', 'user',
                                    'output', 'ramp_sys_data_container_velocity_a')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (141, 401),
                          'move_axis_destination': [-1, -2]}
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)
    num_scenarios = 2
    family = 'velocity'
    data_setup = {}
    for ind in range(1, num_scenarios+1):
        data_setup[ind] = {'folder': os.path.join('vp_sim{:04}'.format(ind), 'model')}
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
                      ))
    # Add parameters of the container
    dc.add_par('index', value=1, vary=False)
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()

    # Get saved data from files
    data = sm.collect_gridded_observations_as_time_series(
        dc, 'delta_velocity', output_directory, indices=[7], rlzn_number=0)[0]

    # Plot results
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    zs = 10*np.arange(141)
    xs = 4000 + 10*np.arange(401)
    xxs, zzs = np.meshgrid(xs, zs, indexing='ij')

    pic = ax.scatter(xxs, zzs, c=data, cmap='viridis', marker='s',
                vmin=np.min(data), vmax=np.max(data))
    ax.invert_yaxis()
    ax.set_title('Change in velocity at t = 80 years')
    ax.set_ylabel('depth, [m]')
    ax.set_xlabel('x, [m]')
    fig.colorbar(pic, ax=ax)
