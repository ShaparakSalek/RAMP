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
    obs_name = ['pressure', 'saturation']
    data_directory = os.path.join('..', '..', 'data', 'user', 'pressure')
    output_directory = os.path.join('..', 'user',
                                    'output', 'ramp_sys_data_container_pressure')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_h5_file_reader
    data_reader_kwargs = {'obs_name': obs_name}

    num_time_points = len(time_points)
    num_scenarios = 5
    family = 'insitu'
    data_setup = {}
    for ind in range(1, num_scenarios+1):
        data_setup[ind] = {'folder': 'sim0001_0100'}
        for t_ind in range(1, num_time_points+1):
            data_setup[ind]['t{}'.format(t_ind)] = 'sim{:04}.h5'.format(ind)
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
    dc.add_par('index', value=1, vary=False)
    # Add gridded observation
    for nm in obs_name:
        dc.add_grid_obs(nm, constr_type='matrix', output_dir=output_directory)

    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()

    # Get saved data from files
    # delta pressure, saturation
    dpressure_data = sm.collect_gridded_observations_as_time_series(
        dc, 'pressure', output_directory, rlzn_number=0)  # (12, 40, 20, 32)
    saturation_data = sm.collect_gridded_observations_as_time_series(
        dc, 'saturation', output_directory, rlzn_number=0)

    # Number of points in x-, y-, and z-directions
    nx = 40
    ny = 20
    nz = 32
    xmin, xmax = 4000, 8000
    ymin, ymax = 1500, 3500
    zmin, zmax = 0, 1410.80

    # vertex: vx, vy and vz
    # voxel center: x, y, z
    vx = np.linspace(xmin, xmax, nx+1)
    vy = np.linspace(ymin, ymax, ny+1)
    x = np.linspace((vx[0]+vx[1])/2.0, (vx[-2]+vx[-1])/2.0, nx)
    y = np.linspace((vy[0]+vy[1])/2.0, (vy[-2]+vy[-1])/2.0, ny)

    z = np.array([2.5, 7.5, 34.4, 83.1, 131.9, 180.6, 229.4, 278.1, 326.9,
                  375.6, 424.4, 473.1, 521.9, 570.5, 619.0, 667.5, 716.0,
                  764.5, 813.0, 861.5, 910.0, 958.5, 1007.0, 1055.5, 1104.0,
                  1152.5, 1201.0, 1248.5, 1295.0, 1341.5, 1376.3, 1399.6])

    # Plot delta pressure and saturation z-slice plots
    t_ind = 5
    z_ind = 25
    dx = 5
    dy = 4
    dz = 4
    cmap = 'hot_r'
    fig = plt.figure(figsize=(16, 6))
    # first subplot
    ax = fig.add_subplot(121)
    ax_im = ax.pcolormesh(x, y, dpressure_data[t_ind, :, :, z_ind].T, cmap=cmap)
    ax.set_title('Delta pressure at t={} years, depth={} m'.format(
        time_points[t_ind], z[z_ind]))
    plt.colorbar(ax_im, label='dP, [Pa]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_ylabel('y, [m]')

    # second subplot
    ax = fig.add_subplot(122)
    ax_im = ax.pcolormesh(x, y, saturation_data[t_ind, :, :, z_ind].T, cmap=cmap)
    ax.set_title('CO2 saturation at t={} years, depth={} m'.format(
        time_points[t_ind], z[z_ind]))
    plt.colorbar(ax_im, label='S, [-]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_ylabel('y, [m]')
    fig.tight_layout()

    # y-slice
    y_ind = 8
    fig = plt.figure(figsize=(16, 6))
    # first subplot
    ax = fig.add_subplot(121)
    ax_im = ax.pcolormesh(x, z, dpressure_data[t_ind, :, y_ind, :].T, cmap=cmap)
    ax.set_title('Delta pressure at t={} years, y={} m'.format(
        time_points[t_ind], y[y_ind]))
    plt.colorbar(ax_im, label='dP, [Pa]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(z[0:-1:dz], labels=z[0:-1:dz])
    ax.set_ylabel('z, [m]')
    ax.invert_yaxis()

    # second subplot
    ax = fig.add_subplot(122)
    ax_im = ax.pcolormesh(x, z, saturation_data[t_ind, :, y_ind, :].T, cmap=cmap)
    ax.set_title('CO2 saturation at t={} years, y={} m'.format(
        time_points[t_ind], y[y_ind]))
    plt.colorbar(ax_im, label='S, [-]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(z[0:-1:dz], labels=z[0:-1:dz])
    ax.set_ylabel('z, [m]')
    fig.tight_layout()
    ax.invert_yaxis()

    # x-slice
    x_ind = 6
    fig = plt.figure(figsize=(16, 6))
    # first subplot
    ax = fig.add_subplot(121)
    ax_im = ax.pcolormesh(y, z, dpressure_data[t_ind, x_ind, :, :].T, cmap=cmap)
    ax.set_title('Delta pressure at t={} years, x={} m'.format(
        time_points[t_ind], x[x_ind]))
    plt.colorbar(ax_im, label='dP, [Pa]')
    ax.set_xticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_xlabel('y, [m]')
    ax.set_yticks(z[0:-1:dz], labels=z[0:-1:dz])
    ax.set_ylabel('z, [m]')
    ax.invert_yaxis()

    # second subplot
    ax = fig.add_subplot(122)
    ax_im = ax.pcolormesh(y, z, saturation_data[t_ind, x_ind, :, :].T, cmap=cmap)
    ax.set_title('CO2 saturation at t={} years, x={} m'.format(
        time_points[t_ind], x[x_ind]))
    plt.colorbar(ax_im, label='S, [-]')
    ax.set_xticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_xlabel('y, [m]')
    ax.set_yticks(z[0:-1:dz], labels=z[0:-1:dz])
    ax.set_ylabel('z, [m]')
    ax.invert_yaxis()
    fig.tight_layout()
