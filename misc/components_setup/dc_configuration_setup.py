"""
Module contains script creating setup file for Seismic Data Container component.
"""
import os
import numpy as np
import pandas as pd


def create_seismic_data_setup(indices, filename='seismic_setup_file.csv',
                              t_filename='time_points.csv'):

    folders = np.array([os.sep.join(['data_sim{:04}'.format(ind),
                                     'data']) for ind in indices])

    setup_data = {'index': indices, 'folder': folders}
    for ind1 in range(20):
        setup_data['t{}'.format(ind1+1)] = np.array([
            'data_sim{:04}_t{}.bin'.format(ind2, (ind1+1)*10) for ind2 in indices])

    df = pd.DataFrame(data=setup_data)

    df.to_csv(path_or_buf=os.sep.join([
        '..', '..', 'data', 'user', 'seismic', filename]), sep=',', index=False)

    # Save time points
    time_points = 10*np.arange(1, 21)
    t_file_path = os.sep.join([
        '..', '..', 'data', 'user', 'seismic', t_filename])
    with open(t_file_path, 'w') as f:
        f.write(','.join([str(val) for val in time_points])+'\n')


def create_velocity_data_setup(indices, filename='velocity_setup_file.csv',
                               t_filename='time_points.csv'):
    folders = np.array([os.sep.join(['vp_sim{:04}'.format(ind), 'model']) for ind in indices])

    setup_data = {'index': indices, 'folder': folders}
    for ind1 in range(20):
        setup_data['t{}'.format(ind1+1)] = np.array([
            'model_sim{:04}_t{}.bin'.format(ind2, (ind1+1)*10) for ind2 in indices])

    df = pd.DataFrame(data=setup_data)

    df.to_csv(path_or_buf=os.sep.join([
        '..', '..', 'data', 'user', 'velocity', filename]), sep=',', index=False)

    # Save time points
    time_points = 10*np.arange(1, 21)
    t_file_path = os.sep.join([
        '..', '..', 'data', 'user', 'velocity', t_filename])
    with open(t_file_path, 'w') as f:
        f.write(','.join([str(val) for val in time_points])+'\n')


if __name__ == '__main__':
    start_ind = 1
    end_ind = 5
    indices = np.arange(start_ind, end_ind+1)

    create_seismic_data_setup(indices)
