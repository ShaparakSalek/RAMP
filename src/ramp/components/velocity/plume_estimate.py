# -*- coding: utf-8 -*-
"""
Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openiam import SystemModel
from ramp.components.base import DataContainer
from ramp.utilities.data_readers import default_bin_file_reader
from ramp.components.base import MonitoringTechnology


class PlumeEstimate(MonitoringTechnology):

    def __init__(self, name, parent, coordinates=None, data_shape=None, size=1,
                 criteria=1, max_num_plumes=3, no_plume_flag=-999.0):
        """
        Constructor of PlumeEstimate class.

        Parameters
        ----------
        name : str
            Name of PlumeEstimate instance under which it will be known to its parent
        parent : SystemModel class instance
            System model to which the instance belongs
        coordinates : dict
            Coordinates associated with data. Possible keys: 1, 2, or 1, 2, 3
            1 corresponds to coordinates for data in the first dimension
            2 corresponds to coordinates for data in the second dimension
            3 corresponds to coordinates for data in the third dimension, and so on.
            For example, coordinates = {1: np.linspace(1, 11, num=4),
                                        2: np.linspace(0, 100, num=11)}
            If coordinates is None, then the coordinates are assumed to be indices
            corresponding to the number of points in each dimension of data.
        data_shape : tuple
            Tuple representing size of data in each dimension
        size : float
            Area or volume associated with each point of data. It is used to
            estimate the size of plume in area or volume terms. The estimate works
            when all the coordinates are uniformly spaced in each dimension
        criteria : int
            Flag variable indicating the way a data will be compared to a threshold
            Possible values:
                1 - data will be compared to a threshold
                2 - absolute value of data will be compared to a threshold
                3 - relative comparison will be performed: baseline data will
                be subtracted from data and divided by baseline to see where
                the relative change exceeds the threshold value.

        Returns
        -------
        Instance of PlumeEstimate class.

        """
        # Setup keyword arguments of the 'model' method provided by the system model
        model_kwargs = {'time_point': 0.0}  # default value of 0 days
        super().__init__(name, parent, model_kwargs=model_kwargs)

        # Setup keys of gridded observation
        # plume is matrix of the same size as input data
        self.grid_obs_keys = ['plume']

        # Add type attribute
        self.class_type = 'PlumeEstimate'

        # Get coordinates
        self.process_coordinates(coordinates, data_shape)

        # Criteria according to which the data or derivative of data will be
        # compared to a threshold
        self.criteria = criteria

        # Set maximum number of plume to be analyzed (arranged by size)
        self.max_num_plumes = max_num_plumes

        # Set value that would indicate that no plume is detected for
        # indices and coordinates observations
        self.no_plume_flag = no_plume_flag

        # Setup size
        self.cell_size = size  # can be width/length for 1d, area for 2d data, and volume for 3d data

        # Add default threshold parameter
        self.add_default_par('threshold', value=0.0)

    def process_coordinates(self, coordinates=None, data_shape=None):
        if coordinates is None:
            if data_shape is not None:
                self.num_dims = len(data_shape)
                self.coordinates = {ind: np.arange(0, data_shape[ind-1]) \
                                    for ind in range(1, self.num_dims+1)}
                self.max_distance = 1.42  # slightly larger than sqrt(2)=1.4142 but less than 2
            else:
                self.coordinates = None
                self.num_dims = 0
                self.max_distance = 0.0
                self.grid = None
        else:
            self.num_dims = len(coordinates.keys())
            self.coordinates = {ind: coordinates[ind] for ind in range(1, self.num_dims+1)}
            # Distance between consecutive coordinates
            dist = np.array([coordinates[ind][1] - coordinates[ind][0] \
                             for ind in range(1, self.num_dims+1)])
            # Largest distance along each dimension multipled by 1.42
            # (sqrt(2) if distances are the same along each dimension)
            self.max_distance = np.max(1.42*dist)

        # Setup grid of coordinates
        if self.coordinates is not None:
            self.grid = np.meshgrid(
                *(self.coordinates[ind] for ind in range(1, self.num_dims+1)),
                indexing='ij')

    def process_data(self, p, time_point=None, data=None, baseline=None):
        """
        Compare data to the user provided threshold and return masked data, as
        well as maximum extent of plume in each of the dimension as well as
        coordinates of plume extent.

        If the coordinates associated with data were not provided when the
        instance was created, the coordinates will be calculated based
        on the data shape as indices.

        It is possible that no data points are above the user defined threshold,
        in this case, extents are zeros. Coordinates of the left and right boundaries
        are assumed to be np.nan. This is done so that it is possible
        to calculate statistics of values as numpy can ignore nan values.

        If there is a single point at which the data is above the threshold,
        extent is zero but min and max coordinates are the same.

        Parameters
        ----------
        p : dict
            Parameters of component
        time_point : float
            time point (in days) for which the component outputs
            are to be calculated; by default, its value is 0 days
        data : numpy.ndarray of shape (n1, n2) or (n1, n2, n3)
            Data to be processed.
        baseline : numpy.ndarray of shape (n1, n2) or (n1, n2, n3)
            Baseline data to be processed

        Returns
        -------
        Dictionary of outputs. Possible keys:
            plume: numpy.ndarray of the same shape as the original data
            plume_size : number of points with data above the threshold multipled
            by cell size
            extent1, extent2, ... - maximum extent of the plume along each dimension
            min1, min2, ... - minimum of coordinates along each dimension when
            extent is positive, and if it's -999.0 then extent is zero
            max1, max2, ... - maximum of coordinates along each dimension when
            extent is positive, and if it's -999.0 then extent is zero.
            min_ind1, min_ind2, ... - index of coordinates along each dimension
            corresponding to the minimum of coordinates, and if it's -999
            then extent is zero
            max_ind1, max_ind2, ... - index of coordinates along each dimension
            corresponding to the maximum of coordinates and if it's -999
            then extent is zero

        """
        # Obtain the default values of the parameters from dictionary of default parameters
        actual_p = {k: v.value for k, v in self.default_pars.items()}
        # Update default values of parameters with the provided ones
        actual_p.update(p)

        # Check whether data coordinates were setup before
        if self.coordinates is None:
            self.process_coordinates(coordinates=None, data_shape=data.shape)

        out = {}

        threshold = actual_p['threshold']

        out['plume'] = np.zeros(data.shape)
        if self.criteria == 1:
            selected_indices = np.where(data >= threshold)
            out['plume_data'] = data
        elif self.criteria == 2:
            selected_indices = np.where(np.abs(data) >= threshold)
            out['plume_data'] = np.abs(data)
        elif self.criteria == 3:
            if baseline is None:
                err_msg = ''.join([
                    'For a selected criteria (option 3) the output cannot be ',
                    'calculated as baseline data is not provided.'])
                logging.error(err_msg)
                raise KeyError(err_msg)
            else:
                data_ratio = np.abs((data - baseline)/baseline)*100.0
                selected_indices = np.where(data_ratio >= threshold)
                out['plume_data'] = data_ratio

        out['plume'][selected_indices] = 1.0

        if selected_indices[0].size > 0: # non-empty array
            out.update(self.present_plume_output(selected_indices))
        else:
            out.update(self.no_plume_output())

        return out

    def setup_data_for_clustering_analysis(self, where_plume_is_one_indices):
        coords = [self.grid[ind][where_plume_is_one_indices] for ind in range(self.num_dims)]
        data = np.vstack(coords).T
        return data

    def present_plume_output(self, where_plume_is_one_indices):
        cluster_data = self.setup_data_for_clustering_analysis(where_plume_is_one_indices)
        # Split points into clusters
        results = DBSCAN(eps=self.max_distance, min_samples=3).fit(cluster_data)

        # Get labels of the cluster
        all_labels = results.labels_
        # Get unique labels
        unique_labels, label_counts = np.unique(
            all_labels, return_counts=True)

        if unique_labels[0] == -1:
            unique_labels = unique_labels[1:]
            label_counts = label_counts[1:]

        # Sort label_counts from largest count to smallest
        arg_sort_counts = np.argsort(label_counts)[::-1]
        label_counts = label_counts[arg_sort_counts]
        unique_labels = unique_labels[arg_sort_counts]

        # Initialize outputs dictionary
        out = {'num_plumes': len(unique_labels),  # total number of detected plumes
               'plume_size': np.zeros(self.max_num_plumes)}
        # Initialize additional outputs: assign default values of either -999 or 0
        for ind in range(1, self.num_dims + 1):
            out['min{}'.format(ind)] = self.no_plume_flag*np.ones(self.max_num_plumes)
            out['max{}'.format(ind)] = self.no_plume_flag*np.ones(self.max_num_plumes)
            out['extent{}'.format(ind)] = np.zeros(self.max_num_plumes)

        # Determine whether max of the number of plumes is smalles or larger
        # than the number of detected plumes
        min_num_plumes = min(out['num_plumes'], self.max_num_plumes)

        coords_data = {}
        for plume_ind in range(min_num_plumes):
            # Size of plume of a current index is product of number of points
            # with a particular label and cell size
            out['plume_size'][plume_ind] = label_counts[plume_ind]*self.cell_size

            # Get plume label (cluster id)
            label_val = unique_labels[plume_ind]

            # Get coordinates of the points belonging to the plume with a given label
            coords_data[plume_ind] = [cluster_data[:, ind][np.where(all_labels==label_val)] \
                                      for ind in range(self.num_dims)]

            # Determine min and max in each dimension, and extent of the plume
            for ind in range(1, self.num_dims + 1):
                out['min{}'.format(ind)][plume_ind] = np.min(coords_data[plume_ind][ind-1])
                out['max{}'.format(ind)][plume_ind] = np.max(coords_data[plume_ind][ind-1])
                out['extent{}'.format(ind)][plume_ind] = \
                    (out['max{}'.format(ind)][plume_ind] - out['min{}'.format(ind)][plume_ind])

        return out

    def no_plume_output(self):
        out = {}
        out['num_plumes'] = 0
        out['plume_size'] = np.zeros(self.max_num_plumes)
        for ind in range(1, self.num_dims + 1):
            # out['min_ind{}'.format(ind)] = self.no_plume_flag*np.ones(self.max_num_plumes)
            # out['max_ind{}'.format(ind)] = self.no_plume_flag*np.ones(self.max_num_plumes)
            out['min{}'.format(ind)] = self.no_plume_flag*np.ones(self.max_num_plumes)
            out['max{}'.format(ind)] = self.no_plume_flag*np.ones(self.max_num_plumes)
            out['extent{}'.format(ind)] = np.zeros(self.max_num_plumes)

        return out

def plume_scatter_plot(xxs, zzs, data, xlabel, ylabel, title, colorbar_label,
                       vmin, vmax, cmap='viridis', marker='s', labelsize=14):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    pic = ax.scatter(xxs, zzs, c=data, cmap=cmap, marker=marker,
                     vmin=vmin, vmax=vmax)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=labelsize+1)
    ax.set_ylabel(ylabel, fontsize=labelsize+1)
    ax.set_title(title, fontsize=labelsize+3)

    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    cbar = fig.colorbar(pic, ax=ax)
    cbar.ax.tick_params(labelsize=labelsize-1)
    cbar.set_label(label=colorbar_label, size=labelsize)
    fig.tight_layout()
    return fig, ax


def plume_scatter_4x5_plot(xxs, zzs, data, xlabel, ylabel, title, colorbar_label,
                           vmin=None, vmax=None, cmap='viridis', marker='s', labelsize=14):
    if vmin is None and vmax is None:
            vmin = np.min(data)
            vmax = np.max(data)

    fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(18, 10), sharex=True)
    for ind in range(20):
        row = ind//5
        col = ind%5
        pic = axs[row, col].scatter(xxs, zzs, c=data[ind],
                                    cmap=cmap, marker=marker,
                                    vmin=vmin, vmax=vmax)
        axs[row, col].invert_yaxis()
        axs[row, col].set_title('t = {} years'.format((ind+1)*10))
    cax = plt.axes([0.92, 0.1, 0.025, 0.8])
    cbar = fig.colorbar(pic, ax=axs, cax=cax)
    cbar.ax.tick_params(labelsize=labelsize-1)
    cbar.set_label(label=colorbar_label, size=labelsize)

    for row in range(4):
        axs[row, 0].set_ylabel(ylabel)
    for col in range(5):
        axs[3, col].set_xlabel(xlabel)
    fig.suptitle(title)
    return fig, axs


def test_plume_estimate():
    # Define keyword arguments of the system model
    final_year = 200
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'velocity'
    data_directory = os.path.join('..', '..', '..', '..', 'data', 'user', 'velocity')
    output_directory = os.path.join(
        '..', '..', '..', '..', 'examples', 'user', 'output', 'test_plume_estimate')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (141, 401),
                          'move_axis_destination': [-1, -2]}
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)
    selected_scenario = 76
    scenarios = list(range(selected_scenario, selected_scenario+1))
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
    dc.add_par('index', value=scenarios[0], vary=False)
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    # ------------- Add plume estimate component -------------
    coordinates = {1: 4000 + 10*np.arange(401),
                   2: 10*np.arange(141)}  # 1 is x, 2 is z
    criteria = 3
    plest = sm.add_component_model_object(
        PlumeEstimate(name='plest', parent=sm, coordinates=coordinates, size=100,
                      criteria=criteria))  # criteria 2 - compare absolute value
    # Add keyword arguments linked to the data container outputs
    if criteria in [1, 2]:
        threshold = 100.0
        plest.add_kwarg_linked_to_obs(
            'data', dc.linkobs['delta_{}'.format(obs_name)], obs_type='grid')
    elif criteria == 3:
        threshold = 5.0
        plest.add_kwarg_linked_to_obs(
            'data', dc.linkobs[obs_name], obs_type='grid')
        plest.add_kwarg_linked_to_obs(
            'baseline', dc.linkobs['baseline_{}'.format(obs_name)], obs_type='grid')

    # Add threshold parameter
    plest.add_par('threshold', value=threshold, vary=False)
    # Add gridded observations
    plest.add_grid_obs('plume', constr_type='matrix', output_dir=output_directory)
    if criteria == 3:
        plest.add_grid_obs('plume_data', constr_type='matrix', output_dir=output_directory)
    # Add observations related to each dimension
    for nm in ['min1', 'min2', 'max1', 'max2', 'extent1', 'extent2', 'plume_size']:
        plest.add_grid_obs(nm, constr_type='array', output_dir=output_directory)
    # Add scalar observations
    plest.add_obs('num_plumes')

    print('-----------------------------')
    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()
    print('-----------------------------')
    print('Forward simulation finished.')
    print('-----------------------------')

    print('--------------------------------')
    print('Collecting results...')
    print('--------------------------------')
    # Get saved data from files
    time_indices = list(range(num_time_points))

    velocity = sm.collect_gridded_observations_as_time_series(
        dc, 'velocity', output_directory, indices=time_indices, rlzn_number=0)
    delta_velocity = sm.collect_gridded_observations_as_time_series(
        dc, 'delta_velocity', output_directory, indices=time_indices, rlzn_number=0)
    plume = sm.collect_gridded_observations_as_time_series(
        plest, 'plume', output_directory, indices=time_indices, rlzn_number=0)
    if criteria in [1, 2]:
        plume_data = delta_velocity
    elif criteria == 3:
        plume_data = sm.collect_gridded_observations_as_time_series(
            plest, 'plume_data', output_directory, indices=time_indices, rlzn_number=0)

    plume_metrics = {}
    for obs in ['min1', 'min2', 'max1', 'max2', 'extent1', 'extent2']:
        plume_metrics[obs] = sm.collect_gridded_observations_as_time_series(
            plest, obs, output_directory, indices=time_indices, rlzn_number=0)
    # Collect scalar observation
    plume_metrics['num_plumes'] = sm.collect_observations_as_time_series(
        plest, 'num_plumes')

    print('--------------------------------')
    print('Plotting results...')
    print('--------------------------------')

    # Plot results
    # ============= Single time point plots ================
    # Velocity data difference
    xs = coordinates[1]
    zs = coordinates[2]
    xxs, zzs = np.meshgrid(xs, zs, indexing='ij')
    time_ind = 19
    fig, ax = plume_scatter_plot(
        xxs, zzs, delta_velocity[time_ind],
        xlabel='x, [m]', ylabel='depth, [m]',
        title='Change in velocity at t = {} years (Scenario {})'.format(
            (time_ind+1)*10, selected_scenario),
        colorbar_label='[m/s]',
        vmin=np.min(delta_velocity[time_ind]),
        vmax=np.max(delta_velocity[time_ind]),
        cmap='viridis', marker='s', labelsize=14)

    # Velocity data
    fig, ax = plume_scatter_plot(
        xxs, zzs, velocity[time_ind],
        xlabel='x, [m]', ylabel='depth, [m]',
        title='Velocity at t = {} years (Scenario {})'.format(
            (time_ind+1)*10, selected_scenario),
        colorbar_label='[m/s]',
        vmin=np.min(velocity[time_ind]),
        vmax=np.max(velocity[time_ind]),
        cmap='viridis', marker='s', labelsize=14)

    # Plume with plume data above the threshold with surrounding boxes
    fig, ax = plume_scatter_plot(
        xxs, zzs, plume[time_ind],
        xlabel='x, [m]', ylabel='depth, [m]',
        title='Plume above threshold ({}) at t = {} years (Scenario {})'.format(
            threshold, (time_ind+1)*10, selected_scenario),
        colorbar_label='',
        vmin=0, vmax=1,
        cmap='binary', marker='s', labelsize=14)

    for ind in range(min(3, plume_metrics['num_plumes'][time_ind])):
        coord1 = [plume_metrics['min1'][time_ind][ind],
                  plume_metrics['max1'][time_ind][ind],
                  plume_metrics['max1'][time_ind][ind],
                  plume_metrics['min1'][time_ind][ind],
                  plume_metrics['min1'][time_ind][ind]]
        coord2 = [plume_metrics['min2'][time_ind][ind],
                  plume_metrics['min2'][time_ind][ind],
                  plume_metrics['max2'][time_ind][ind],
                  plume_metrics['max2'][time_ind][ind],
                  plume_metrics['min2'][time_ind][ind]]
        ax.plot(coord1, coord2, '-b')

    # # ====================== Scatter 4x5 plots =======================
    # # Velocity data
    fig, axs = plume_scatter_4x5_plot(
        xxs, zzs, velocity, xlabel='x, [m]', ylabel='depth, [m]',
        title='Velocity data (Scenario {})'.format(selected_scenario),
        colorbar_label='[m/s]', cmap='turbo', marker='s', labelsize=14)

    # Plume data
    fig, axs = plume_scatter_4x5_plot(
        xxs, zzs, plume_data, xlabel='x, [m]', ylabel='depth, [m]',
        title='Plume data (|dV|/V0) (Scenario {})'.format(selected_scenario),
        colorbar_label='[%]', cmap='CMRmap_r', marker='s', labelsize=14)

    fig, axs = plume_scatter_4x5_plot(
        xxs, zzs, plume, xlabel='x, [m]', ylabel='depth, [m]',
        vmin=0, vmax=1, title='Plume (Scenario {})'.format(selected_scenario),
        colorbar_label='', cmap='binary', marker='s', labelsize=14)
    for t_ind in range(num_time_points):
        row = t_ind//5
        col = t_ind%5
        for ind in range(min(3, plume_metrics['num_plumes'][t_ind])):
            coord1 = [plume_metrics['min1'][t_ind][ind],
                      plume_metrics['max1'][t_ind][ind],
                      plume_metrics['max1'][t_ind][ind],
                      plume_metrics['min1'][t_ind][ind],
                      plume_metrics['min1'][t_ind][ind]]
            coord2 = [plume_metrics['min2'][t_ind][ind],
                      plume_metrics['min2'][t_ind][ind],
                      plume_metrics['max2'][t_ind][ind],
                      plume_metrics['max2'][t_ind][ind],
                      plume_metrics['min2'][t_ind][ind]]
            axs[row, col].plot(coord1, coord2, '-b')


if __name__ == "__main__":

    test_plume_estimate()
