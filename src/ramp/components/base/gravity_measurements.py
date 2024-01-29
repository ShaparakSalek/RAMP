# -*- coding: utf-8 -*-
"""
@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from ramp.components.base.monitoring_technology import MonitoringTechnology
from ramp.components.base import process_time_points, get_indices
from ramp.components.base.in_situ_measurements import (
    default_map_dim_index_2_point_index, default_map_point_index_2_dim_index)


def distance_based_argsort(point, points):
    """
    Calculate an array of indices sorting euclidean distances between a given point
    and a set of points in ascending order.

    Parameters
    ----------
    point : array-like
        DESCRIPTION.
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Reshape point just in case
    point = point.reshape((1, -1))

    # Calculate difference between point and points
    diff = points-point
    # Calculate distance as norm
    dist = np.linalg.norm(diff, axis=1)
    # Find indices sorting distance array in ascending order
    indices = np.argsort(dist)

    return indices


def get_multiple_measurements(dim_indices, data=None, baseline=None, criteria=1):

    # if data is provided
    if data is not None:
        data_point = data[dim_indices]
        # if baseline data is provided
        if baseline is not None:
            base_data_point = baseline[dim_indices]
            comp_data = data_point - base_data_point
        else:
            comp_data = data_point
        if criteria == 2:
            comp_data = np.abs(comp_data)

    return comp_data


class GravityMeasurements(MonitoringTechnology):

    def __init__(self, name, parent, config, time_points, dim_indices=None,
                 map_dim_index_2_point_index=None, map_point_index_2_dim_index=None,
                 num_neighbors=4, criteria=2):
        """
        The GravityMeasurements class is designed to model gravity monitoring
        at a given location and compare them to a threshold.

        Parameters
        ----------
        name : str
            Name of GravityMeasurements class instance under which it will
            be known to its parent
        parent : SystemModel class instance
            System model to which the instance belongs
        config : BaseConfiguration class instance
            Instance of BaseConfiguration class containing information
            about locations of receivers associated with an incoming linked data
        time_points : str or numpy.array, optional
            Path to the file containing time points associated with data
            if of type string.
            Array of time points associated with data, i.e.,
            time points at which data is provided if of type numpy.array
        dim_indices : int or tuple of int
            Integer index of the data point if data is 1d and tuple of indices
            for each dimension of data if data has 2 or more dimensions
        map_dim_index_2_point_index : method
            Method mapping list of indices into index of point receiver within
            provided Configuration instance config
        num_neighbors : int
            Number of additional data points closest to the selected receiver
            location at which the gravity value has to be above a threshold
            for a leak to be considered detected
        criteria : int
            Flag variable indicating how the data is compared to a threshold
            Value of 1 means data is compared as it is or difference (data-baseline)
            is compared to a threshold if baseline data is provided
            Value of 2 means an absolute value of data is compared or an absolute
            value of the difference (data-baseline) is compared to a threshold
            if baseline data is provided

        Returns
        -------
        Instance of GravityMeasurements class.

        """
        # Setup keyword arguments of the 'model' method provided by the system model
        model_kwargs = {'time_point': 0.0}  # default value of 0 days

        super().__init__(name, parent, model_kwargs=model_kwargs)

        # Add type attribute
        self.class_type = 'GravityMeasurements'

        # Setup additional attributes
        self.configuration = config
        if map_dim_index_2_point_index is None:
            self.index_map = default_map_dim_index_2_point_index
        else:
            self.index_map = map_dim_index_2_point_index

        if map_point_index_2_dim_index is None:
            self.inverse_index_map = default_map_point_index_2_dim_index
        else:
            self.inverse_index_map = map_point_index_2_dim_index

        # Determine what the data will be compared to a threshold
        self.criteria = criteria

        # Setup attribute data_indices
        self.get_indices(dim_indices=dim_indices)

        # Save number of neighbors to look at
        self.num_neighbors = num_neighbors

        # Setup index of the data point
        self.point_index = None  # not known until data is made available to the component

        # Set coordinates of the data point
        self.point_coords = []  # empty list

        # Add default threshold parameter
        self.add_default_par('threshold', value=0.0)

        # Process time_points argument
        self.time_points = process_time_points(
            time_points, data_directory=None,
            component_class=self.class_type, name=self.name)

        # Add accumulator that will keep track of the total time
        # to the first detection
        self.add_accumulator('leak_detected', sim=0)
        self.add_accumulator('detection_time', sim=np.inf)

        # TODO System model does not yet know how to handle those
        self.run_time_indices = get_indices(self._parent.time_array,
                                            self.time_points*365.25)

    def get_indices(self, dim_indices=None):
        if dim_indices is not None:
            if isinstance(dim_indices, (int, tuple)):
                self.data_indices = dim_indices
            else:
                err_msg = 'Argument dim_indices is of wrong type.'
                raise TypeError(err_msg)

    def get_coordinates(self, data_shape):
        if self.index_map is not None:
            # Get point index from dim_indices and data_shape
            self.point_index = self.index_map(self.data_indices, data_shape)
            # Check that the configuration is consistent with point_index
            if self.point_index < self.configuration.num_receivers:
                # Get coordinates
                self.point_coords = self.configuration.receivers.coordinates[
                    self.point_index]
            else:
                err_msg = 'Data shape is inconsistent with requested data point.'
                raise ValueError(err_msg)
        else:
            # If no map is provided
            self.point_coords = list(self.data_indices)

    def find_neighbors_indices(self, data_shape):
        """
        Find tuple indices of n closest data points where n is number of neighbors
        to consider.

        Returns
        -------
        None.

        """
        # Get coordinates of all receivers
        receiver_coords = self.configuration.receivers.coordinates

        # Calculate distance between point of interest and all receivers.
        # Then sort them based on the distance and take the first self.num_neighbors
        # receivers closest to the point of interest excluding the first receiver
        # in the list since it's the same as the point of interest
        neighbors_flat_indices = distance_based_argsort(
            self.point_coords, receiver_coords)[1:self.num_neighbors+1]

        # Transform indices into tuples of data indices
        self.neighbors_indices = self.inverse_index_map(
            neighbors_flat_indices, data_shape)

    def combine_indices(self):
        # Add data point indices to the end of the neighbor point indices
        if isinstance(self.data_indices, int):
            combined_indices = (np.append(self.neighbors_indices, self.data_indices),)
        elif isinstance(self.data_indices, tuple):
            combined_indices = tuple([
                np.append(self.neighbors_indices[ind], self.data_indices[ind])\
                    for ind in range(len(self.neighbors_indices))])

        return combined_indices

    def process_data(self, p, time_point=None, data=None, baseline=None,
                     dim_indices=None, **kwargs):
        """
        Compare provided data at a selected point to a threshold.

        Parameters
        ----------
        p : dict
            Parameters of component
        time_point : float
            time point (in days) for which the component outputs
            are to be calculated; by default, its value is 0 days
        data : numpy.ndarray
            Data to be processed
        baseline : numpy.ndarray of the same shape as data
            Baseline data to be processed
        dim_indices : int or list of int
            Integer index of the data point if data is 1d and list of indices
            for each dimension of data if data has 2 or more dimensions

        Returns
        -------
        None.

        """
        # Obtain the default values of the parameters from dictionary of default parameters
        actual_p = {k: v.value for k, v in self.default_pars.items()}
        # Update default values of parameters with the provided ones
        actual_p.update(p)

        # Check whether new indices were provided: if this is the case
        # attribute self.data_indices will be updated
        self.get_indices(dim_indices=dim_indices)

        # Initialize output dictionary
        out = {}

        time_index = np.where(self.time_points==time_point/365.25)[0][0]
        if time_index == 0:
            # Update point coordinates
            self.get_coordinates(data.shape)
            out['receiver_xyz'] = self.point_coords
            # Update neighbors indices
            self.find_neighbors_indices(data.shape)

        # Set output defaults
        leak_detected_ts = 0
        detection_time_ts = np.inf

        # Get data to compare to a threshold
        comp_data = get_multiple_measurements(
            self.combine_indices(), data=data, baseline=baseline,
            criteria=self.criteria)

        # Compare measurement data against threshold
        if np.all(comp_data >= actual_p['threshold']):
            # These observations keep track of history: ts for time series
            leak_detected_ts = 1
            detection_time_ts = time_point

        out['leak_detected_ts'] = leak_detected_ts
        out['detection_time_ts'] = detection_time_ts

        if leak_detected_ts:  # leak detected at the current time point
            if self.accumulators['leak_detected'].sim == 1:  # leak was detected previously
                # Detection time would be time in the past obtained from accumulator
                out['detection_time'] = self.accumulators['detection_time'].sim
            else:  # leak was not detected previously (attribute sim is 0)
                # Detection time would be current time point
                out['detection_time'] = time_point
                # Update accumulators
                self.accumulators['detection_time'].sim = time_point
                self.accumulators['leak_detected'].sim = 1
            # Leak is considered to be detected at the current time point
            out['leak_detected'] = 1
        else:  # leak is not detected at the current time point
            if self.accumulators['leak_detected'].sim:  # leak was detected previously
                # If leak was detected previously but not now it means
                # the measured quantity is no longer over the threshold
                # It's a rare situation but still possible
                # Detection time would be time in the past obtained from accumulator
                out['detection_time'] = self.accumulators['detection_time'].sim
                # Leak was detected in the past so it's still considered
                # detected even if it's not detected at the current time point
                out['leak_detected'] = 1
            else:  # leak was not detected previously and at the current time point
                # The outputs assume default initial values
                out['detection_time'] = np.inf
                out['leak_detected'] = 0
                self.accumulators['detection_time'].sim = np.inf
                self.accumulators['leak_detected'].sim = 0

        return out


def test_gravity_measurements():
    import matplotlib.pyplot as plt
    from openiam import SystemModel
    from ramp.components.base import DataContainer
    from ramp.utilities.data_readers import default_h5_file_reader
    from ramp.configuration.configuration import BaseConfiguration

    # Define keyword arguments of the system model
    time_points = np.array([5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200])
    time_array = 365.25*time_points
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'gravity'
    data_directory = os.path.join('..', '..', '..', '..', 'data', 'user', 'pressure')
    output_directory = os.path.join('..', '..', '..', '..', 'examples', 'user',
                                    'output', 'test_gravity_measurements')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_h5_file_reader
    data_reader_kwargs = {'obs_name': obs_name}

    num_time_points = len(time_points)
    # scenarios = list(range(100, 116))
    scenarios = [1, 7, 10, 5, 9]
    family = 'gravity'
    data_setup = {}
    for scen in scenarios:
        data_setup[scen] = {'folder': 'sim0001_0100'}
        for t_ind in range(1, num_time_points+1):
            data_setup[scen][f't{t_ind}'] = f'sim{scen:04}.h5'
    baseline = False

    # Create configuration for GravityMeasurements component
    # Number of points in x- and y-directions
    nx = 41
    ny = 21
    num_p = nx*ny
    x = np.linspace(4000.0, 8000.0, nx)
    y = np.linspace(1500.0, 3500.0, ny)

    xx, yy = np.meshgrid(x, y, indexing='ij')
    xyz_coords = np.zeros((num_p, 3))
    xyz_coords[:, 0] = xx.reshape((num_p, ))
    xyz_coords[:, 1] = yy.reshape((num_p, ))

    gravity_data_config = BaseConfiguration(
        sources=None, receivers=xyz_coords, name='Gravity config')

    # Create system model
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # Add data container
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
        dc.add_obs_to_be_linked(nm, obs_type='grid')

    # dim_indices = (11, 13) # leak is detected for some time points
    dim_indices = (10, 11) # leak detected for some time points
    # dim_indices = (35, 11) # leak is not detected at any of the time points
    # Setup the rest of parameters
    gravity_anomaly_threshold = 15 # uGal
    criteria = 2 # compare absolute value of data to a threshold
    num_neighbors = 4 # number of neighbor points at which anomaly should exceed threshold

    # Add gravity measurements component
    meas = sm.add_component_model_object(
        GravityMeasurements(name='meas', parent=sm, config=gravity_data_config,
                           time_points=time_points, dim_indices=dim_indices,
                           num_neighbors=num_neighbors, criteria=criteria))

    meas.add_par('threshold', value=gravity_anomaly_threshold, vary=False)
    # Add keyword arguments linked to the data container outputs
    meas.add_kwarg_linked_to_obs('data', dc.linkobs['gravity'], obs_type='grid')
    # Add observations
    for nm in ['leak_detected', 'detection_time']:
        meas.add_obs(nm)        # covers the whole simulation period
        meas.add_obs(nm + '_ts')  # covers how values change in time
    meas.add_grid_obs('receiver_xyz', constr_type='matrix', output_dir=output_directory)

    # Run system model
    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()

    # Get saved gravity anomaly data from files
    gravity_data = sm.collect_gridded_observations_as_time_series(
        dc, 'gravity', output_directory, rlzn_number=0)  # (12, 41, 21)

    print('Gravity data shape:', gravity_data.shape)

    # Collect data point coordinate
    coords = sm.collect_gridded_observations_as_time_series(
        meas, 'receiver_xyz', output_directory, indices=[0], rlzn_number=0)[0]
    print('Placement of receptor:', coords)

    # Export scalar observations
    evaluations = {}
    for nm in ['leak_detected', 'detection_time']:
        evaluations[nm] = sm.collect_observations_as_time_series(meas, nm)
        evaluations[nm + '_ts'] = sm.collect_observations_as_time_series(meas, nm + '_ts')

    dx = 5
    dy = 3
    cmap = 'jet'
    t_ind = 5
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax_im = ax.pcolormesh(x, y, gravity_data[t_ind, :, :].T, cmap=cmap)
    ax.set_title(f'Scenario {index}: gravity anomaly at t={time_points[t_ind]} years')
    sensor_color = 'grey'
    ax.plot(coords[0], coords[1], 'o', color=sensor_color)
    ax.annotate('Sensor', (coords[0], coords[1]),
                xytext=(coords[0]+30, coords[1]+30), color=sensor_color)
    plt.colorbar(ax_im, label='G, [uGal]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_ylabel('y, [m]')

    # Plot leak_detected metric
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    for ind, nm in enumerate(['leak_detected_ts', 'leak_detected']):
        colors = np.where(evaluations[nm]==1, 'blue', 'red')
        axs[ind].scatter(time_points, evaluations[nm], s=50, c=colors)
        try:
            mark_time_ind = np.where(evaluations[nm]==1)[0][0]
            axs[ind].plot(2*[time_points[mark_time_ind]], [-0.05, 1.1],
                          '--', color='gray')
        except IndexError:
            pass
        axs[ind].set_xlabel('Time, [years]')
    axs[0].set_yticks([0, 1], labels=[0, 1])
    axs[0].set_ylabel('Leak detected (History)')
    axs[1].set_ylabel('Leak detected (Propagated)')
    fig.suptitle(f'Scenario {index}')
    fig.tight_layout()

    # Print whether leak was detected and detection time if applicable
    final_obs_ind = len(time_points)-1
    leak_detected = sm.obs[f'meas.leak_detected_{final_obs_ind}'].sim
    print('Is leak detected?', bool(leak_detected))
    if leak_detected:
        detection_time = sm.obs[f'meas.detection_time_{final_obs_ind}'].sim
        print('Leak is detected for the first time at t = {} days ({} years)'.format(
            detection_time, detection_time/365.25))


if __name__ == "__main__":

    test_gravity_measurements()
