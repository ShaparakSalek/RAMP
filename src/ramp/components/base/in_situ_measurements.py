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


def default_map_dim_index_2_point_index(multi_index, dims):
    """
    Convert a tuple of index arrays into flat index

    Parameters
    ----------
    multi_index : tuple of integers
        A tuple of integers, one integer for each dimension.

    dims : tuple of integers
        The shape of array into which the indices from multi_index apply.

    Returns an index of element in a flat array.
    -------
    None.

    """
    # The elements are assumed to be read in row-major style (C - by default)
    return np.ravel_multi_index(multi_index, dims)

def default_map_point_index_2_dim_index(flat_index, dims):
    """
    Convert a flat index into a tuple of multidimensional indices

    Parameters
    ----------
    flat_index : integer, or array-like of integers
        Index of element within an array.

    dims : tuple of integers
        The shape of array into which the indices for returned multi_index apply.

    Returns a tuple of indices.
    -------
    None.

    """
    return np.unravel_index(flat_index, shape)

def get_measurements(dim_indices, data=None, baseline=None, criteria=1):

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


class InSituMeasurements(MonitoringTechnology):

    def __init__(self, name, parent, config, time_points, dim_indices=None,
                 map_dim_index_2_point_index=None, criteria=1):
        """
        The InSituMeasurements class is designed to model monitoring well
        simulation which can track pressure, TDS, pH, etc at a given location
        and compare them to a threshold.

        Parameters
        ----------
        name : str
            Name of InSituMeasurements class instance under which it will
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
        criteria : int
            Flag variable indicating how the data is compared to a threshold
            Value of 1 means data is compared as it is or difference (data-baseline)
            is compared to a threshold if baseline data is provided
            Value of 2 means an absolute value of data is compared or an absolute
            value of the difference (data-baseline) is compared to a threshold
            if baseline data is provided

        Returns
        -------
        Instance of InSituMeasurements class.

        """
        # Setup keyword arguments of the 'model' method provided by the system model
        model_kwargs = {'time_point': 0.0}  # default value of 0 days

        super().__init__(name, parent, model_kwargs=model_kwargs)

        # Add type attribute
        self.class_type = 'InSituMeasurements'

        # Setup additional attributes
        self.configuration = config
        if map_dim_index_2_point_index is None:
            self.index_map = default_map_dim_index_2_point_index
        else:
            self.index_map = map_dim_index_2_point_index
        self.criteria = criteria

        # Setup attribute data_indices
        self.get_indices(dim_indices=dim_indices)

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

    def process_data(self, p, time_point=None, data=None, baseline=None,
                     dim_indices=None):
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
            # Get the closest neighbors coordinates

        # Set output defaults
        leak_detected_ts = 0
        detection_time_ts = np.inf

        # Get data to compare to a threshold
        comp_data = get_measurements(self.data_indices, data=data,
                                     baseline=baseline, criteria=self.criteria)

        # Compare measurement data against threshold
        if comp_data >= actual_p['threshold']:
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


def test_in_situ_measurements():
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
    obs_name = ['pressure', 'saturation']
    data_directory = os.path.join('..', '..', '..', '..', 'data', 'user', 'pressure')
    output_directory = os.path.join('..', '..', '..', '..', 'examples', 'user',
                                    'output', 'test_in_situ_measurements')
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

    # Create configuration for InSituMeasurements component
    # Number of points in x-, y-, and z-directions
    nx = 40
    ny = 20
    nz = 32
    num_p = nx*ny*nz
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
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    xyz_coords = np.zeros((num_p, 3))
    xyz_coords[:, 0] = xx.reshape((num_p, ))
    xyz_coords[:, 1] = yy.reshape((num_p, ))
    xyz_coords[:, 2] = zz.reshape((num_p, ))

    pressure_data_config = BaseConfiguration(
        sources=None, receivers=xyz_coords, name='Pressure config')

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
        dc.add_obs_to_be_linked(nm, obs_type='grid')

    # dim_indices = (4, 10, 25) # leak is detected for some points
    dim_indices = (6, 10, 25) # leak detected for some points
    # dim_indices = (38, 10, 25) # leak is not detected
    meas = sm.add_component_model_object(
        InSituMeasurements(name='meas', parent=sm, config=pressure_data_config,
                           time_points=time_points, dim_indices=dim_indices,
                           criteria=1))
    delta_pressure_threshold = 15
    meas.add_par('threshold', value=delta_pressure_threshold, vary=False)
    # Add keyword arguments linked to the data container outputs
    meas.add_kwarg_linked_to_obs('data', dc.linkobs['pressure'], obs_type='grid')
    # Add observations
    for nm in ['leak_detected', 'detection_time']:
        meas.add_obs(nm)        # covers the whole simulation period
        meas.add_obs(nm + '_ts')  # covers how values change in time
    meas.add_grid_obs('receiver_xyz', constr_type='matrix', output_dir=output_directory)

    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()

    # Get saved data from files
    # delta pressure, saturation
    dpressure_data = sm.collect_gridded_observations_as_time_series(
        dc, 'pressure', output_directory, rlzn_number=0)  # (12, 40, 20, 32)
    saturation_data = sm.collect_gridded_observations_as_time_series(
        dc, 'saturation', output_directory, rlzn_number=0)

    # Collect data point coordinate
    coords = sm.collect_gridded_observations_as_time_series(
        meas, 'receiver_xyz', output_directory, indices=[0], rlzn_number=0)[0]
    print('Placement of receptor:', coords)

    # Export scalar observations
    evaluations = {}
    for nm in ['leak_detected', 'detection_time']:
        evaluations[nm] = sm.collect_observations_as_time_series(meas, nm)
        evaluations[nm + '_ts'] = sm.collect_observations_as_time_series(meas, nm + '_ts')

    # Plot delta pressure and saturation z-slice plots
    t_ind = 3
    z_ind = 25
    dx = 5
    dy = 4
    dz = 4
    cmap = 'hot_r'
    fig = plt.figure(figsize=(16, 6))
    # first subplot
    ax = fig.add_subplot(121)
    ax_im = ax.pcolormesh(x, y, dpressure_data[t_ind, :, :, z_ind].T, cmap=cmap)
    ax.plot(coords[0], coords[1], 'og')
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
    ax.plot(coords[0], coords[1], 'og')
    ax.set_title('CO2 saturation at t={} years, depth={} m'.format(
        time_points[t_ind], z[z_ind]))
    plt.colorbar(ax_im, label='S, [-]')
    ax.set_xticks(x[0:-1:dx], labels=x[0:-1:dx])
    ax.set_xlabel('x, [m]')
    ax.set_yticks(y[0:-1:dy], labels=y[0:-1:dy])
    ax.set_ylabel('y, [m]')
    fig.tight_layout()

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

    # Print whether leak was detected and detection time if applicable
    final_obs_ind = len(time_points)-1
    leak_detected = sm.obs['meas.leak_detected_{}'.format(final_obs_ind)].sim
    print('Is leak detected?', bool(leak_detected))
    if leak_detected:
        detection_time = sm.obs['meas.detection_time_{}'.format(final_obs_ind)].sim
        print('Leak is detected for the first time at t = {} days ({} years)'.format(
            detection_time, detection_time/365.25))


if __name__ == "__main__":

    test_in_situ_measurements()
