# -*- coding: utf-8 -*-
"""
The reservoir data interpolator estimates pressure and |CO2| saturation
at the location of interest using precomputed lookup tables.
The calculations are based on linear interpolation for irregular grids.
"""

import csv
import logging
import sys
import os
import h5py
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openiam import SystemModel
    from openiam.iam_gridded_observation import interp_weights
    from openiam.iam_gridded_observation import interpolate
except ImportError as err:
    print('Unable to load IAM class module: {}'.format(err))


def check_file_format(data_file, data_type='data'):
    """
    Check whether data comes from *.csv or *.hdf5 file.

    :param data_file: name of *.csv or *.hdf5 file to read observation (e.g.,
        pressure and |CO2| saturation) data from
    :type filename: string
    """
    if data_file.lower().endswith('.csv'):
        hdf5_data_file = False
    elif data_file.lower().endswith('.hdf5') or data_file.lower().endswith('.h5'):
        hdf5_data_file = True
    else:
        err_msg = ''.join([
            'Format of the provided {} file is not valid: it is neither .csv ',
            'nor .hdf5. Please check your input.']).format(data_type)
        logging.error(err_msg)
        raise Exception(err_msg)

    return hdf5_data_file, data_file


def read_time_points(is_hdf5_file, header_file_dir, time_file):
    """
    Load time points from user-provided data files.
    """
    time_file_full_path = os.path.join(header_file_dir, time_file)
    # Obtain time points data
    if not is_hdf5_file: # time file has .csv extension
        time_points = np.loadtxt(time_file_full_path, delimiter=",", dtype='f8')
    else:  # time file has .hdf5 extension
        # Read provided hdf5 file containing key t for time points data
        with h5py.File(time_file_full_path, 'r') as hf:
            time_points = hf['t'][()]

    # Determine number of time points in the data
    num_data_time_points = len(time_points)

    return num_data_time_points, time_points


def read_data_headers(is_hdf5_file, header_file_dir, data_file):
    """
    Determine type of observation (e.g., x, y, z, pressure, saturation, etc.)
    in the provided data file.
    """
    data_file_full_path = os.path.join(header_file_dir, data_file)
    # Obtain headers of the data file with observations
    if not is_hdf5_file:  # csv data file format
        data = np.genfromtxt(data_file_full_path, names=True, delimiter=',', max_rows=1)
        data_headers = list(data.dtype.names)
        msg = 'Data file headers: {}'.format(data_headers)
        logging.debug(msg)

    else:  # hdf5 data file format
        with h5py.File(data_file_full_path, 'r') as hf:
            data_headers = list(hf.keys())

    return data_headers


class AnimatedScatterPlot(object):
    def __init__(self, interp_obj, obs_nm='pressure'):
        """
        Constructor method of AnimatedScatterPlot class.

        Create object that would animate the time evolution of the indicated data
        associated with the supplied interpolator.

        :param interp_obj: the ReservoirDataInterpolator object whose data
            is needed to be plotted
        :type interp_obj: ReservoirDataInterpolator object

        :param obs_nm: name of observational data to be shown as animation;
            possible values: name of the observation varying (at least) in time
        :type obs_nm: str

        :returns: AnimatedScatterPlot class object
        """
        self.num_time_points = len(interp_obj.time_points)
        self.num_points = len(interp_obj.points[:, 0])

        self.time_points = interp_obj.time_points
        self.points = interp_obj.points
        self.interp_obj = interp_obj
        self.obs_nm = obs_nm

        self.data = interp_obj.data
        self.colorbar_title = interp_obj.default_units

        self.plot_title = interp_obj.title_names_orig

        self.stream = list(range(0, self.num_time_points))

        # Setup the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Set the x and y limits
        self.ax.axis([np.min(self.points[:, 0]), np.max(self.points[:, 0]),
                      np.min(self.points[:, 1]), np.max(self.points[:, 1])])
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal', 'datalim')

        # Determine min and max values of observations
        data_min = min([np.min(self.data[self.obs_nm][key]) for key in self.data[self.obs_nm]])
        data_max = max([np.max(self.data[self.obs_nm][key]) for key in self.data[self.obs_nm]])

        # For the colors range we take the min and max of data for all time points
        self.scat = self.ax.scatter(x=self.points[:, 0], y=self.points[:, 1],
                                    c=np.linspace(data_min, data_max, self.num_points),
                                    animated=True)
        # Add colorbar
        plt.colorbar(self.scat, ax=self.ax, label=self.colorbar_title[self.obs_nm])

        # Plot data at the first available time point which have index 1
        self.scat.set_array(self.data[self.obs_nm][1])

        # Add title of the scatter plot as text
        self.title = self.ax.text(.5, 1.05, self.plot_title[self.obs_nm],
                                  transform=self.ax.transAxes,
                                  va='center', ha='center', size='large')

        # Setup FuncAnimation
        self.ani = animation.FuncAnimation(
            self.fig, self.update, self.stream, init_func=self.setup,
            blit=True, interval=250, save_count=self.num_time_points)

    def setup(self):
        """ Return initial state of the plot."""
        return self.scat,

    def update(self, ind):
        """ Update the scatter plot."""

        # Set colors of the scatter plot
        self.scat.set_array(self.data[self.obs_nm][ind+1])

        # Return the updated artists for FuncAnimation to draw.
        return self.scat,


class ReservoirDataInterpolator(object):
    """ NRAP-Open-IAM ReservoirDataInterpolator class. """
    def __init__(self, name, parent, header_file_dir, time_file, data_file, index,
                 signature, default_values=None, interp_2d=True,
                 triangulation=None, build_on_the_fly=False):
        """
        Constructor method of ReservoirDataInterpolator

        :param name: name of interpolator
        :type name: str

        :param parent: name of the system model interpolator belongs to
        :type parent: SystemModel object

        :param header_file_dir: location (directory) of the reservoir
            simulation data that will be used for interpolation
        :type header_file_dir: string

        :param time_file: name of *.csv file to read information about time
            points (in years) at which observations (e.g., pressure and |CO2|
            saturation) were calculated
        :type filename: string

        :param data_file: name of *.csv or *.hdf5 file to read observation (e.g.,
            pressure and |CO2| saturation) data from
        :type filename: string

        :param index: index enumerating the lookup table within the data set
        :type index: int

        :param signature: dictionary of parameters and their corresponding values
            associated with the given simulation data file
        :type signature: dict()

        :param default_values: dictionary of default values of observations
            which are not provided in the data_file. By default,
            it is assumed that data provided in the data_file is enough,
            thus, by default, the parameter is None.
            The values provided are assumed to be constant for all time steps
            and all spatial points.
        :type default_values: dict()

        :param interp_2d: flag variable indicating whether the data provided
            in the lookup tables should be treated as 2d (or 3d) for the purpose
            of interpolation. By default, the value is True for
            the data to be interpolated as 2d. False value means that
            the data is 3d and 3d interpolation will be used.
        :type interp_2d: boolean

        :param triangulation: Delaunay tesselation in 2- or 3-d. By default,
            triangulation is None which means that for the given interpolator
            it has to be calculated based on the data available through
            the provided data_file.
        :type triangulation: scipy.spatial.Delaunay

        :param build_on_the_fly: flag variable indicating whether the data
            from the lookup table corresponding to the given interpolators
            will be read only if needed, e.g., at the first
            call of interpolator. By default, interpolator is created
            at the initialization of the corresponding ReservoirDataInterpolator
            class object (value=False).
        :type build_on_the_fly: boolean

        :returns: ReservoirDataInterpolator class object
        """
        # Setup attributes of the ReservoirDataInterpolator class object
        self.name = name             # name of interpolator
        self.parent = parent         # a system model interpolator belongs to
        self.header_file_dir = header_file_dir  # path to the directory with simulation data files
        # Name of file with time points data
        self.hdf5_time_file, self.time_file = check_file_format(
            time_file, data_type='time')

        # data_file is a name of file with pressure and saturation data
        self.hdf5_data_format, self.data_file = check_file_format(
            data_file, data_type='data')

        # Check type of the provided argument index
        if isinstance(index, int):
            self.index = index  # index of the lookup table within the data set
        else:
            err_msg = ''.join([
                'Argument "index" of interpolator {} has a wrong type. ',
                'Integer type should be used.']).format(name)
            logging.error(err_msg)
            raise TypeError(err_msg)

        # Signature can be an empty dictionary
        self.signature = signature   # dictionary of parameters with values
        # defining this interpolator

        self.default_values = default_values  # values (if any) specified
        # for the interpolator observation

        # Setup default units of measurements for possible observations
        # (mainly for plotting purposes).
        # The dictionary can be updated later when we figure out whether any
        # additional observations are possible/needed; or dictionary can be updated
        # in script
        self.default_units = {'pressure': 'Pa',
                              'CO2saturation': '[-]',
                              'salinity': '',
                              'temperature': r'$^\circ$C',
                              'area': r'm$^2$'}

        # Setup possible titles for different observations (for use in show_data method)
        self.title_names_orig = {'pressure': 'Pressure',
                                 'CO2saturation': r'CO$_2$ saturation',
                                 'salinity': 'Salinity',
                                 'temperature': 'Temperature',
                                 'area': 'Cell area'}
        self.title_names = {}
        for key in self.title_names_orig:
            self.title_names[key] = self.title_names_orig[key]+' at time t = {:.2f} years'

        # Save current triangulation
        self.triangulation = triangulation

        # By default the data is treated as 2d so interpolation is going to be 2d
        self.interp_2d = interp_2d

        # Initialize a flag showing whether the data were read and interpolator was created
        self.created = False

        # Create models for each output type for each time step
        if not build_on_the_fly:
            self.create_data_interpolators(self.triangulation)

        # Log message about creating the interpolator
        msg = 'ReservoirDataInterpolator created with name {name}'.format(name=name)
        logging.debug(msg)

    def process_csv_data_file(self):
        """
        Analyze the data provided in the csv data file and populate the corresponding
        attributes of the instance.
        """
        data = pd.read_csv(os.path.join(self.header_file_dir, self.data_file))

        # Initialize list of constant in space observations
        constant_obs = {}

        # Check whether z coordinate is present
        if self.data_headers[2] == 'z':
            all_names = self.data_headers[3:]
            delta_ind = 3
            constant_obs['depth'] = 2  # index of data in the csv file
        else:
            all_names = self.data_headers[2:]
            delta_ind = 2

        # Loop over all observation names in the header except the first two or three
        # (x, y and z if present)
        for ind, nm in enumerate(all_names):
            # Find index of the last occurence of underscore in the column name
            underscore_ind = nm.rfind('_')

            if underscore_ind == -1:
                # If there is no underscore in the column name
                # then the observation is constant for all time points
                constant_obs[nm] = ind+delta_ind   # save index of the column
            else:
                # If there is an underscore in the name of the observation
                # find base name of the observation and time index
                obs_nm = nm[0:underscore_ind]        # name of observation

                try:
                    # Try is added to check whether symbols after last underscore represent number
                    # time index of observation starts with 1
                    time_ind = int(nm[(underscore_ind+1):])-1
                    # Check whether the obs_nm is already added to the data dictionary
                    if not obs_nm in self.data:
                        # Initialize dictionary corresponding to obs_nm
                        self.data[obs_nm] = {}
                    self.data[obs_nm][time_ind+1] = data.iloc[:, ind+delta_ind].values
                except ValueError:
                    # If symbols after underscore are not numbers (e.g. cell_area)
                    # assume that the data in the column is time independent
                    constant_obs[nm] = ind+delta_ind  # save index of the column

        for nm, ind in constant_obs.items():
            # nm observation is both specified as time dependent and independent
            if nm in self.data:
                msg = ''.join([
                    'Observation "{}" is specified as time ',
                    'independent in the column {} and as time ',
                    'varying in another column(s) in the file {}. The column ',
                    'corresponding to the time independent data ',
                    'will be used for calculations.']).format(nm, ind+1, self.data_file)
                logging.warning(msg)
            # -1 corresponds to the data constant in time but not in space
            self.data[nm] = {-1: data.iloc[:, ind].values}

        if self.default_values is not None:
            for nm, val in self.default_values.items():
                if nm in self.data:
                    msg = ''.join([
                        'Default value is specified for observation "{}" as ',
                        'argument of the ReservoirDataInterpolator ',
                        'constructor method. Observation is also ',
                        'defined in the lookup data file "{}". Value provided ',
                        'to the constructor method will be used ',
                        'for calculations.']).format(nm, self.data_file)
                    logging.warning(msg)
                self.data[nm] = {-2: val}   # -2 corresponds to data constant both in space and time

        # Setup x, y (and z if needed) points
        if self.interp_2d:
            self.points = data.iloc[:, 0:2].values
        else:
            self.points = data.iloc[:, 0:3].values

        self.num_xy_points = len(self.points)

    def process_hdf5_data_file(self):
        """
        Analyze the data provided in the hdf5 data file and populate the corresponding
        attributes of the class instance.
        """
        # Initialize list of constant in space observations
        constant_obs = {}
        if 'z' in self.data_headers:
            constant_obs['depth'] = 'z'  # saving key where the data is kept

        coord_keys = ['x', 'y', 'z', 'ijk', 't']
        all_names = [nm for nm in self.data_headers if nm not in coord_keys]

        # Loop over all observation names in the header except coordinates
        # (x, y and z if present)
        for ind, nm in enumerate(all_names):
            # Find index of the last occurence of underscore in the observation name
            underscore_ind = nm.rfind('_')

            if underscore_ind == -1:
                # If there is no underscore in the column name
                # then the observation is constant for all time points
                constant_obs[nm] = nm   # save name of the data
            else:
                # If there is an underscore in the name of the observation
                # find base name of the observation and time index
                obs_nm = nm[0:underscore_ind]        # name of observation

                try:
                    # Try is added to check whether symbols after last underscore represent number
                    # time index of observation starts with 1
                    time_ind = int(nm[(underscore_ind+1):])
                    # Check whether the obs_nm is already added to the data dictionary
                    if not obs_nm in self.data:
                        # Initialize dictionary corresponding to obs_nm
                        self.data[obs_nm] = {}
                    self.data[obs_nm][time_ind] = nm
                except ValueError:
                    # If symbols after underscore are not numbers (e.g. cell_area)
                    # assume that the data is time independent
                    constant_obs[nm] = nm  # save name of observation

        for nm, ind in constant_obs.items():
            # nm observation is both specified as time dependent and independent
            if nm in self.data:
                msg = ''.join([
                    'Observation "{}" is specified as time ',
                    'independent in the entry {} and as time ',
                    'varying in another entry {} in the file {}. The values ',
                    'corresponding to the time independent data ',
                    'will be used for calculations.']).format(
                        nm, nm, list(self.data[nm].keys())[0], self.data_file)
                logging.warning(msg)
            # -1 corresponds to the data constant in time but not in space
            if nm != 'depth':
                self.data[nm] = {-1: nm}
            else:
                self.data[nm] = {-1: constant_obs[nm]}

        if self.default_values is not None:
            for nm, val in self.default_values.items():
                if nm in self.data:
                    msg = ''.join([
                        'Default value is specified for observation "{}" as ',
                        'argument of the ReservoirDataInterpolator ',
                        'constructor method. Observation is also ',
                        'defined in the lookup data file "{}". Value provided ',
                        'to the constructor method will be used ',
                        'for calculations.']).format(nm, self.data_file)
                    logging.warning(msg)

                # -2 corresponds to data constant both in space and time
                self.data[nm] = {-2: val}

        data_file = os.path.join(self.header_file_dir, self.data_file)
        # Setup x, y (and z if needed) points
        with h5py.File(data_file, 'r') as hf:
            x = hf['x'][()]
            y = hf['y'][()]
            self.num_xy_points = len(x)

            if self.interp_2d:
                self.points = np.zeros((self.num_xy_points, 2))
            else:
                self.points = np.zeros((self.num_xy_points, 3))
                self.points[:, 2] = hf['z'][()]

            self.points[:, 0] = x
            self.points[:, 1] = y

    def create_data_interpolators(self, triangulation=None):
        """
        Setup attributes storing applicable observation data.

        Create data attribute of the class object for use in the model method of the class.

        :param triangulation: Delaunay tesselation in 2-d. By default,
            triangulation is None which means that for the given interpolator
            it has to be calculated based on the data available through
            the provided data_file.
        :type triangulation: scipy.spatial.Delaunay
        """
        # Read time points data
        self.num_data_time_points, self.time_points = read_time_points(
            self.hdf5_time_file, self.header_file_dir, self.time_file)

        # Read data headers information
        self.data_headers = read_data_headers(self.hdf5_data_format,
                                              self.header_file_dir,
                                              self.data_file)

        # Setup data dictionary
        self.data = {}

        if not self.hdf5_data_format:     # csv data file format
            self.process_csv_data_file()
        else:                             # hdf5 data file format
            self.process_hdf5_data_file()

        # Perform different checks of the data
        for nm, data_item in self.data.items():
            if (-1 not in data_item) and (-2 not in data_item):
                # Check that the number of time points coincides with the number of columns
                # provided for an observation varying both in space and time
                if len(data_item) != self.num_data_time_points:
                    err_msg = ''.join([
                        'The number of columns/entries ({}) provided for ',
                        'observation "{}" in the file "{}" is not equal to ',
                        'the number of time points ({}) in the file "{}", ',
                        'lookup table for interpolator {}.']).format(
                            len(data_item), nm, self.data_file,
                            self.num_data_time_points, self.time_file, self.name)
                    logging.error(err_msg)
                    raise ValueError(err_msg)
                # Check that the indexing of columns with time dependent observations
                # start with 1 and not 0
                if 0 in data_item:
                    err_msg = ''.join([
                        'Enumeration of time points for observation {} ',
                        'in the file "{}" should start with 1 not with 0.']).format(
                            nm, self.data_file)
                    logging.error(err_msg)
                    raise ValueError(err_msg)
                # Check that the indexing of the columns is sequential, i.e.
                # all the supposed indices are present
                for col_ind in range(1, self.num_data_time_points+1):
                    if col_ind not in data_item:
                        err_msg = ''.join([
                            'Column (or entry) "{}_{}" corresponding to time point {} ',
                            '(index {}) is not present in the file "{}".']).format(
                                nm, col_ind, self.time_points[col_ind-1],
                                col_ind, self.data_file)
                        logging.error(err_msg)
                        raise KeyError(err_msg)

        # Check that pressure and CO2 saturation data is present
        for nm in ['pressure', 'CO2saturation']:
            if nm not in self.data:
                err_msg = "".join([
                    'No columns/entries for observation "{}" are provided ',
                    'in the file "{}".']).format(nm, self.data_file)
                logging.error(err_msg)
                raise KeyError(err_msg)

        # Determine triangulation for the data points
        if triangulation is None:
            self.triangulation = Delaunay(self.points)

        # Record that the data is read and the interpolator is created
        self.created = True

    def data_units(self, obs_nm):
        """
        Return measurement units of the provided observations.

        :param obs_nm: name of the observation for which units of measurements
            are to be provided.
        :type obs_nm: str

        :returns: units of measurements as str. For observation not
            in the list of possible known names, the string 'Unknown' is returned.
        """
        if obs_nm in self.default_units:
            return self.default_units[obs_nm]

        return 'Unknown'

    def show_data(self, time=None, obs_to_show='all'):
        """
        Show data linked to the interpolator dealing with 2d data.

        :param time: time point (in years) at which the data need to be shown. If time
            point does not coincide with any of the time points
            associated with the interpolator only the extent of the domain
            is shown. If no time point is provided, i.e., time is None
            (by default), an animation is created.
        :type time: float

        :param obs_to_show: list of names of observations to be shown; by default,
            all data linked to interpolator is shown.
        :type kwargs: list()
        """
        if not self.interp_2d:
            raise ValueError(''.join([
                'Interpolator is linked to 3d data which ',
                'cannot be vizualized by method show_data.']))

        # Check whether the interpolator was created
        if not self.created:
            self.create_data_interpolators(self.triangulation)

        # Create a list of observation names to show
        if obs_to_show == 'all':
            obs_nms = list(self.data.keys())
        else:
            obs_nms = obs_to_show

        # TODO For now method only works for csv files; it needs to be updated
        # to visualize data from hdf5 files
        # Loop over all observations to be shown on plots
        for nm in obs_nms:
            unchanging_plot = True
            if -2 in self.data[nm]:     # obs is constant in space and time
                data_colors = self.data[nm][-2]*np.ones(self.num_xy_points)
            elif -1 in self.data[nm]:   # obs is constant in time but not space
                data_colors = self.data[nm][-1]
            else:
                if time is None:
                    if not hasattr(self, 'sca_plt'):
                        self.sca_plt = {}
                    self.sca_plt[nm] = AnimatedScatterPlot(self, obs_nm=nm)
                    unchanging_plot = False
                else:
                    # Find index of time array point
                    ind = np.where(self.time_points == time)[0]
                    if len(ind) == 1:
                        data_colors = self.data[nm][ind[0]+1][:]
                    else:
                        # Show extent of the data
                        fig = plt.figure(figsize=(8, 8))
                        ax = fig.add_subplot(111)
                        ax.plot(self.points[:, 0], self.points[:, 1], 'o')
                        ax.set_title('Boundaries of the domain')
                        ax.set_xlabel('x [m]')
                        ax.set_ylabel('y [m]')
                        ax.set_aspect('equal', 'datalim')
                        plt.show()
                        msg = 'Data is not available at time t = {} years'.format(time)
                        logging.warning(msg)
                        return

            # Plot observation
            if unchanging_plot:
                fig = plt.figure(figsize=(8, 8))
                ax = fig.add_subplot(111)
                im = ax.scatter(self.points[:, 0], self.points[:, 1], c=data_colors)
                ax.set_xlabel('x [m]')
                ax.set_ylabel('y [m]')
                ax.set_aspect('equal', 'datalim')
                fig.colorbar(im, label=self.data_units(nm), ax=ax)
                t = 0 if time is None else time
                ax.set_title(self.title_names[nm].format(t))
                plt.show()

    def calculate_csv_based_output(self, dtime, vtx=None, wts=None):
        """
        Calculate outputs using approach based on csv files.
        """
        # Create dictionary of output
        out = dict()
        # Check whether a single point is requested
        if vtx is not None and wts is not None:
            # Check whether weights are reasonable; if they are not, then it means
            # that the point at which we need to interpolate is outside the data range
            if np.any(abs(wts) > 1.0+1.0e-8):
                msg = ''.join(['Input array of weights {} has invalid ',
                               'values: some or all values are ',
                               'greater than 1.']).format(wts)
                logging.warning(msg)
            # Cycle over all observations
            for nm, data_item in self.data.items():
                if -2 in data_item:     # obs is constant in space and time
                    out[nm] = np.array([data_item[-2]])
                elif -1 in data_item:   # obs is constant in time but not space
                    out[nm] = interpolate(data_item[-1], vtx, wts)
                else:                       # obs varies in space and time
                    for ind in range(self.num_data_time_points):
                        # If time point coincides with one of the data time points
                        if dtime == self.time_points[ind]:
                            out[nm] = interpolate(data_item[ind+1], vtx, wts)
                        # If time point is between the data time points
                        elif self.time_points[ind] < dtime < self.time_points[ind+1]:
                            out1 = interpolate(data_item[ind+1], vtx, wts)
                            out2 = interpolate(data_item[ind+2], vtx, wts)
                            delta_t = self.time_points[ind+1]-self.time_points[ind]

                            # Interpolate between two data points
                            out[nm] = (out2-out1)/(delta_t)*(
                                dtime-self.time_points[ind])+out1
        else:  # return data corresponding to all points in the data file
            for nm, data_item in self.data.items():
                if -2 in data_item:     # obs is constant in space and time
                    out[nm] = data_item[-2]*np.ones(self.num_xy_points)
                elif -1 in data_item:   # obs is constant in time but not space
                    out[nm] = data_item[-1]
                else:
                    for ind in range(self.num_data_time_points):
                        # If time point coincides with one of the data time points
                        if dtime == self.time_points[ind]:
                            out[nm] = data_item[ind+1]
                        elif self.time_points[ind] < dtime < self.time_points[ind+1]:
                            out1 = data_item[ind+1]
                            out2 = data_item[ind+2]
                            delta_t = self.time_points[ind+1]-self.time_points[ind]
                            # Interpolate between two data points
                            out[nm] = (out2-out1)/(delta_t)*(
                                dtime-self.time_points[ind])+out1
        return out

    def calculate_hdf5_based_output(self, dtime, vtx=None, wts=None):
        """
        Calculate outputs using approach based on csv files.
        """
        # Create dictionary of output
        out = dict()
        # Get path to data file
        data_file = os.path.join(self.header_file_dir, self.data_file)
        # Check whether a single point is requested
        if vtx is not None and wts is not None:
            # Check whether weights are reasonable; if they are not, then it means
            # that the point at which we need to interpolate is outside the data range
            if np.any(abs(wts) > 1.0+1.0e-8):
                msg = ''.join(['Input array of weights {} has invalid ',
                               'values: some or all values are ',
                               'greater than 1.']).format(wts)
                logging.warning(msg)
            # Cycle over all observations
            for nm, data_item in self.data.items():
                if -2 in data_item:     # obs is constant in space and time
                    out[nm] = np.array([data_item[-2]])
                elif -1 in data_item:   # obs is constant in time but not space
                    with h5py.File(data_file, 'r') as hf:
                        data = hf[data_item[-1]][()]
                    out[nm] = interpolate(data, vtx, wts)
                else:                       # obs varies in space and time
                    for ind in range(self.num_data_time_points):
                        # If time point coincides with one of the data time points
                        if dtime == self.time_points[ind]:
                            with h5py.File(data_file, 'r') as hf:
                                data = hf[data_item[ind+1]][()]
                            out[nm] = interpolate(data, vtx, wts)
                        # If time point is between the data time points
                        elif self.time_points[ind] < dtime < self.time_points[ind+1]:
                            with h5py.File(data_file, 'r') as hf:
                                data1 = hf[data_item[ind+1]][()]
                                data2 = hf[data_item[ind+2]][()]
                            out1 = interpolate(data1, vtx, wts)
                            out2 = interpolate(data2, vtx, wts)
                            delta_t = self.time_points[ind+1]-self.time_points[ind]

                            # Interpolate between two data points
                            out[nm] = (out2-out1)/(delta_t)*(
                                dtime-self.time_points[ind])+out1
        else:  # return data corresponding to all points in the data file
            for nm, data_item in self.data.items():
                if -2 in data_item:     # obs is constant in space and time
                    out[nm] = data_item[-2]*np.ones(self.num_xy_points)
                elif -1 in data_item:   # obs is constant in time but not space
                    with h5py.File(data_file, 'r') as hf:
                        data = hf[data_item[-1]][()]
                    out[nm] = data
                else:
                    for ind in range(self.num_data_time_points):
                        # If time point coincides with one of the data time points
                        if dtime == self.time_points[ind]:
                            with h5py.File(data_file, 'r') as hf:
                                data = hf[data_item[ind+1]][()]
                            out[nm] = data
                        elif self.time_points[ind] < dtime < self.time_points[ind+1]:
                            with h5py.File(data_file, 'r') as hf:
                                out1 = hf[data_item[ind+1]][()]
                                out2 = hf[data_item[ind+2]][()]
                            delta_t = self.time_points[ind+1]-self.time_points[ind]
                            # Interpolate between two data points
                            out[nm] = (out2-out1)/(delta_t)*(
                                dtime-self.time_points[ind])+out1
        return out

    def __call__(self, time, vtx=None, wts=None):
        """
        Return observation data at the point of interest.

        :param time: time point (in days) for which observations (e.g.,
            pressure and saturation) are to be calculated
        :type time: float

        :param vtx: array of indices of simplex vertices which enclose
            the point of interest; array should have a shape (1,3); indices
            do not exceed the number of data points
        :type vtx: numpy.array of int

        :param wts: array of weights of data at simplex vertices which enclose
            the point of interest; array should have a shape (1,3); weights
            values should be between 0 and 1, and sum up to 1
        :type wts: numpy.array of floats

        :returns: out - dictionary of observations; keys depend on the data
            provided in the data files. Possible keys:
            ['pressure','CO2saturation', 'salinity', 'temperature', 'area'].
        """
        if not self.created:
            self.create_data_interpolators(self.triangulation)

        # Convert days to years for interpolation
        dtime = time/365.25

        # Check whether time point is within the acceptable range
        if (dtime < self.time_points[0]) or (dtime > self.time_points[-1]):
            range_str = ', '.join((str(self.time_points[0]), str(self.time_points[-1])))

            err_msg = ''.join(['Time point t = {} years provided by ',
                                'system model is beyond the available time ',
                                'range [{}] of the data in {}.']).format(
                                    dtime, range_str, self.data_file)
            logging.error(err_msg)
            raise ValueError(err_msg)

        if self.hdf5_data_format:
            out = self.calculate_hdf5_based_output(dtime, vtx=vtx, wts=wts)
        else:
            out = self.calculate_csv_based_output(dtime, vtx=vtx, wts=wts)
        return out


def test_case1():
    # The code below (test 1) tests only interpolator and plotting/animation
    # capabilities but it needs a system model for the parent parameter.
    # Thus, we need to create a system model first.
    # Define logging level
    logging.basicConfig(level=logging.DEBUG)

    # Create system model
    sm = SystemModel()

    # Create interpolator
    int1 = sm.add_interpolator(
        ReservoirDataInterpolator(
            name='int1', parent=sm, header_file_dir=os.path.join(
                '..', 'components', 'reservoir', 'lookuptables', 'Kimb_54_sims'),
            time_file='time_points.csv',
            data_file='Reservoir_data_sim02.csv',
            index=2,
            signature={'logResPerm': -13.3, 'reservoirPorosity': 0.25,
                       'logShalePerm': -18.7},
            default_values={'salinity': 0.1, 'temperature': 50.0}),
        intr_family='reservoir')

    # Print signature of the interpolator
    msg = 'Signature of created interpolator is {}'.format(int1.signature)
    logging.debug(msg)

    # Setup location of interest
    locX, locY = [37478.0, 48333.0]

    # Calculate weights of the location of interest
    vertices, weights = interp_weights(int1.triangulation, np.array([[locX, locY]]))

    for t in 365.25*10.*np.arange(11):
        out = int1(t, vertices, weights)
        print(' '.join(['At time t = {} years pressure is {} MPa',
                        'and CO2 saturation is {}.']).format(
                            t/365.25, out['pressure'][0]/1.0e+6,
                            out['CO2saturation'][0]))

    # Close all figures
    plt.close("all")

    # Show all data at 120 years
    int1.show_data(120.0)

    logging.info(''.join(['If running code in IPython make sure to ',
                          'switch to the interactive mode to see ',
                          'the animation.']))
    # Show all data of interpolator int1 for available time points
    int1.show_data(obs_to_show=['pressure', 'CO2saturation'])

    # to_save_figure = False
    # # In order to save the animated figure ffmpeg should be installed.
    # # If using Anaconda, in Anaconda prompt enter
    # # conda install -c conda-forge ffmpeg
    # # Worked on Windows 7
    # # Advice is taken from:
    # # https://stackoverflow.com/questions/13316397/
    # # matplotlib-animation-no-moviewriters-available/14574894#14574894
    # if to_save_figure:
    #     for obs_nm in int1.sca_plt:
    #         int1.sca_plt[obs_nm].ani.save(obs_nm+'_anim.mp4')


def test_case2():
    num_cases = 3
    points_set = {}
    points_set[1] = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    points_set[2] = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    points_set[3] = np.array([[0, 0, 1], [0, 1, 2], [1, 0, 3], [1, 1, 4],
                              [2, 0, 1], [1, 2, 3]])

    data_val = {}
    data_val[1] = np.array([1, 2, 2, 3])
    data_val[2] = np.array([1, 2, 2, 3, 2, 3, 3, 4])
    data_val[3] = np.array([1, 2, 2, 3, 4, 2])

    # Calculate triangulation for data points
    triangulation = {}
    for ind in range(1, num_cases+1):
        triangulation[ind] = Delaunay(points_set[ind])

    # Plot the triangulation for the first set of points
    plt.triplot(points_set[1][:, 0], points_set[1][:, 1],
                triangulation[1].simplices, color='red')
    plt.title('Triangulation for 2d points')

    for ind in range(1, num_cases+1):
        print('Triangulation {} simplices:\n'.format(ind),
              triangulation[ind].simplices)

    # Setup locations of interest
    location = {}
    location[1] = [0.3, 0.5]        # 2d
    location[2] = [0.1, 1, 0.3]     # 3d
    location[3] = [0.0, 0.5, 1.5]   # 3d

    # Calculate vertices of the simplices and weights of the locations of interest
    vertices = {}
    weights = {}
    for ind in range(1, num_cases+1):
        vertices[ind], weights[ind] = interp_weights(
            triangulation[ind], np.array([location[ind]]))

    for ind in range(1, num_cases+1):
        print('Vertices (indices) and weights for location {} :\n'.format(ind),
              vertices[ind], '\n', weights[ind])

    # The output
    # Triangulation 1 simplices:
    #  [[3 1 0]
    #  [2 3 0]]
    # Triangulation 2 simplices:
    #  [[3 2 4 0]
    #  [3 1 4 0]
    #  [3 6 2 4]
    #  [3 6 7 4]
    #  [3 5 1 4]
    #  [3 5 7 4]]
    # Triangulation 3 simplices:
    #  [[2 3 5 4]
    #  [1 2 4 0]
    #  [1 2 5 4]
    #  [1 2 3 5]]
    # Vertices (indices) and weights for location 1 :
    #  [[3 1 0]]
    #  [[0.3 0.2 0.5]]
    # Vertices (indices) and weights for location 2 :
    #  [[3 6 2 4]]
    #  [[0.3 0.1 0.6 0. ]]
    # Vertices (indices) and weights for location 3 :
    #  [[1 2 4 0]]
    #  [[0.5 0.  0.  0.5]]


if __name__ == "__main__":

    # Setup test
    test = 1

    if test == 1:
        test_case1()

    elif test == 2:
        test_case2()
