# -*- coding: utf-8 -*-
"""
Component that will serve as source of data.

Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import logging
import os
import sys
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from openiam import SystemModel, ComponentModel
from ramp.utilities.data_readers import default_bin_file_reader


class DataContainer(ComponentModel):
    def __init__(self, name, parent, family, obs_name, data_directory='',
                 data_setup=None, time_points=None, baseline=False,
                 data_reader=None, data_reader_kwargs=None,
                 data_reader_time_index=False,
                 container_class='DataContainer',
                 model_kwargs=None, presetup=False):
        """
        Component of RAMP tool providing means to load user data into simulations.

        Parameters
        ----------
        name : str
            Name of data container under which it will be known in its parent
        parent : SystemModel class instance
            System model to which the data container belongs
        family : str
            Family of data containers to which this one belongs to.
        obs_name : str or list of strings
            Name(s) of observation with which the loaded data will be associated
        data_directory : str, optional
            Path to the folder containing data (and possibly setup and time points)
            files. The default is None.
        data_setup : str or dict, optional
            Path to the csv file providing setup of multiple data files
            associated with the given data set if of type string. Dictionary with
            index type keys providing setup of multiple data files associated with
            the given data set if of type dict. The keys are integers corresponding
            to indices of particular simulation. The values are dictionaries with
            the following keys
                'signature': dict, optional, can contain information about parameters
                that can be sampled
                'folder': str, optional, path to a specific if data files for different
                simulations are kept in separate folders. key is optional if
                subsequent keys containing pathes also contain information about folders.
                't1', ..., 'tn' - keys corresponding to data at different time
                points. n is a number of time points provided with time_points
                argument.
        time_points : str or numpy.array, optional
            Path to the file containing time points associated with data
            if of type string.
            Array of time points associated with data, i.e.,
            time points at which data is provided if of type numpy.array
        baseline : boolean
            Flag indicating whether data setup contains the baseline data.
            If it does it should correspond to the file with key 't1',
            or column 't1'. If baseline is True then for each time point
            the component can also return a difference between a baseline
            and current time point data. In the latter case a new observation
            called 'delta_###' where '###' is obs_name can also be returned.
        data_reader : str or function
            name of function that reads data file and returns a numpy.ndarray
            containing data or dictionary of numpy.ndarrays
        data_reader_kwargs : dict
            Dictionary of additional keyword arguments applicable to a particular
            data reader method
        data_reader_time_index : boolean
            Flag indicating whether data reader requires time index to be passed
            as one of the key arguments. This might be needed if all time points
            data is saved in one data file versus multiple and file name does not
            determine what time point the data corresponds to
        presetup : boolean
            Flag indicating whether the add_obs_to_be_linked method should be
            used on this object to add observations to be used as inputs for
            other components

        Returns
        -------
        Instance of DataContainer class.

        """
        # Setup keyword arguments of the 'self.model' method provided by the system model
        if model_kwargs is None:
            model_kwargs = {'time_point': 0.0}  # default value of 0 days
        elif 'time_point' not in model_kwargs:
            model_kwargs['time_point'] = 0.0

        super().__init__(name, parent, model=self.export,
                         model_kwargs=model_kwargs)
        self.family = family

        # Setup baseline data related attributes
        self.baseline_data = {}  # empty dictionary
        self.baseline_in = baseline

        # Process obs_name argument
        if isinstance(obs_name, str):
            self.obs_names = [obs_name]
        elif isinstance(obs_name, list):
            self.obs_names = obs_name

        # Setup gridded observations keys
        self.grid_obs_keys = self.obs_names

        # If we need to deal with baseline data, add delta-type of observation
        if self.baseline_in:
            for nm in self.obs_names:
                self.grid_obs_keys = self.grid_obs_keys +[
                    'delta_{}'.format(nm), 'baseline_{}'.format(nm)]

        # Process data files related attributes
        self.data_directory = data_directory
        self.reader = data_reader
        self.reader_kwargs = {}
        if data_reader_kwargs is not None:
            self.reader_kwargs = data_reader_kwargs
        self.reader_time_index = data_reader_time_index

        # Add type attribute
        self.class_type = container_class

        # Process time_points argument
        self.time_points = process_time_points(
            time_points, data_directory=data_directory,
            component_class=self.class_type, name=self.name)

        # Process data_setup argument
        self.data_setup = {}  # initialize dictionary
        self.indices = []     # list of index keys
        self.num_indices = 0
        self.signatures = []  # if other than index parameters are involved
        self.process_data_setup(data_setup, data_directory=data_directory)

        # Add default parameter
        # index value can be between minimum and maximum indices in the linked
        # data set; -2 means that index was not sampled
        self.add_default_par('index', value=-2)

        # Check whether additional presetup is requested: in particular, one
        # related to the obs_to_be_linked
        if presetup:
            for nm in self.obs_names:
                self.add_obs_to_be_linked(nm, obs_type='grid')
                if self.baseline_in:
                    for key in ['delta_', 'baseline_']:
                        self.add_obs_to_be_linked(key+nm, obs_type='grid')

        # TODO System model does not yet know how to handle those
        if self._parent.time_array is not None:
            self.run_time_indices = get_indices(self._parent.time_array,
                                                self.time_points*365.25)

    def process_data_setup(self, data_setup, data_directory=None):
        """

        Parameters
        ----------
        data_setup : str or dict, optional
            Path to the csv file providing setup of multiple data files
            associated with the given data set if of type string. Dictionary with
            index type keys providing setup of multiple data files associated with
            the given data set if of type dict. The keys are integers corresponding
            to indices of particular simulation. The values are dictionaries with
            the following keys
                'signature': dict, optional, can contain information about parameters
                that can be sampled
                'folder': str, optional, path to a specific if data files for different
                simulations are kept in separate folders. key is optional if
                subsequent keys containing pathes also contain information about folders.
                't1', ..., 'tn' - keys corresponding to data at different time
                points. n is a number of time points provided with time_points
                argument.
        data_directory : str, optional
            Path to the folder containing data (and possibly setup and time points)
            files. The default is None.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If one of the files which name is specified in data_setup file
            or dictionary does not exist
        TypeError
            If provided argument data_setup is neither of string nor
            of dictionary type

        """
        if isinstance(data_setup, dict):  # dictionary containing setup
            self.configure_data_setup(data_setup)
        elif isinstance(data_setup, str): # csv file with all the setup information
            if data_directory:
                setup_file = os.path.join(data_directory, data_setup)
            else:
                setup_file = data_setup
            # TODO For future development we might want to be able to
            # find files located in relative locations with respect to RAMP
            # directory, for example.
            if os.path.isfile(setup_file):
                data_setup_dict = self.read_data_setup_file(setup_file)
                self.configure_data_setup(data_setup_dict)
            else:
                raise FileNotFoundError('File {} is not found.'.format(setup_file))
        else:
            raise TypeError('Argument data_setup is of wrong type.')

    def configure_data_setup(self, data):
        """
        Process information provided through data setup dictionary and save
        into corresponding instance attributes.

        Parameters
        ----------
        data : dict
            Dictionary with index type keys providing setup of multiple data files
            associated with the given data set if of type dict. The keys are
            integers corresponding to indices of particular simulation.
            The values are dictionaries with the following keys
                'signature' : dict, optional
                    Dictionary that can contain information about parameters
                    that can be sampled
                'folder' : str, optional
                    Path to a specific folder if data files for different
                    simulations are kept in separate folders. key is optional if
                    subsequent keys containing paths also contain information
                    about folders.
                't1', ..., 'tn' : str
                    Keys corresponding to data file names at different time
                    points. n is a number of time points provided with time_points
                    argument.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If one of the keys within data dictionary has an unexpected name

        """
        # Get keys and save them into indices attribute
        self.indices = list(data.keys())
        self.num_indices = len(self.indices)
        # Initialize list of signatures
        signatures = self.num_indices*[None]

        # Go over all indices
        for key_ind, key_val in enumerate(self.indices):
            self.data_setup[key_val] = {'folder': '',
                                        'time_indices': []}
            # Cycle over all keys provided
            for data_nm in data[key_val]:
                # TODO need to process signature to get keys of signature
                # and save them to know which parameters can be sampled
                if data_nm == 'signature':
                    signatures[key_ind] = data[key_val]['signature']
                elif data_nm == 'folder':
                    self.data_setup[key_val]['folder'] = data[key_val]['folder']
                elif isinstance(data_nm, str) and data_nm[0] == 't':
                    try:
                        time_index = int(data_nm[1:])
                    except ValueError:
                        warn_msg = ''.join(['Unexpected key {} is provided ',
                                            'in the dictionary data_setup']).format(
                                                data_nm)
                        logging.warning(warn_msg)
                    else:
                        # Copy file name
                        self.data_setup[key_val][time_index] = data[key_val][data_nm]
                        self.data_setup[key_val]['time_indices'].append(time_index)

        # If there is at least one None in signatures we won't allow for other
        # parameters sampling except index
        if not (None in signatures):
            self.signatures = signatures


    @staticmethod
    def read_data_setup_file(data_setup_file):
        """
        Read file containing information about setup associated with given
        component.

        Parameters
        ----------
        data_setup_file : str
            Path to the csv file providing setup of multiple data files
            associated with the given data set.

        Returns
        -------
        setup_data : dict
            Dictionary needed for the data setup attribute of component.

        """
        # Read file content
        df = pd.read_csv(data_setup_file, delimiter=',')

        # Get names of columns
        header = list(df.columns.values)

        if header[0] != 'index':
            err_msg = "".join([
                "The first column of the file {} should contain the indices ",
                "associated with a given data set (filename) and ",
                "be named 'index'. Please update the file by adding ",
                "the corresponding column and try again. "]).format(data_setup_file)
            logging.error(err_msg)
            raise NameError(err_msg)

        # Get indices
        indices = df['index'].iloc[:].values
        num_indices = len(indices)

        # Initialize dictionary with data
        setup_data = {ind: {'signature': {}} for ind in indices}
        # Go over names in header
        for nm in header[1:]:
            if nm == 'folder':
                for ind, scen in enumerate(indices):
                    setup_data[scen]['folder'] = df['folder'].iloc[ind]
            elif nm[0] == 't':
                try:
                    t_ind = int(nm[1:])
                except ValueError:
                    for ind, scen in enumerate(indices):
                        setup_data[scen]['signature'][nm] = df[nm].iloc[ind]
                else:
                    for ind, scen in enumerate(indices):
                        setup_data[scen][nm] = df[nm].iloc[ind]

        return setup_data

    def check_input_parameters(self, p):
        """
        Check whether input parameters fall within specified boundaries.

        Parameters
        ----------
        p : dict
            Dictionary of input parameters for a given DataContainer instance.

        Raises
        ------
        ValueError
            If index parameter is not in the instance indices attribute.

        """
        if 'index' in p:
            index = p['index']
            if index not in self.indices:
                err_msg = ''.join([
                    'Value {} of index parameter does not correspond ',
                    'to any of the linked data sets.']).format(index)
                logging.error(err_msg)
                raise ValueError(err_msg)

    def export(self, p, time_point=None):
        """
        Read and return data corresponding to the requested index value and time
        point.

        Parameters
        ----------
        p : dict
            Parameters and values associated with data to be returned.
        time_point : float, optional
            Time point at which data is to be returned.

        Returns
        -------
        out : dict
            Dictionary of outputs with keys being names of data extracted from files.

        """
        # TODO Check/make sure that data returned by any data container
        # (e.g., seismic data) can be saved as numpy.npz files or another format
        # if needed right in the system model

        # Obtain the default values of the parameters from dictionary of default parameters
        actual_p = {k: v.value for k, v in self.default_pars.items()}

        # Update default values of parameters with the provided ones
        actual_p.update(p)

        # Extract index from updated dictionary
        index = int(actual_p.pop('index'))
        if index == -2:  # user didn't specify an index value
            # Sample default index value (if == -2) or other parameters if applicable
            if actual_p:  # other parameters are present after removing index
                # TODO need to add code addressing sampling of parameters
                # signature will still be converted to index value
                pass
            else:  # Sample default value of index corresponding to the first scenario
                index = self.indices[0]
                warn_msg = ''.join([
                    'Parameter index was not setup. Default value of {} will ',
                    'be used in a simulation.']).format(index)
                logging.warning(warn_msg)

        # Initialize outputs dictionary
        out = {}

        # Check that time_point coincide with one of the saved time_points
        try:
            time_index = np.where(self.time_points==time_point/365.25)[0][0]
        except IndexError: # time index was not found
            err_msg = ''.join([
                'Observation {} of data container {} is requested at time ',
                'point {} not presented in the linked data set. Please check ',
                'setup of the data container.']).format(
                    self.obs_names[0], self.name, time_point)
            logging.error(err_msg)
            raise IndexError(err_msg)
        else:
            # Get filename with data at initial point corresponding
            # to the specified index
            file_name = os.path.join(self.data_directory,
                                     self.data_setup[index]['folder'],
                                     self.data_setup[index][time_index+1])
            # Get data
            if self.reader_time_index:
                self.reader_kwargs['time_index'] = time_index
            obtained_data = self.reader(file_name, **self.reader_kwargs)
            if isinstance(obtained_data, dict):
                for obs_nm in self.obs_names:
                    try:
                        out[obs_nm] = obtained_data[obs_nm]
                    except KeyError:
                        err_msg = ''.join([
                            'Data files linked to data container {} do not ',
                            'contain data corresponding to observation {}. ',
                            'Please check setup of the data container.']).format(
                                self.name, obs_nm)
                        logging.error(err_msg)
                        raise KeyError(err_msg)
            else:  # if no dictionary is returned, assume this is the data requested
                if len(self.obs_names) == 1:
                    out[self.obs_names[0]] = obtained_data
                else:
                    err_msg = ''.join([
                        'Several observation names were provided ({}) but ',
                        'only one data structure is returned for data container {}. ',
                        'Please check setup of the data container.']).format(
                            self.obs_names, self.name)
                    logging.error(err_msg)
                    raise ValueError(err_msg)

            # Check whether baseline data is present
            if self.baseline_in:
                if time_index == 0:  # initial_point
                    # Assign baseline data to the instance attribute
                    for nm in self.obs_names:
                        self.baseline_data[nm] = out[nm]
                        out['delta_{}'.format(nm)] = np.zeros(
                            out[nm].shape, dtype=np.float32)
                else:
                    # Obtain difference between baseline and current time point data
                    for nm in self.obs_names:
                        out['delta_{}'.format(nm)] = \
                            out[nm] - self.baseline_data[nm]
                for nm in self.obs_names:
                    out['baseline_{}'.format(nm)] = self.baseline_data[nm]

            return out

    def plot(self, data=None, p=None, time_point=None,
             axis_labels=None, title=None):
        """
        Plot data corresponding to a given index or set of parameters and time
        points.

        For now, method can be only used to plot 1d or 2d data. If data is of 3
        or higher number of dimensions the warning message is shown.

        Parameters
        ----------
        p : dict
            dictionary of input parameters to sample, read and show the data
        time_point : float or int in years
            time point at which the data is to be shown. If time_point is None
            then by default the data corresponding to t0, the first point
            in the time array, is displayed.
        axis_labels : list
            list of labels to use for axis 1 and axis 2 on the produced plot.
        title: dict
            dictionary with titles corresponding to different keys in data
            dictionary

        Returns
        -------
        None

        """
        if data is None:
            # Get data first
            if p is None:
                p = {}
            if time_point is None:
                time_point = self.indices[0]*365.25
            output_data = self.export(p=p, time_point=time_point)
            data = output_data[self.obs_names[0]]

        # Get data shape
        data_shape = data[list(data.keys())[0]].shape

        # Check the data shape
        if len(data_shape) > 2:
            warn_msg = 'Method "plot" does not work on data of dimension 3 or higher.'
            logging.warning(warn_msg)
            return
        else:
            if 1 in data_shape:
                fig = plt.figure(figsize=(12, 4))
                ax = fig.add_subplot(111)
                ax.plot(data, '-')
            else:
                for key in data:
                    fig = plt.figure(figsize=(12, 4))
                    ax = fig.add_subplot(111)
                    ax.imshow(data[key].T, cmap='gray', aspect='auto')


def get_indices(complete_array, sub_array):
    """
    Get indices of elements in sub_array with respect to elements in
    complete_array

    Parameters
    ----------
    complete_array : numpy.array
        Array whose elements are assumed to be arranged in increasing order
    sub_array : numpy.array
        Array whose elements might be the same as some elements within the complete_array

    Returns
    -------
    indices : list
        List of index positions of sub_array elements within the complete_array.

    """
    # Find the largest value of complete array
    max_t = complete_array[-1]
    indices = []
    for elem in sub_array:
        if elem <= max_t:
            indices.append(np.where(complete_array==elem)[0][0])
    return indices


def process_time_points(time_points, data_directory=None,
                        component_class=None, name=None):
    """
    Analyze type of data provided in time_points argument of the constructor.

    Parameters
    ----------
    time_file : TYPE
        DESCRIPTION.
    data_directory : str, optional
        Path to the folder containing data (and possibly setup and time points)
        files. The default is None.
    component_class : str
        Name of class for which the the time points processing is being done
    name : str
        Name of component class instance for which the the time points
        processing is being done

    Raises
    ------
    ValueError if time_points argument is None
    FileNotFoundError

    Returns
    -------
    time_points : numpy.ndarray
        Array of time points read from a file or originally provided array.

    """
    if time_points is None:
        err_msg = 'No time points are provided for {} {}.'.format(
            component_class, name)
        logging.error(err_msg)
        raise ValueError(err_msg)
    if isinstance(time_points, np.ndarray):  # no need to read any data
        return time_points
    if isinstance(time_points, str):       # time_points is of type string
        if data_directory:
            time_file = os.path.join(data_directory, time_points)
        else:
            time_file = time_points
        # TODO For future development we might want to be able to
        # find files located in relative locations with respect to RAMP
        # directory, for example.
        if os.path.isfile(time_file):
            time_points = np.loadtxt(time_file, delimiter=",", dtype='f8') # in years
            return time_points

        # If file path is incorrect or does not exist raise error
        raise FileNotFoundError('File {} is not found.'.format(time_file))

    # If provided time_points is of unexpected type raise error
    raise TypeError('Argument time_points is of wrong type.')


def test_data_container():
    """
    Test work of DataContainer in a simple scenario.

    Returns
    -------
    None.

    """
    # Define keyword arguments of the system model
    final_year = 90
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'velocity'
    data_directory = os.path.join('..', '..', '..', '..', 'data', 'user', 'velocity')
    output_directory = os.path.join('..', '..', '..', '..', 'examples', 'user',
                                    'output', 'test_data_container')
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


if __name__ == "__main__":

    test_data_container()
