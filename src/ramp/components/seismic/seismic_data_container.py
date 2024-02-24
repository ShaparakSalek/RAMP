# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:01:50 2023

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import logging
import os
import sys
import numpy as np

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from openiam import SystemModel
from ramp.components.base import DataContainer
from ramp.components.seismic import SeismicSurveyConfiguration
from ramp.utilities.data_readers import default_bin_file_reader

SDC_GRID_OBSERVATIONS = ['source_xyz', 'receiver_xyz']


class SeismicDataContainer(DataContainer):

    def __init__(self, name, parent, survey_config, total_duration, sampling_interval,
                 family='seismic', obs_name='seismic', data_directory=None,
                 data_setup=None, time_points=None, baseline=False,
                 data_reader=default_bin_file_reader, data_reader_kwargs=None,
                 data_reader_time_index=False,
                 container_class='SeismicDataContainer',
                 presetup=False):
        """
        Constructor of SeismicDataContainer class.

        Parameters
        ----------
        name : str
            Name of data container under which it will be known in its parent
        parent : SystemModel class instance
            System model to which the data container belongs
        survey_config : SeismicSurveyConfiguration class instance
            Seismic Survey containing information about locations of sources
            and receivers associated with a linked data
        total_duration : float
            Total duration of a seismic data in sec
        sampling_interval : float
            Time between which seismic data measurements are taken in sec
        family : str
            Family of data containers to which this one belongs to. By default,
            the family is 'seismic' data.
        obs_name : str
            Name of observation with which the loaded data will be associated.
            By default, data linked to the SeismicDataContainer is 'seismic'.
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
            If it does, it should correspond to the file with key 't1',
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
        Instance of SeismicDataContainer class.

        """
        # Setup keyword arguments of the 'model' method provided by the system model
        model_kwargs = {'time_point': 0.0}  # default value of 0 days

        super().__init__(name, parent, family=family, obs_name=obs_name,
                         data_directory=data_directory, data_setup=data_setup,
                         time_points=time_points, baseline=baseline,
                         data_reader=data_reader,
                         data_reader_kwargs=data_reader_kwargs,
                         container_class='SeismicDataContainer',
                         model_kwargs=model_kwargs, presetup=presetup)

        # Add type attribute
        self.class_type = 'SeismicDataContainer'
        self.grid_obs_keys = self.grid_obs_keys + SDC_GRID_OBSERVATIONS

        # Setup additional attributes
        self.configuration = survey_config
        self.total_duration = total_duration
        self.sampling_interval = sampling_interval
        self.num_time_samples = int(total_duration/sampling_interval)+1

        # Get x, y, z coordinates associated with sources and receivers
        self.coordinates = {'source_xyz': self.configuration.sources.coordinates,
                            'receiver_xyz': self.configuration.receivers.coordinates}

        # Get number of sources and receivers
        self.num_sources = len(self.coordinates['source_xyz'])
        self.num_receivers = len(self.coordinates['receiver_xyz'])

        # Add coordinates observations to be linked
        self.add_obs_to_be_linked('source_xyz', obs_type='grid')
        self.add_obs_to_be_linked('receiver_xyz', obs_type='grid')

    def check_data_shape(self, data, name):
        """
        Check whether shape of the data linked to the container is consistent
        with the structure of linked seismic survey configuration.

        Parameters
        ----------
        data : numpy.ndarray
            Data read from the linked data files and returned as output of the
            class.
        name : str
            Observation name of data extracted from data files.

        Returns
        -------
        None.

        """
        d_shape = data.shape
        if d_shape[0] != self.num_sources or d_shape[1] != self.num_receivers or\
                d_shape[2] != self.num_time_samples:
            warn_msg = ''.join(['Shape of the {} data obtained by data reader {}',
                                'is not consistent with the setup of the seismic '
                                'survey configuration.']).format(
                                    name, self.reader)
            logging.warning(warn_msg)

    def export(self, p, time_point=None):
        """
        Read and return data corresponding to the requested index value and time
        point.

        Parameters
        ----------
        p : dict
            Parameters and values associated with data to be returned.
        time_point : float
            Time point at which data is to be returned.

        Returns
        -------
        out : dict
            Dictionary of outputs with keys being names of data extracted from files
            and coordinates of the associated sources and receivers. The keys
            associated with the last two outputs are 'source_xyz' and 'receiver_xyz'.

        """
        data_out = super().export(p, time_point)
        for key in data_out:
            self.check_data_shape(data_out[key], key)

        out = {**data_out, **self.coordinates}
        return out


def test_seismic_data_container():
    """
    Test work of SeismicDataContainer class.

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
    obs_name = 'seismic'
    data_directory = os.path.join('..', '..', '..', '..', 'data', 'user', 'seismic')
    output_directory = os.path.join(
        '..', '..', '..', '..', 'examples', 'user', 'output', 'test_seismic_data_container')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (1251, 101, 9),
                          'move_axis_destination': [-1, -2, -3]}
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)
    num_scenarios = 5
    family = 'seismic'
    data_setup = {}
    for ind in range(1, num_scenarios+1):
        data_setup[ind] = {'folder': os.path.join('data_sim{:04}'.format(ind),
                                                  'data')}
        for t_ind in range(1, num_time_points+1):
            data_setup[ind]['t{}'.format(t_ind)] = 'data_sim{:04}_t{}.bin'.format(
                ind, t_ind*10)
    baseline = True

    # Define coordinates of sources
    num_sources = 9
    sources = np.c_[4000 + np.array([240, 680, 1120, 1600, 2040, 2480, 2920, 3400, 3840]),
                    np.zeros(num_sources),
                    np.zeros(num_sources)]

    # Define coordinates of receivers
    num_receivers = 101
    receivers = np.c_[4000 + np.linspace(0, 4000, num=num_receivers),
                      np.zeros(num_receivers),
                      np.zeros(num_receivers)]

    # Create survey with defined coordinates
    survey_config = SeismicSurveyConfiguration(sources, receivers, name='Test Survey')

    # ------------- Create system model -------------
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # ------------- Add data container -------------
    dc = sm.add_component_model_object(
        SeismicDataContainer(name='dc', parent=sm, survey_config=survey_config,
                             total_duration=2.5,
                             sampling_interval=0.002,
                             family=family, obs_name=obs_name,
                             data_directory=data_directory, data_setup=data_setup,
                             time_points=time_points, baseline=baseline,
                             data_reader=data_reader,
                             data_reader_kwargs=data_reader_kwargs))
    # Add parameters of the container
    dc.add_par('index', value=2, vary=False)
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    print('-----------------------------')
    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()
    print('-----------------------------')
    print('Forward simulation finished.')
    print('-----------------------------')

    # Get saved data from files
    time_ind = final_year//10 - 1
    source_ind = 5
    delta_seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'delta_seismic', output_directory, indices=[time_ind], rlzn_number=0)[0]
    seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'seismic', output_directory, indices=[time_ind], rlzn_number=0)[0]

    # Plot results
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    ax.imshow(seismic[source_ind, :, :].T, cmap='gray', aspect='auto')
    ax.set_title('Seismic data at {} years'.format((time_ind+1)*10))
    y = np.linspace(0, 1200, num=5)
    ylabels = np.linspace(0, 2.5, num=5)
    ax.set_yticks(y, labels=ylabels)
    ax.set_ylabel('Time, [sec]')
    x = np.linspace(0, 100, num=6)
    xlabels = 4000 + np.linspace(0, 4000, num=6)
    ax.set_xticks(x, labels=xlabels)
    ax.set_xlabel('Receiver location, [m]')

    ax = fig.add_subplot(122)
    ax.imshow(delta_seismic[source_ind, :, :].T, cmap='gray', aspect='auto')
    ax.set_title('Seismic data difference at {} years'.format((time_ind+1)*10))
    x = np.linspace(0, 100, num=6)
    xlabels = 4000 + np.linspace(0, 4000, num=6)
    ax.set_xticks(x, labels=xlabels)
    ax.set_xlabel('Receiver location, [m]')
    ax.set_yticks(y, labels=ylabels)


if __name__ == "__main__":

    test_seismic_data_container()
