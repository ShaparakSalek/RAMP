# -*- coding: utf-8 -*-
"""
Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openiam import SystemModel
from ramp.components.base import process_time_points, get_indices
from ramp.utilities.data_readers import default_bin_file_reader
from ramp.components.base import MonitoringTechnology
from ramp.components.seismic import SeismicSurveyConfiguration, SeismicDataContainer


class SeismicMonitoring(MonitoringTechnology):

    def __init__(self, name, parent, survey_config, time_points,
                 source_ind=None, receiver_ind=None):
        """
        Constructor of SeismicMonitoring class.

        Parameters
        ----------
        name : str
            Name of SeismicMonitoring class instance under which it will
            be known to its parent
        parent : SystemModel class instance
            System model to which the instance belongs
        survey_config : SeismicSurveyConfiguration class instance
            Seismic Survey containing information about locations of sources
            and receivers associated with an incoming linked data
        time_points : str or numpy.array, optional
            Path to the file containing time points associated with data
            if of type string.
            Array of time points associated with data, i.e.,
            time points at which data is provided if of type numpy.array
        source_ind : list
            Indices of sources that will be used for new configuration.
        receiver_ind : list
            Indices of receivers that will be used for new configuration.

        Returns
        -------
        Instance of SeismicMonitoring class.

        """
        # Setup keyword arguments of the 'model' method provided by the system model
        model_kwargs = {'time_point': 0.0}  # default value of 0 days
        super().__init__(name, parent, model_kwargs=model_kwargs)

        # Setup keys of gridded observation
        self.grid_obs_keys = ['NRMS']

        # Add type attribute
        self.class_type = 'SeismicMonitoring'

        # Setup additional attributes
        self.configuration = survey_config

        # Get x, y, z coordinates associated with sources and receivers
        self.coordinates = {'source_xyz': self.configuration.sources.coordinates,
                            'receiver_xyz': self.configuration.receivers.coordinates}

        # Get number of sources and receivers
        self.num_sources = len(self.coordinates['source_xyz'])
        self.num_receivers = len(self.coordinates['receiver_xyz'])

        # Check whether indices are provided and update if needed
        # Assign default values of sources and receivers indices
        self.source_ind = list(range(self.num_sources))
        self.receiver_ind = list(range(self.num_receivers))
        self.sub_configuration = self.configuration

        if source_ind is not None or receiver_ind is not None:
            self.update_indices(source_ind=source_ind, receiver_ind=receiver_ind)

        self.coordinates = {
            'source_xyz': self.sub_configuration.sources.coordinates,
            'receiver_xyz': self.sub_configuration.receivers.coordinates}

        # Process time_points argument
        self.time_points = process_time_points(
            time_points, data_directory=None,
            component_class=self.class_type, name=self.name)

        # TODO System model does not yet know how to handle those
        self.run_time_indices = get_indices(self._parent.time_array,
                                            self.time_points*365.25)

    def update_indices(self, source_ind=None, receiver_ind=None):
        """
        Update indices of sources and receivers associated with a given
        instance of SeismicMonitoring class.

        Parameters
        ----------
        source_ind : list
            Indices of sources to be used, indices are between 0 and ns.
        receiver_ind : list
            Indices of receivers to be used, indices are between 0 and nr.

        Returns
        -------
        None.

        """
        # Check whether indices or sources and receivers are provided
        if source_ind is not None:
            self.source_ind = source_ind
        if receiver_ind is not None:
            self.receiver_ind = receiver_ind
        self.sub_configuration = self.configuration.create_survey_configuration(
            name='{}_subconfig'.format(self.name),
            source_indices=self.source_ind,
            receiver_indices=self.receiver_ind)

    def process_data(self, p, time_point=None, data=None, baseline=None,
                     source_ind=None, receiver_ind=None,
                     first_time_ind=0, num_time_samples=None):
        """
        Calculate NRMS metric for all traces defined by provided indices of sources
        and receivers.

        Parameters
        ----------
        p : dict
            Parameters of component
        time_point : float
            time point (in days) for which the component outputs
            are to be calculated; by default, its value is 0 days
        data : numpy.ndarray of shape (ns, nr, nt)
            Seismic data to be processed: ns - number of sources, nr - number of
            receivers, nt - number of time sample intervals.
        baseline : numpy.ndarray of shape (ns, nr, nt)
            Baseline data to be processed
        source_ind : list
            Indices of sources to be used, indices are between 0 and ns.
        receiver_ind : list
            Indices of receivers to be used, indices are between 0 and nr.
        first_time_ind : int
            Index (between 0 and nt-1) of the first data points in each trace
            from which the calculations of NRMS will be performed
        num_time_samples : int
            Number of data points to use from each trace to calculate NRMS values

        Returns
        -------
        Dictionary of outputs with keys being names of metrics calculated from
        input data. Possible keys are 'NRMS', 'ave_NRMS', 'max_NRMS', 'min_NRMS'.

        """
        if source_ind is not None or receiver_ind is not None:
            self.update_indices(source_ind=source_ind, receiver_ind=receiver_ind)
            self.coordinates = {
                'source_xyz': self.sub_configuration.sources.coordinates,
                'receiver_xyz': self.sub_configuration.receivers.coordinates}

        if num_time_samples is None:
            num_time_samples = data.shape[2]-first_time_ind
        else:
            num_time_samples = min(num_time_samples, data.shape[2]-first_time_ind)

        nrms = calculate_nrms(data, baseline, self.source_ind, self.receiver_ind,
                              list(range(first_time_ind, first_time_ind+num_time_samples)))

        ave_nrms = np.mean(nrms)
        max_nrms = np.max(nrms)
        min_nrms = np.min(nrms)

        data_out = {'NRMS': nrms, 'ave_NRMS': ave_nrms,
                    'max_NRMS': max_nrms, 'min_NRMS': min_nrms}

        time_index = np.where(self.time_points==time_point/365.25)[0][0]
        if time_index == 0:
            out = {**data_out, **self.coordinates}
            return out
        else:
            return data_out

        return out


def calculate_nrms(data, baseline, source_ind, receiver_ind, time_ind):
    """
    Calculate NRMS from the provided data based on the indices selected
    for sources, receivers and time points.

    Parameters
    ----------
    data : numpy.ndarray of shape (ns, nr, nt)
        Seismic data to be processed: ns - number of sources, nr - number of
        receivers, nt - number of time sample intervals.
    baseline : numpy.ndarray of shape (ns, nr, nt)
        Baseline data to be processed
    source_ind : list
        Indices of sources to be used, indices are between 0 and ns.
    receiver_ind : list
        Indices of receivers to be used, indices are between 0 and nr.
    time_ind : list
        Indices of time points to be used, indices are between 0 and nt.

    Returns
    -------
    NRMS values as numpy.ndarray of shape (len(source_ind), len(receiver_ind)).

    """
    trace_t = data[np.ix_(source_ind, receiver_ind, time_ind)]

    trace_t0 = baseline[np.ix_(source_ind, receiver_ind, time_ind)]

    trace_t_sqr = trace_t**2
    trace_t0_sqr = trace_t0**2
    diff_trace_sqr = (trace_t - trace_t0)**2

    nrms = 200*(np.sum(diff_trace_sqr, axis=2)**0.5)/(
        np.sum(trace_t_sqr, axis=2)**0.5 + np.sum(trace_t0_sqr, axis=2)**0.5)

    return nrms

def test_seismic_monitoring():
    # Define keyword arguments of the system model
    final_year = 90
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'seismic'
    data_directory = os.path.join(
        '..', '..', '..', '..', 'data', 'user', 'seismic')
    output_directory = os.path.join(
        '..', '..', '..', '..', 'examples', 'user', 'output', 'test_seismic_monitoring')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (1251, 101, 9),
                          'move_axis_destination': [-1, -2, -3]}
    time_points = np.linspace(10.0, final_year, num=num_intervals+1)
    num_time_points = len(time_points)
    num_scenarios = 5
    scenarios = list(range(1, num_scenarios+1))
    family = 'seismic'
    data_setup = {}
    for ind in scenarios:
        data_setup[ind] = {'folder': os.path.join('data_sim{:04}'.format(ind), 'data')}
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
                             data_reader_kwargs=data_reader_kwargs,
                             presetup=True))
    # Add parameters of the container
    scen_ind = np.random.randint(0, num_scenarios)
    selected_scenario = scenarios[scen_ind]
    print('Scenario to be run', selected_scenario)
    dc.add_par('index', value=selected_scenario, vary=False)
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    # ------------- Add seismic monitoring technology -------------
    smt = sm.add_component_model_object(
        SeismicMonitoring(name='smt', parent=sm, survey_config=survey_config,
                          time_points=time_points))
    # Add keyword arguments linked to the data container outputs
    smt.add_kwarg_linked_to_obs('data', dc.linkobs['seismic'], obs_type='grid')
    smt.add_kwarg_linked_to_obs(
        'baseline', dc.linkobs['baseline_seismic'], obs_type='grid')
    # Add gridded observation
    smt.add_grid_obs('NRMS', constr_type='matrix', output_dir=output_directory)
    # Add scalar observations
    for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
        smt.add_obs(nm)

    print('-----------------------------')
    print('Forward simulation started...')
    print('-----------------------------')
    sm.forward()
    print('-----------------------------')
    print('Forward simulation finished.')
    print('-----------------------------')

    # Get saved gridded observations from files
    time_ind = final_year//10 - 1
    source_ind = 5
    delta_seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'delta_seismic', output_directory, indices=[time_ind], rlzn_number=0)[0]
    seismic = sm.collect_gridded_observations_as_time_series(
        dc, 'seismic', output_directory, indices=[time_ind], rlzn_number=0)[0]
    nrms = sm.collect_gridded_observations_as_time_series(
        smt, 'NRMS', output_directory, indices=[time_ind], rlzn_number=0)

    # Export scalar observations
    metrics = {}
    for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
        metrics[nm] = sm.collect_observations_as_time_series(smt, nm)

    # Plot results
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(121)
    ax.imshow(seismic[source_ind, :, :].T, cmap='gray', aspect='auto')
    ax.set_title('Seismic data at {} years (Source {})'.format(
        (time_ind+1)*10, source_ind+1))
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
    ax.set_title('Seismic data difference at {} years (Source {})'.format(
        (time_ind+1)*10, source_ind+1))
    x = np.linspace(0, 100, num=6)
    xlabels = 4000 + np.linspace(0, 4000, num=6)
    ax.set_xticks(x, labels=xlabels)
    ax.set_xlabel('Receiver location, [m]')
    ax.set_yticks(y, labels=ylabels)

    # Plot NRMS histogram
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    res = ax.hist(nrms[0], bins=25)
    ax.set_xlabel('NRMS values')
    ax.set_ylabel('Counts')
    ax.set_title('t = {} years'.format((time_ind+1)*10))

    # Plot time changes in NRMS metrics
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    for ind, (nm, obs_label) in enumerate(zip(['ave_NRMS', 'min_NRMS', 'max_NRMS'],
                                              ['Average', 'Minimum', 'Maximum'])):
        axs[ind].plot(time_points, metrics[nm], '-ok')
        axs[ind].set_xlabel('Time, [years]')
        axs[ind].set_ylabel(obs_label + ' NRMS')
    fig.tight_layout()


if __name__ == "__main__":

    test_seismic_monitoring()
