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
from openiam import SystemModel
from ramp.data_container import default_bin_file_reader, process_time_points, get_indices
from ramp.monitoring_technology import MonitoringTechnology
from ramp.seismic_data_container import SeismicDataContainer
from ramp.seismic_configuration import SeismicSurveyConfiguration
from ramp.seismic_monitoring import SeismicMonitoring


class SeismicEvaluation(MonitoringTechnology):

    def __init__(self, name, parent, time_points, criteria=1):
        """
        Constructor of SeismicEvaluation class.

        Parameters
        ----------
        name : str
            Name of SeismicEvaluation instance under which it will be known to its parent
        parent : SystemModel class instance
            System model to which the instance belongs
        time_points : str or numpy.array, optional
            Path to the file containing time points associated with data
            if of type string.
            Array of time points associated with data, i.e.,
            time points at which data is provided if of type numpy.array
            These are time points at which the evaluation of the data or
            related metrics is performed
        criteria : int
            Index of criteria according to which the metric will be checked against
            the threshold.

        Returns
        -------
        Instance of SeismicEvaluation class.

        """
        # Setup keyword arguments of the 'model' method provided by the system model
        model_kwargs = {'time_point': 0.0}  # default value of 0 days
        super().__init__(name, parent, model_kwargs=model_kwargs)

        # Add type attribute
        self.class_type = 'SeismicEvaluation'

        # Add criteria attribute
        self.criteria = criteria

        # Process time_points argument
        self.time_points = process_time_points(
            time_points, data_directory=None,
            component_class=self.class_type, name=self.name)

        # Add accumulator that will keep track of the total time
        # to the first detection
        self.add_accumulator('leak_detected', sim=0)
        self.add_accumulator('detection_time', sim=np.inf)

        # Add default threshold parameter
        self.add_default_par('threshold', value=10.0)

        # TODO System model does not yet know how to handle those
        self.run_time_indices = get_indices(self._parent.time_array,
                                            self.time_points*365.25)

    def reset(self):
        """
        Reset values of accumulators to their default values.

        """
        self.accumulators['leak_detected'].sim = 0
        self.accumulators['detection_time'].sim = np.inf

    def check_input_parameters(self, p):
        """
        Check whether input parameters fall within specified boundaries.

        :param p: input parameters of component model
        :type p: dict
        """
        # Do nothing for now
        pass

    def process_data(self, p, time_point=None, metric=None):
        """
        Compare metric(s) against the threshold according to the criteria.

        Parameters
        ----------
        p : dict
            Parameters of component
        time_point :
        metric : float
            Data metric to be compared to a threshold.

        Returns
        -------
        Dictionary of outputs.

        """
        # Obtain the default values of the parameters from dictionary of default parameters
        actual_p = {k: v.value for k, v in self.default_pars.items()}
        # Update default values of parameters with the provided ones
        actual_p.update(p)

        # TODO Include check for criteria if need arises in the future

        # Compare metric against threshold
        # Set defaults
        leak_detected_ts = 0
        detection_time_ts = np.inf

        if metric >= actual_p['threshold']:
            leak_detected_ts = 1
            detection_time_ts = time_point

        out = {'leak_detected_ts': leak_detected_ts,
               'detection_time_ts': detection_time_ts}

        if leak_detected_ts:  # leak detected at the current time point
            if self.accumulators['leak_detected'].sim:  # leak was detected previously
                out['detection_time'] = self.accumulators['detection_time'].sim
            else:  # leak was not detected previously
                out['detection_time'] = time_point
                self.accumulators['detection_time'].sim = time_point
                self.accumulators['leak_detected'].sim = 1
            out['leak_detected'] = 1
        else:  # leak is not detected at the current time point
            if self.accumulators['leak_detected'].sim:  # leak was detected previously
                out['detection_time'] = self.accumulators['detection_time'].sim
                out['leak_detected'] = 1
            else:  # leak was not detected previously
                out['detection_time'] = np.inf
                out['leak_detected'] = 0
                self.accumulators['detection_time'].sim = np.inf
                self.accumulators['leak_detected'].sim = 0

        return out


def test_scenario_seismic():
    # Define keyword arguments of the system model
    final_year = 90
    num_intervals = (final_year-10)//10
    time_array = 365.25*np.linspace(10.0, final_year, num=num_intervals+1)
    sm_model_kwargs = {'time_array': time_array}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'seismic'
    data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
    output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                    'test_seismic_evaluation')
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
                             data_reader_kwargs=data_reader_kwargs,
                             presetup=True))
    # Add parameters of the container
    dc.add_par('index', value=1, vary=False)
    # Add gridded observation
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    # ------------- Add seismic monitoring technology -------------
    smt = sm.add_component_model_object(
        SeismicMonitoring(name='smt', parent=sm, survey_config=survey_config,
                          time_points=time_points))
    # Add keyword arguments linked to the seismic data container outputs
    smt.add_kwarg_linked_to_obs('data', dc.linkobs['seismic'], obs_type='grid')
    smt.add_kwarg_linked_to_obs(
        'baseline', dc.linkobs['baseline_seismic'], obs_type='grid')
    # Add gridded observation
    smt.add_grid_obs('NRMS', constr_type='matrix', output_dir=output_directory)
    # Add scalar observations
    for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
        smt.add_obs(nm)
        smt.add_obs_to_be_linked(nm)

    # ------------- Add seismic evaluation component -------------
    seval = sm.add_component_model_object(
        SeismicEvaluation(name='seval', parent=sm, time_points=time_points))
    # Add threshold parameter
    threshold = 80.0
    seval.add_par('threshold', value=threshold, vary=False)
    # Add keyword arguments linked to the seismic monitoring metric
    seval.add_kwarg_linked_to_obs('metric', smt.linkobs['ave_NRMS'])
    # Add observations
    for nm in ['leak_detected', 'detection_time']:
        seval.add_obs(nm)        # covers the whole simulation period
        seval.add_obs(nm + '_ts')  # covers how values change in time

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

    evaluations = {}
    for nm in ['leak_detected', 'detection_time']:
        evaluations[nm] = sm.collect_observations_as_time_series(seval, nm)
        evaluations[nm + '_ts'] = sm.collect_observations_as_time_series(seval, nm + '_ts')

    # Plot results
    plt.close('all')
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
    fig.tight_layout()

    # Plot NRMS as image
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax_im = ax.imshow(nrms[0], aspect='auto')
    # Set title
    ax.set_title('NRMS at {} years'.format((time_ind+1)*10))
    # Set x-labels
    x = np.linspace(0, 100, num=11)
    xlabels = np.linspace(1, 101, num=11, dtype=int)
    ax.set_xticks(x, labels=xlabels)
    ax.set_xlabel('Receivers')
    # Set y-labels
    y = np.linspace(0, 8, num=9)
    ylabels = np.linspace(1, 9, num=9, dtype=int)
    ax.set_yticks(y, labels=ylabels)
    ax.set_ylabel('Sources')
    # Add colorbar
    plt.colorbar(ax_im, label='Percentage, %')
    fig.tight_layout()

    # Plot NRMS histogram
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(nrms[0], bins=25)
    ax.set_xlabel('NRMS values')
    ax.set_ylabel('Counts')
    ax.set_title('t = {} years'.format((time_ind+1)*10))
    fig.tight_layout()

    # Plot time changes in NRMS metrics
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    for ind, (nm, obs_label) in enumerate(zip(['ave_NRMS', 'min_NRMS', 'max_NRMS'],
                                              ['Average', 'Minimum', 'Maximum'])):
        axs[ind].plot(time_points, metrics[nm], '-ok')
        axs[ind].set_xlabel('Time, [years]')
        axs[ind].set_ylabel(obs_label + ' NRMS')
    axs[0].plot([time_points[0], time_points[-1]], 2*[threshold], '--r')
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
    final_obs_ind = final_year//10 - 1
    leak_detected = sm.obs['seval.leak_detected_{}'.format(final_obs_ind)].sim
    print('Is leak detected?', bool(leak_detected))
    if leak_detected:
        detection_time = sm.obs['seval.detection_time_{}'.format(final_obs_ind)].sim
        print('Leak is detected for the first time at t = {} days ({} years)'.format(
            detection_time, detection_time/365.25))


if __name__ == "__main__":

    test_scenario_seismic()
