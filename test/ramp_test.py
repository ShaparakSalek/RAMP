import os
import sys
import unittest
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

try:
    from openiam import SystemModel
    from ramp import (DataContainer, PlumeEstimate,
                      SeismicSurveyConfiguration, SeismicDataContainer,
                      SeismicMonitoring, SeismicEvaluation)
    from ramp.utilities.data_readers import default_bin_file_reader
except ModuleNotFoundError:
    try:
        sys.path.append(os.sep.join(['..', 'src']))
        from openiam import SystemModel
        from ramp import (DataContainer, PlumeEstimate,
                          SeismicSurveyConfiguration, SeismicDataContainer,
                          SeismicMonitoring, SeismicEvaluation)
        from ramp.utilities.data_readers import default_bin_file_reader
    except ImportError as err:
        print('Unable to load RAMP class modules: {}'.format(err))

CURRENT_WORK_DIR = os.getcwd()


def create_system_model_with_data_container(time_points, output_directory):
    # Keyword arguments of the system model
    sm_model_kwargs = {'time_array': 365.25*time_points}   # time is given in days

    # Setup required information for data container before creating one
    obs_name = 'velocity'
    data_directory = os.sep.join(['data', 'velocity'])

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (141, 401),
                          'move_axis_destination': [-1, -2]}
    num_time_points = len(time_points)
    num_scenarios = 2
    family = 'velocity'
    data_setup = {}
    for ind in range(1, num_scenarios+1):
        data_setup[ind] = {'folder': 'sim{:04}'.format(ind)}
        for t_ind in range(1, num_time_points+1):
            data_setup[ind]['t{}'.format(t_ind)] = 't{}.bin'.format(t_ind*10)
    baseline = True

    #  Create system model
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # Add data container
    dc = sm.add_component_model_object(
        DataContainer(name='dc', parent=sm, family=family, obs_name=obs_name,
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

    return sm, dc


def create_system_model_with_seismic_data_container(time_points, output_directory):
    # Keyword arguments of the system model
    sm_model_kwargs = {'time_array': 365.25*time_points}   # time is given in days

    # Setup required information for seismic data container before creating one
    obs_name = 'seismic'
    data_directory = os.sep.join(['data', 'seismic'])

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    data_reader = default_bin_file_reader
    data_reader_kwargs = {'data_shape': (1251, 101, 9),
                          'move_axis_destination': [-1, -2, -3]}
    num_time_points = len(time_points)
    num_scenarios = 2
    family = 'seismic'
    data_setup = {}
    for ind in range(1, num_scenarios+1):
        data_setup[ind] = {'folder': 'sim0001'.format(ind)} # repeat used scenarios
        for t_ind in range(1, num_time_points+1):
            data_setup[ind]['t{}'.format(t_ind)] = 't{}.bin'.format(t_ind*10)
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

    # Create system model
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # Add seismic data container
    dc = sm.add_component_model_object(
        SeismicDataContainer(name='dc', parent=sm, survey_config=survey_config,
                             total_duration=2.5,
                             sampling_interval=0.002,
                             family=family, obs_name=obs_name,
                             data_directory=data_directory, data_setup=data_setup,
                             time_points=time_points, baseline=baseline,
                             data_reader=data_reader,
                             data_reader_kwargs=data_reader_kwargs,
                             presetup=True)) # presetup creates obs to be linked within constructor
    # Add parameters of the container
    dc.add_par('index', value=1, vary=False)
    # Add gridded observations
    dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
    dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
                    output_dir=output_directory)

    return sm, dc, survey_config


class Tests(unittest.TestCase):

    def setUp(self):
        """Defines the actions performed before each test."""
        # return to original directory
        os.chdir(CURRENT_WORK_DIR)

    def shortDescription(self):
        """Defines information that is printed about each of the tests.

        Method redefines method of TestCase class. In the original method
        only the first line of docstring is printed. In this method
        the full docstring of each test in Tests class is printed.
        """
        doc = self._testMethodDoc
        return doc

    def test_data_container(self):
        """Tests data container.

        Tests a data container component in a forward model against
        expected output for 3 different time points of data.
        """
        final_year = 30
        num_intervals = (final_year-10)//10
        time_points = np.linspace(10.0, final_year, num=num_intervals+1)
        output_directory = os.sep.join(['output', 'test_data_container'])

        # Create system model with data container for velocity data
        sm, dc = create_system_model_with_data_container(
            time_points, output_directory)

        # Run forward simulation
        sm.forward()

        # Get saved data from files at all time points
        vel_data = sm.collect_gridded_observations_as_time_series(
            dc, 'velocity', output_directory, rlzn_number=0)
        dv_data = sm.collect_gridded_observations_as_time_series(
            dc, 'delta_velocity', output_directory, rlzn_number=0)

        # Data shape
        true_data_shape = (3, 401, 141)

        # Sum of values for different time points
        true_sum_vel = [135598680.26904297, 134947024.04333496, 133979135.81420898]
        sum_vel = [np.sum(vel_data[ind]) for ind in range(num_intervals+1)]

        true_sum_dv = [0.0, -651656.2257080078, -1619544.4548339844]
        sum_dv = [np.sum(dv_data[ind]) for ind in range(num_intervals+1)]

        # Data at two time points and selected locations
        ind_to_check1 = (np.array([15, 194, 195, 195, 195], dtype=int),
                         np.array([3, 4, 3, 4, 5], dtype=int))
        true_dv_data1 = np.array([0.18994141, 0.19311523, 0.19140625, 0.21533203, 0.19018555])
        dv_data1 = dv_data[1][ind_to_check1]

        ind_to_check2 = (np.array([65, 65, 65, 66, 66, 66, 67, 68], dtype=int),
                         np.array([22, 23, 24, 22, 23, 24, 23, 23], dtype=int))
        true_dv_data2 = np.array([0.03051758, 0.03833008, 0.02978516, 0.02734375,
                                  0.03417969, 0.02661133, 0.02978516, 0.02563477])
        dv_data2 = dv_data[2][ind_to_check2]

        # Check data shape
        self.assertTrue(vel_data.shape==true_data_shape,
                        'Shape of returned data is {} but should be {}'.format(
                            vel_data.shape, true_data_shape))

        self.assertTrue(dv_data.shape==true_data_shape,
                        'Shape of returned data is {} but should be {}'.format(
                            vel_data.shape, true_data_shape))

        # Check sum of values for velocity and delta velocity
        for true_val, val, tp in zip(true_sum_vel, sum_vel, time_points):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Sum of values at time t={} years is {} but should be {}.'
                            .format(str(tp), str(val), str(true_val)))

        for true_val, val, tp in zip(true_sum_dv, sum_dv, time_points):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Sum of values at time t={} years is {} but should be {}.'
                            .format(str(tp), str(val), str(true_val)))

        # Check selected values
        for true_val, val in zip(true_dv_data1, dv_data1):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Data value at a selected location is {} but should be {}.'
                            .format(str(val), str(true_val)))

        for true_val, val in zip(true_dv_data2, dv_data2):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Data value at a selected location is {} but should be {}.'
                            .format(str(val), str(true_val)))

    def test_plume_estimate(self):
        """Tests plume estimate component linked to a data container.

        Tests a plume estimate component in a forward model against
        expected output for 3 different time points of data.
        """
        final_year = 30
        num_intervals = (final_year-10)//10
        time_points = np.linspace(10.0, final_year, num=num_intervals+1)
        output_directory = os.sep.join(['output', 'test_plume_estimate'])

        # Create system model with data container for velocity data
        sm, dc = create_system_model_with_data_container(
            time_points, output_directory)

        # Setup for plume estimate component
        coordinates = {1: 4000 + 10*np.arange(401),
                       2: 10*np.arange(141)}  # 1 is x, 2 is z
        criteria = 3 # criteria 3 - (data-baseline)/baseline is compared to a threshold
        plest = sm.add_component_model_object(
            PlumeEstimate(name='plest', parent=sm,
                          coordinates=coordinates, size=100,
                          criteria=criteria, max_num_plumes=1))
        # Add keyword arguments linked to the data container outputs
        threshold = 5.0
        plest.add_kwarg_linked_to_obs(
            'data', dc.linkobs['velocity'], obs_type='grid')
        plest.add_kwarg_linked_to_obs(
            'baseline', dc.linkobs['baseline_velocity'], obs_type='grid')

        # Add threshold parameter
        plest.add_par('threshold', value=threshold, vary=False)

        # Add gridded observations
        plest.add_grid_obs('plume', constr_type='matrix', output_dir=output_directory)
        plest.add_grid_obs('plume_data', constr_type='matrix', output_dir=output_directory)
        # Add observations related to each dimension
        for nm in ['min1', 'min2', 'max1', 'max2', 'extent1', 'extent2', 'plume_size']:
            plest.add_grid_obs(nm, constr_type='array', output_dir=output_directory)

        # Add scalar observations
        plest.add_obs('num_plumes')

        # Run forward simulation
        sm.forward()

        # Collect gridded observations
        plume = sm.collect_gridded_observations_as_time_series(
            plest, 'plume', output_directory, rlzn_number=0)
        plume_data = sm.collect_gridded_observations_as_time_series(
            plest, 'plume_data', output_directory, rlzn_number=0)

        # Collect array-type observations
        plume_metrics = {}
        for obs in ['min1', 'min2', 'max1', 'max2', 'extent1', 'extent2']:
            plume_metrics[obs] = sm.collect_gridded_observations_as_time_series(
                plest, obs, output_directory, rlzn_number=0)

        # Collect scalar observation
        plume_metrics['num_plumes'] = sm.collect_observations_as_time_series(
            plest, 'num_plumes')

        true_sum_plume = [0.0, 1508.0, 3215.0]
        sum_plume = [np.sum(plume[ind]) for ind in range(num_intervals+1)]

        true_sum_plume_data = [0.0, 28133.52205760463, 70831.30440246908]
        sum_plume_data = [np.sum(plume_data[ind]) for ind in range(num_intervals+1)]

        true_metrics = {
            'min1': [-999., 4960., 4670.],
            'min2': [-999., 50., 10.],
            'max1': [-999., 5180., 5610.],
            'max2': [-999., 850., 850.],
            'extent1': [0., 220., 940.],
            'extent2': [0., 800., 840.],
            'num_plumes': [0., 1.0, 1.0]}

        # Check sum of values for plume and plume_data observations
        for true_val, val, tp in zip(true_sum_plume, sum_plume, time_points):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Sum of values at time t={} years is {} but should be {}.'
                            .format(str(tp), str(val), str(true_val)))

        for true_val, val, tp in zip(true_sum_plume_data, sum_plume_data, time_points):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Sum of values at time t={} years is {} but should be {}.'
                            .format(str(tp), str(val), str(true_val)))

        # Check metrics of plumes
        for metric_nm, true_values in true_metrics.items():
            for true_val, val, tp in zip(true_values, plume_metrics[metric_nm], time_points):
                self.assertTrue(true_val==val,
                                'Value of {} at time t={} years is {} but should be {}.'
                                .format(metric_nm, str(tp), str(val), str(true_val)))

    def test_seismic_data_container(self):
        """Tests seismic data container as a standalone component.

        Tests a seismic data container component in a forward model against
        expected output for 3 different time points of data.
        """
        final_year = 30
        num_intervals = (final_year-10)//10
        time_points = np.linspace(10.0, final_year, num=num_intervals+1)
        output_directory = os.sep.join(['output', 'test_seismic_data_container'])

        # Create system model with data container for seismic data
        sm, dc, _ = create_system_model_with_seismic_data_container(
            time_points, output_directory)

        # Run forward simulation
        sm.forward()

        # Collect gridded observations
        seismic_data = sm.collect_gridded_observations_as_time_series(
            dc, 'seismic', output_directory, rlzn_number=0)
        dseismic_data = sm.collect_gridded_observations_as_time_series(
            dc, 'delta_seismic', output_directory, rlzn_number=0)

        # Data shape
        true_data_shape = (3, 9, 101, 1251)

        # Sum of values for different time points
        true_sum_sd = [52.61768409921891, 88.42940828856192, 96.09528723575019]
        sum_sd = [np.sum(seismic_data[ind]) for ind in range(num_intervals+1)]

        true_sum_dsd = [0.0, 35.811725323487714, 43.47758869211108]
        sum_dsd = [np.sum(dseismic_data[ind]) for ind in range(num_intervals+1)]

        # Data at two time points and selected locations
        ind_to_check1 = (np.array([0, 0, 4, 8, 8], dtype=int),
                         np.array([6,  6,  51,  96,  96], dtype=int),
                         np.array([54, 55, 54, 54, 61], dtype=int))
        true_seismic_data1 = np.array([
            42.97988129, 46.06204224, 42.98212814, 42.98988342, 43.18157196])
        seismic_data1 = seismic_data[1][ind_to_check1]

        ind_to_check2 = (np.array([1, 1, 1, 1, 3, 3, 3, 3], dtype=int),
                         np.array([22, 23, 23, 23, 31, 31, 32, 32], dtype=int),
                         np.array([119, 129, 130, 131, 160, 161, 148, 149], dtype=int))
        true_dseismic_data2 = np.array([
            12.00019836, 12.45207405, 12.87016773, 13.01250458,
            12.01769733, 12.03115845, 12.05310059, 12.0701704])
        dseismic_data2 = dseismic_data[2][ind_to_check2]

        # Check data shape
        self.assertTrue(seismic_data.shape==true_data_shape,
                        'Shape of returned data is {} but should be {}'.format(
                            seismic_data.shape, true_data_shape))

        self.assertTrue(dseismic_data.shape==true_data_shape,
                        'Shape of returned data is {} but should be {}'.format(
                            dseismic_data.shape, true_data_shape))

        # Check sum of values for seismic and delta seismic
        for true_val, val, tp in zip(true_sum_sd, sum_sd, time_points):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Sum of values at time t={} years is {} but should be {}.'
                            .format(str(tp), str(val), str(true_val)))

        for true_val, val, tp in zip(true_sum_dsd, sum_dsd, time_points):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Sum of values at time t={} years is {} but should be {}.'
                            .format(str(tp), str(val), str(true_val)))

        # Check selected values
        for true_val, val in zip(true_seismic_data1, seismic_data1):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Data value at a selected location is {} but should be {}.'
                            .format(str(val), str(true_val)))

        for true_val, val in zip(true_dseismic_data2, dseismic_data2):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Data value at a selected location is {} but should be {}.'
                            .format(str(val), str(true_val)))


    def test_seismic_monitoring(self):
        """Tests seismic monitoring component linked to a seismic data container.

        Tests a seismic monitoring component in a forward model against
        expected output for 3 different time points of data.
        """
        final_year = 30
        num_intervals = (final_year-10)//10
        time_points = np.linspace(10.0, final_year, num=num_intervals+1)
        output_directory = os.sep.join(['output', 'test_seismic_monitoring'])

        # Create system model with data container for seismic data
        sm, dc, survey_config = create_system_model_with_seismic_data_container(
            time_points, output_directory)

        # Add seismic monitoring technology
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

        # Run forward simulation
        sm.forward()

        # Export gridded observation
        nrms = sm.collect_gridded_observations_as_time_series(
            smt, 'NRMS', output_directory, rlzn_number=0)

        # Export scalar observations
        metrics = {}
        for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
            metrics[nm] = sm.collect_observations_as_time_series(smt, nm)

        # Data shape
        true_data_shape = (3, 9, 101)

        # Sum of values for different time points
        true_sum_nrms = [0.0, 25420.28990545869, 72030.09895187244]
        sum_nrms = [np.sum(nrms[ind]) for ind in range(num_intervals+1)]

        true_metrics = {
            'ave_NRMS': [0., 27.965117, 79.24103],
            'max_NRMS': [0., 114.57167, 172.38902],
            'min_NRMS': [0., 0.07454574, 0.03501776]}

        # Check data shape
        self.assertTrue(nrms.shape==true_data_shape,
                        'Shape of returned data is {} but should be {}'.format(
                            nrms.shape, true_data_shape))

        # Check sum of values for NRMS data
        for true_val, val, tp in zip(true_sum_nrms, sum_nrms, time_points):
            self.assertTrue(abs(true_val-val) < 1.0e-6,
                            'Sum of values at time t={} years is {} but should be {}.'
                            .format(str(tp), str(val), str(true_val)))

        # Check NRMS metrics
        for metric_nm, true_values in true_metrics.items():
            for true_val, val, tp in zip(true_values, metrics[metric_nm], time_points):
                self.assertTrue(abs(true_val-val) < 1.0e-4,
                                'Value of {} at time t={} years is {} but should be {}.'
                                .format(metric_nm, str(tp), str(val), str(true_val)))

    def test_seismic_evaluation(self):
        """Tests seismic evaluation component linked to a seismic monitoring
        component linked to a seismic data container.

        Tests a seismic evaluation component in a forward model against
        expected output for 3 different time points of data.
        """
        final_year = 30
        num_intervals = (final_year-10)//10
        time_points = np.linspace(10.0, final_year, num=num_intervals+1)
        output_directory = os.sep.join(['output', 'test_seismic_evaluation'])

        # Create system model with data container for seismic data
        sm, dc, survey_config = create_system_model_with_seismic_data_container(
            time_points, output_directory)

        # Add seismic monitoring technology component
        smt = sm.add_component_model_object(
            SeismicMonitoring(name='smt', parent=sm, survey_config=survey_config,
                              time_points=time_points))
        # Add keyword arguments linked to the data container outputs
        smt.add_kwarg_linked_to_obs('data', dc.linkobs['seismic'], obs_type='grid')
        smt.add_kwarg_linked_to_obs(
            'baseline', dc.linkobs['baseline_seismic'], obs_type='grid')
        # Add scalar observations to be linked
        for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
            smt.add_obs_to_be_linked(nm)

        # Add seismic evaluation component
        seval = sm.add_component_model_object(
            SeismicEvaluation(name='seval', parent=sm, time_points=time_points))
        # Add threshold parameter
        threshold = 80.0
        seval.add_par('threshold', value=threshold, vary=False)
        # Add keyword arguments linked to the seismic monitoring component metric
        seval.add_kwarg_linked_to_obs('metric', smt.linkobs['max_NRMS'])
        # Add observations
        for nm in ['leak_detected', 'detection_time']:
            seval.add_obs(nm)          # covers the whole simulation period
            seval.add_obs(nm + '_ts')  # covers how and whether values change in time

        # Run forward simulation
        sm.forward()

        # Export scalar observations
        metrics = {}
        for nm in ['leak_detected', 'detection_time']:
            metrics[nm] = sm.collect_observations_as_time_series(seval, nm)
            metrics[nm + '_ts'] = sm.collect_observations_as_time_series(seval, nm + '_ts')

        true_metrics = {
            'leak_detected': [0, 1, 1],
            'leak_detected_ts': [0, 1, 1],
            'detection_time': [np.inf, 7305., 7305.],
            'detection_time_ts': [np.inf, 7305., 10957.5]}

        # Check leak detection metrics
        for nm in ['leak_detected', 'leak_detected_ts']:
            for true_val, val, tp in zip(true_metrics[nm], metrics[nm], time_points):
                self.assertTrue(true_val==val,
                                'Value of {} at time t={} years is {} but should be {}.'
                                .format(nm, str(tp), str(val), str(true_val)))
        # Check detection times
        for nm in ['detection_time', 'detection_time_ts']:
            for true_val, val, tp in zip(true_metrics[nm], metrics[nm], time_points):
                self.assertTrue(true_val==val,
                                'Value of {} at time t={} years is {} days but should be {}.'
                                .format(nm, str(tp), str(val), str(true_val)))


BASE_TESTS = [
    'test_data_container',
    'test_plume_estimate',
    'test_seismic_data_container',
    'test_seismic_monitoring',
    'test_seismic_evaluation'
    ]

CONTAINER_TESTS = [
    'test_data_container',
    'test_seismic_data_container'
    ]


def suite(case):
    """Determines set of tests to be performed."""

    suite = unittest.TestSuite()

    if case == 'all':
        for test_nm in BASE_TESTS:
            suite.addTest(Tests(test_nm))
        return suite
    if case == 'base':
        for test_nm in BASE_TESTS:
            suite.addTest(Tests(test_nm))
        return suite
    if case == 'container':
        for test_nm in CONTAINER_TESTS:
            suite.addTest(Tests(test_nm))
        return suite

    return suite

if __name__ == '__main__':
    # For multiprocessing in Spyder
    __spec__ = None

    if len(sys.argv) > 1:
        case = sys.argv[1]
    else:
        case = 'base'

    # Second keyword argument makes testing messages printed
    # when running test suite in IPython console
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stderr)
    test_suite = suite(case)
    result = runner.run(test_suite)
    if not result.wasSuccessful():
        sys.exit(1)
