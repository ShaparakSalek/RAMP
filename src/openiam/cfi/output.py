"""
Processing of system model output depending on the type of analysis chosen by user.
Last modified: September 16th, 2022

Contributors: Seth King, Veronika Vasylkivska, Nate Mitchell, Paul Holcomb
"""
import os
import pickle
import logging
import pandas as pd
import numpy as np


def extract_data(s, output_list, results, time_array, data_flip):
    """
    Create data frames from extracted sampleset data and return them with
    corresponding file names.

    Parameters
    ----------
    s : SampleSet
        Sampleset containing results of simulations for LHS and parstudy
    output_list : dict
        Dictionary with keys being a component class instance and values being the
        outputs added by user in the control file example setup
    results : numpy.array
        array containing all outputs of the simulations both for deterministic
        and stochastic runs
    time_array : array-like
        Array-like of time points at which the outputs were simulated
    data_flip : boolean
        Flag variable indicating whether the outputs will be saved in column-wise
        or row-wise format. # False - row-wise; True - column-wise

    Returns
    -------
    file_names : list
        List of filenames to save the outputs
    outputs : dict
        Dictionary with keys 'parameters' and 'observations' and values being
        a data frame and a list of data frames.

    """
    # Number of realizations
    num_reals = len(s.indices)
    real_labels = ['Time (days)'] + \
        ["Realization {}".format(ind) for ind in range(1, num_reals + 1)]

    # Number of time points
    if time_array is not None:
        num_time_points = len(time_array)
    else:
        time_array = [0]
        num_time_points = 1

    # Get names of parameters
    par_names = s.parnames

    # For comparison with another way to construct observation names
    sm_obs_names = s._parent.obs_base_names

    # Get observations and number of them
    output_names = []
    for output_component in list(output_list.keys()):
        for output_name in output_list[output_component]:
            output_names.append('.'.join([output_component.name, output_name]))
    num_obs = len(output_names)

    # Run checks of observations
    debug_msg = ''
    if len(sm_obs_names) != num_obs:
        debug_msg += 'Method "extract_data": Length of observation arrays are not equal.\n'
    else:
        for ind in range(num_obs):
            if output_names[ind] != sm_obs_names[ind]:
                debug_msg += 'Method "extract_data": Different observations at index {}'.format(
                    ind + 1)
    if debug_msg != '':
        logging.debug(debug_msg)

    # Get parameter results array
    par_array = s.samples.values  # of size (num_reals, num_pars)

    # Observations with endings corresponding to different time points
    all_obs_names = s.obsnames
    results_array = results

    if num_obs == 0:
        time_array = np.tile(time_array, len(all_obs_names))
    else:
        time_array = np.tile(time_array, num_obs)
    # results_array is of size (num_reals+1, len(all_obs_names))
    results_array = np.insert(results_array, 0, time_array, 0)

    outputs = {'parameters': None, 'observations': []}
    if data_flip:
        # For all observations in one data frame
        all_output = pd.DataFrame(
            data=results_array.transpose(), columns=real_labels, index=all_obs_names)
        outputs['observations'].append(all_output)

        # For individual observations in one data frame
        for ind in range(num_obs):
            start_ind = ind * num_time_points
            end_ind = start_ind + num_time_points
            indiv_obs_array = pd.DataFrame(
                data=results_array.transpose()[start_ind:end_ind],
                columns=real_labels, index=all_obs_names[start_ind:end_ind])
            outputs['observations'].append(indiv_obs_array)

        # For parameters in one data frame
        outputs['parameters'] = pd.DataFrame(
            data=par_array.transpose(), columns=real_labels[1:],
            index=par_names)
    else:
        # For all observations in one data frame
        all_output = pd.DataFrame(
            data=results_array, columns=all_obs_names, index=real_labels)
        outputs['observations'].append(all_output)

        # For individual observations in one data frame
        for ind in range(num_obs):
            start_ind = ind * num_time_points
            end_ind = start_ind + num_time_points
            indiv_obs_array = pd.DataFrame(
                data=results_array[:, start_ind:end_ind],
                columns=all_obs_names[start_ind:end_ind], index=real_labels)
            outputs['observations'].append(indiv_obs_array)

        # For parameters in one data frame
        outputs['parameters'] = pd.DataFrame(
            data=par_array, columns=par_names, index=real_labels[1:])

    file_names = ['output_realizations'] + output_names

    return file_names, outputs


def get_statistics(data, ax):
    """
    Calculate some statistics (min, max, mean, standard deviation and percentiles)
    for provided data.
    """
    stats = {1: data.min(axis=ax), 2: data.max(axis=ax), 3: data.mean(axis=ax),
             4: data.std(axis=ax), 5: data.var(axis=ax),
             6: np.percentile(data, 2.5, axis=ax),
             7: np.percentile(data, 5.0, axis=ax),
             8: np.percentile(data, 50.0, axis=ax),
             9: np.percentile(data, 95.0, axis=ax),
             10: np.percentile(data, 97.5, axis=ax)}

    return stats


def extract_stats(outputs, data_flip):
    """
    Create data frames oriented either column- or row-wise with statistics
    of the provided outputs.

    Parameters
    ----------
    outputs : dict
        Dictionary with keys 'parameters' and 'observations' and values being
        a data frame and a list of data frames.
    data_flip : boolean
        Flag variable indicating whether the outputs will be saved in column-wise
        or row-wise format. # False - row-wise; True - column-wise

    Returns
    -------
    all_stats : dict
        Dictionary with keys 'parameters' and 'observations' and values being
        data frames containing statistics regarding parameters and observations

    """
    stat_types = ['Time (days)', 'min', 'max', 'mean', 'stdev', 'variance',
                  '2.5%tile', '5.0%tile', '50.0%tile', '95.0%tile', '97.5%tile']
    all_obs = outputs['observations'][0]
    all_pars = outputs['parameters']

    # Initialize dictionary to be output
    all_stats = {}
    if data_flip:
        # Labels for rows:
        # observations names augmented with time points indices
        obs_rows_labels = list(all_obs.index)
        # Parameter names
        par_rows_labels = list(all_pars.index)

        # Convert data frame to numpy array
        all_obs = all_obs.to_numpy()
        all_pars = all_pars.to_numpy()

        # Axis along which the analysis is performed
        ax = 1

        # Array to keep results of analysis for observations
        obs_analysis_array = np.zeros((all_obs.shape[0], len(stat_types)))
        obs_analysis_array[:, 0] = all_obs[:, 0]  # time points

        # Array to keep results of analysis for parameters
        # Time points are not saved for parameters
        par_analysis_array = np.zeros((len(par_rows_labels), len(stat_types) - 1))

        # Get statistics for all observations (time points at column 0 are ignored)
        obs_stats = get_statistics(all_obs[:, 1:], ax)

        # Get statistics for all parameters
        par_stats = get_statistics(all_pars, ax)

        # Assign results to the right elements
        for ind in range(1, len(stat_types)):  # ind goes from 1 to 10
            obs_analysis_array[:, ind] = obs_stats[ind]
            par_analysis_array[:, ind - 1] = par_stats[ind]

        all_stats['observations'] = pd.DataFrame(
            data=obs_analysis_array, columns=stat_types, index=obs_rows_labels)
        all_stats['parameters'] = pd.DataFrame(
            data=par_analysis_array, columns=stat_types[1:], index=par_rows_labels)

    else:
        # Labels for columns:
        # observations names augmented with time points indices
        obs_cols_labels = list(all_obs.columns)
        # Parameter names
        par_cols_labels = list(all_pars.columns)

        # Convert data frame to numpy array
        all_obs = all_obs.to_numpy()
        all_pars = all_pars.to_numpy()

        # Axis along which the analysis is performed
        ax = 0

        # Array to keep results of analysis for observations
        obs_analysis_array = np.zeros((len(stat_types), all_obs.shape[1]))
        obs_analysis_array[0, :] = all_obs[0, :]  # time points

        # Array to keep results of analysis for parameters
        # Time points are not saved for parameters
        par_analysis_array = np.zeros((len(stat_types) - 1, len(par_cols_labels)))

        # Get statistics for all observations (time points at row 0 are ignored)
        obs_stats = get_statistics(all_obs[1:, :], ax)

        # Get statistics for all parameters
        par_stats = get_statistics(all_pars, ax)

        # Assign results to the right elements
        for ind in range(1, len(stat_types)):  # ind goes from 1 to 10
            obs_analysis_array[ind] = obs_stats[ind]
            par_analysis_array[ind - 1] = par_stats[ind]

        all_stats['observations'] = pd.DataFrame(
            data=obs_analysis_array, columns=obs_cols_labels, index=stat_types)
        all_stats['parameters'] = pd.DataFrame(
            data=par_analysis_array, columns=par_cols_labels, index=stat_types[1:])

    return all_stats


def process_output(yaml_data, model_data, output_list, out_dir, sm, s, analysis,
                   time_array, csv_files_dir):
    """
    Save simulation results as combined and/or individual data files, depending on input
    values.

    If 'GenerateOutputFiles' is in model_data and set to False, data files
    will not be saved. This option can be useful if a large number of
    locations are being evaluated. For example, a large number of locations are
    created when placing OpenWellbores with the range option and running the
    AoR_plot() function. This approach could result in the creation of hundreds
    of data files - one file for the output of each component at each location.
    Similarly, if 'GenerateCombOutputFile' in model_data is set to False,
    a combined data file will not be created. With a large number of locations
    and components, these combined data files can become quite large.
    """
    # Use results for csv files processing
    results = yaml_data['Results']

    # By default the data is saved as column-wise for consistency with previous formatting
    # applicable for control files with no specified preferences
    data_flip = model_data.get('OutputType', True)

    # This is set up so that you specifically have to enter
    # GenerateOutputFiles: False to prevent the creation of text files
    to_generate_output_files = model_data.get('GenerateOutputFiles', True)

    # This is set up so that you specifically have to enter 'GenerateCombOutputFile: False'
    # to prevent the creation of combined output file
    to_generate_comb_output = model_data.get('GenerateCombOutputFile', True)

    # This is set up so that you specifically have to enter 'GenerateStatFiles: False'
    # to prevent the creation of combined output file
    to_generate_stat_files = model_data.get('GenerateStatFiles', True)

    if not to_generate_output_files:
        to_generate_comb_output = False
        to_generate_stat_files = False

    if to_generate_output_files or to_generate_comb_output or to_generate_stat_files:

        if not os.path.exists(csv_files_dir):
            os.mkdir(csv_files_dir)

        if analysis == 'forward':

            # Initialize list of names of all added observations for all system model components
            output_labels = []
            # Initialize list of all observations
            all_obs = []

            # Loop over all components with added observations
            for output_component in list(output_list.keys()):
                for output_name in output_list[output_component]:
                    # Get observation name
                    curr_label = '.'.join([output_component.name, output_name])
                    # Add to the list of output labels
                    output_labels.append(curr_label)

                    if output_name in ['seal_flag', 'seal_time']:
                        # The observations are only returned at t=0
                        indices = [0]
                        # Repeat observation number of times equal to the number of time points
                        output_array = sm.collect_observations_as_time_series(
                            output_component, output_name, indices=indices)
                        output_values = len(time_array) * output_array.tolist()
                    else:
                        indices = None
                        output_array = sm.collect_observations_as_time_series(
                            output_component, output_name, indices=indices)
                        output_values = output_array

                    # # Get file name to save outputs: original code
                    # outfile = os.path.join(out_dir, curr_label)
                    # np.savetxt(outfile + '.txt', output_array, fmt='%1.12e')
                    output_values = np.expand_dims(output_values, 0)
                    # Add individual output to the list of all outputs
                    all_obs.append(output_values)

                    if to_generate_output_files:
                        # Prepare data for individual output saving
                        if data_flip:
                            curr_output = pd.DataFrame(
                                data=output_values.transpose(),
                                columns=[curr_label], index=time_array.transpose())
                        else:
                            curr_output = pd.DataFrame(
                                data=output_values, columns=time_array,
                                index=[curr_label])

                        # Save individual output
                        curr_output.to_csv(
                            os.path.join(out_dir, 'csv_files', curr_label) + '.csv',
                            index_label="Time (days)", float_format='%1.12e')

            if to_generate_comb_output:
                if data_flip:
                    all_output = pd.DataFrame(
                        data=np.squeeze(np.array(all_obs)).transpose(),
                        columns=output_labels, index=time_array)
                else:
                    all_output = pd.DataFrame(
                        data=np.squeeze(np.array(all_obs)),
                        columns=time_array, index=output_labels)

                all_output.to_csv(
                    os.path.join(out_dir, 'csv_files', 'simulation_results') + ".csv",
                    index_label="Time (days)", float_format='%1.12e')

        elif analysis in ['lhs', 'parstudy']:
            # # Original capability
            # outfile = os.path.join(out_dir, '{analysis}_results.txt'.format(
            #     analysis=analysis))
            # s.savetxt(outfile)

            # # Original capability
            # analysisfile = os.path.join(
            #     out_dir, '{analysis}_statistics.txt'.format(analysis=analysis))
            # s.savestats(analysisfile)

            # Labels for new saved files
            obs_index_label_reals = {0: 'Realization', 1: 'Observation'}
            obs_index_label_stats = {0: 'Statistic', 1: 'Observation'}
            par_index_label_reals = {0: 'Realization', 1: 'Parameter'}
            par_index_label_stats = {0: 'Statistic', 1: 'Parameter'}

            # Get data and file names
            file_names, outputs = extract_data(
                s, output_list, results, time_array, data_flip)

            # Save observation data
            if to_generate_comb_output:
                outputs['observations'][0].to_csv(
                    os.path.join(out_dir, 'csv_files', file_names[0]) + '.csv',
                    index_label=obs_index_label_reals[data_flip],
                    float_format='%1.12e')

                # Save parameter data
                outputs['parameters'].to_csv(
                    os.path.join(out_dir, 'csv_files', 'parameter_realizations.csv'),
                    float_format='%1.12e', index_label=par_index_label_reals[data_flip])

            if to_generate_output_files:
                for fname, df in zip(file_names[1:], outputs['observations'][1:]):
                    df.to_csv(os.path.join(out_dir, 'csv_files', fname) + '.csv',
                              index_label=obs_index_label_reals[data_flip],
                              float_format='%1.12e')

            if to_generate_stat_files:
                all_stats = extract_stats(outputs, data_flip)
                all_stats['observations'].to_csv(
                    os.path.join(out_dir, 'csv_files', 'output_stats.csv'),
                    index_label=obs_index_label_stats[data_flip], float_format='%1.12e')
                all_stats['parameters'].to_csv(
                    os.path.join(out_dir, 'csv_files', 'parameter_stats.csv'),
                    index_label=par_index_label_stats[data_flip], float_format='%1.12e')
        else:
            pass

        if time_array is not None:
            timefile = os.path.join(out_dir, 'csv_files', 'time_series.csv')
            np.savetxt(timefile, time_array / 365.25, fmt='%.12e',
                   delimiter=',', header='Time (years)')

    try:
        # No need to save pickled output for script runs
        not_yaml = yaml_data['not_yaml']
    except KeyError:
        if to_generate_comb_output:
            outfile = os.path.join(out_dir, 'combined_output.pkl')
            with open(outfile, 'wb') as output_dump:
                pickle.dump(yaml_data, output_dump)
