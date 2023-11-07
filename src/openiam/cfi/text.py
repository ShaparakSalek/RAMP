"""
Processing of system model and component model objects into text output
depending on the type of analysis chosen by user.
Last modified: November 4th, 2022

Author: Paul Holcomb
"""
import os
import numpy as np
import sys
import collections
import matk
import re

import warnings
# Suppress FutureWarning on empty string check
warnings.simplefilter(action='ignore', category=FutureWarning)

# Set keys to filter out of system model parameters
SYSTEM_PAR_IGNORE = ['user_value', 'init_value', 'deps', 'stderr', 'correl',
                     'mean', 'std', '_nominal']

# Set list of keys to ignore in component model parameters
COMPONENT_PAR_IGNORE = ['name', 'user_value', 'init_value', 'deps', 'stderr',
                        'correl', 'mean', 'std', '_nominal', '_parent']

# Set list of keys to ignore in component model attributes
COMPONENT_ATTRIBUTE_IGNORE = ['_parent', 'name', 'sol',
                              'x_scaler_p', 'y_scaler_p', 'model_p',
                              'x_scaler_s', 'y_scaler_s', 'model_s']


def system_model_to_text(sm, out_dir, analysis):
    """
    Method system_model_to_text decomposes, filters, and prints the system model object
    into a human-readable text file for developers.

    Inputs:
    :param sm: current system model
    :type sm: instance of SystemModel class

    :param out_dir: current output directory for simulation
    :type out_dir: str

    :param analysis: type of analysis performed
    :type analysis: str

    Output:
    Formatted text file named {analysis}_system_model_output.txt, where {analysis}
    is the current analysis type as defined in the 'analysis' input parameter.
    This file is created and stored in a folder named 'components_data'
    within the directory defined by the 'out_dir' input parameter.
    """
    # Get original default print output location
    original_stdout = sys.stdout

    # Set keys to filter system model information
    # Only keys in this list will be output
    temp_keys = ['_model', '_model_args', '_model_kwargs', '_workdir',
                 '_results_file', '_seed', 'sample_size', 'pars',
                 'discrete_pars', 'obs', 'sampleset', '_current',
                 'single_time_point_flag', 'time_array',
                 'component_models', 'interpolators', 'interp_creators',
                 'observation2components', 'obs_base_names',
                 'collectors']

    # Get values for each key in temp_keys
    temp_values = [sm.__dict__[v] for v in temp_keys if v in sm.__dict__.keys()]

    # Set empty string to filter parameter names
    last_parname = ''

    # If the components_data folder does not exist, create it
    if not os.path.exists(os.path.join(out_dir, 'components_data')):
        os.makedirs(os.path.join(out_dir, 'components_data'))

    # Create and open the system model output file in write mode
    with open(os.path.join(out_dir, 'components_data',
                           '{}_system_model_output.txt'.format(analysis)), 'w') as y_file:

        # Set the default print location to the currently open file
        sys.stdout = y_file

        # For each key, value pair in the filtered system model
        for k, v in zip(temp_keys, temp_values):
            if k == 'obs':  # if the current key is observations ('obs')
                # Print the location of the CSV files for the model observations output
                print("{}: {}".format(
                    k, os.path.join(out_dir, 'csv_files', 'simulation_results') + ".csv"))

            elif k == 'pars':  # if the current key is parameters ('pars')
                # If there are no parameters to report
                if len(v) == 0:
                    # Set the value to None, print, and go to the next key
                    v = None
                    print("{}: {}".format(k, v))

                else:
                    # Print the "pars:" header
                    print("{}: ".format(k))
                    # For each parameter
                    for m, n in zip(v.keys(), v.values()):
                        # Expand the dictionary of values
                        n = n.__dict__

                        # For every aspect of the parameter
                        for o, p in zip(n.keys(), n.values()):
                            # If it isn't in the SYSTEM_PAR_IGNORE list
                            if o not in SYSTEM_PAR_IGNORE:
                                if o == 'name':  # if the key is 'name'
                                    # Split the full name into name and type strings
                                    parname = p[:p.find('.')]
                                    partype = p[p.find('.') + 1:]

                                    if parname != last_parname:  # If the parameter name is new
                                        # Print the parameter name as the header
                                        # and then print the type
                                        print("\t{}:\n\t\t{}:".format(parname, partype))
                                    else:  # if the parameter name is the same as the last
                                        # Only print the new type
                                        print("\t\t{}:".format(partype))

                                    last_parname = parname

                                else:  # for all other attributes of the parameters
                                    # unless their keys are '_parent' or 'from_internal',
                                    if o not in ['_parent', 'from_internal']:
                                        # print the key, value pair
                                        print("\t\t\t{}: {}".format(o, p))

            elif k == 'component_models':  # if the current key is for component models ('component_models')
                # If there are no component models to report
                if len(v) == 0:
                    # Set the value equal to None and print
                    v = None
                    print("{}: {}".format(k, v))
                else:  # otherwise
                    # Print the "component_models:" header
                    print("{}: ".format(k))
                    # Go over each component model
                    for m in v.keys():
                        # print the name
                        print("\t{}".format(m))

            # If the current key is for the sample set ('sampleset')
            elif k == 'sampleset':
                # If there are no sample sets to report
                if len(v) == 0:
                    # Set the value equal to None and print
                    v = None
                    print("{}: {}".format(k, v))
                else:  # otherwise
                    # Print the "sampleset:" header,
                    print("{}:".format(k))
                    # Go each dictionary contained in the sample set
                    for m, n in zip(v.keys(), v.values()):
                        # Print the "name" of the dictionary
                        print("\t{}:".format(m))
                        # Expand the dictionary of values
                        n = n.__dict__
                        # For each key, value pair in the dictionary
                        for o, p in zip(n.keys(), n.values()):
                            # If the current key is '_parent' or '_obsnames', skip it
                            if o in ['_parent', "_obsnames"]:
                                pass

                            elif o == '_indices':  # if the current key is '_indices'
                                # Print indices described as a list with the first and last index
                                print("\t\t{}: list[{}-{}]".format(o, p[0], p[-1]))
                                # Determine the step size for indices
                                d_index = np.diff(p)[0]
                                # Print the index step size
                                print("\t\t_index_step: {}".format(d_index))

                            # If the current key is either 'samples' or 'responses'
                            elif o in ['samples', 'responses']:
                                # Print the header using the appropriate key
                                print("\t\t{}:".format(o))
                                # Expand the dictionary of values
                                p = p.__dict__
                                # For each key,value pair in the dictionary
                                for q, r in zip(p.keys(), p.values()):
                                    # If the key is '_values', '_mins', or '_maxs',
                                    if q in ["_values", "_mins", "_maxs"]:
                                        # Print the key and the size of the value array
                                        print("\t\t\t{}: Array{}".format(q, np.array(r).shape))

                                    elif q == "_names":  # if the key is '_names'
                                        if o == 'samples':  # if the parent key is 'samples'
                                            # Print the "_names:" header
                                            print("\t\t\t{}:".format(q))
                                            # Print each sample name in values
                                            for s in r:
                                                print("\t\t\t\t{}".format(s))

                                        else:  # if the parent key is 'responses'
                                            # Print the "responses:" header
                                            print("\t\t\t{}:".format(q))
                                            # Set empty strings to track the last
                                            # observation name and number
                                            last_obs_name = ''
                                            last_obs_number = ''

                                            # for each value
                                            for s in r:
                                                # Get the observation name
                                                # before the first underscore
                                                if '_' in s:
                                                    obs_name = s[:s.rfind('_')]
                                                    # Get the observation time index after the underscore
                                                    try:
                                                        obs_number = int(s[s.rfind('_') + 1:])
                                                    except:
                                                        obs_name = s
                                                        obs_number = ''
                                                else:
                                                    obs_name = s
                                                    obs_number = ''
                                                # If this is the first time through,
                                                # set the current name
                                                # to the last_obs_name variable
                                                if len(last_obs_name) == 0:
                                                    last_obs_name = obs_name

                                                # if this is the first time seeing this observation name
                                                if last_obs_name != obs_name:
                                                    # Create a string for the name of the observation,
                                                    # with a list from 0 to the last observation number
                                                    if obs_number != '':
                                                        obs_string = "{},[0-{}]".format(
                                                            last_obs_name, last_obs_number)
                                                    else:
                                                        obs_string = obs_name
                                                    # Set the current observation name
                                                    # as the last observation name
                                                    last_obs_name = obs_name
                                                    # print the observation string
                                                    print("\t\t\t\t{}".format(obs_string))

                                                # Set the current observation number
                                                # as the last observation number
                                                last_obs_number = obs_number

                            else:  # if the key is anything else
                                # Print the key, value pair
                                print("\t\t{}: {}".format(o, p))

            # If the current key is a time array ('time_array')
            elif k == 'time_array':
                if v is not None:
                    time_start, time_end, time_diff, timestep_check = check_time_steps(v)

                    if timestep_check:  # if the time step is regular
                        # Print the 'time_array:' header and the beginning and ending time step
                        print("{}: [{}-{}]".format(k, time_start, time_end))
                        # Print the time step
                        print("time_step: {}".format(time_diff[0]))

                    else:  # if the time step is non-uniform
                        # Print the time array list
                        print("{}: {}".format(k, v))

            # If the current key is the connections of observation
            # to components ('observation2components')
            elif k == 'observation2components':
                # Print the 'observation2components:' header
                print("{}:".format(k))
                # For each key,value pair in the observation2components value
                for m, n in zip(v.keys(), v.values()):
                    # Print the key, value pair
                    print("\t{}: {}".format(m, n[0]))

            # If the current key is observation base names ('obs_base_names')
            elif k == 'obs_base_names':
                if len(v) != 0:
                    # Print the 'obs_base_names:' header
                    print("{}:".format(k))
                    # For each value in the 'obs_base_names' list
                    for n in v:
                        # Print the value
                        print("\t{}".format(n))

            elif k == 'interpolators':  # if the current key is 'interpolators'
                if len(v) == 0:  # if there are no interpolators to report
                    # Print the interpolators header with a value of None
                    print("{}: None".format(k))

                else:  # otherwise
                    # Print the 'interpolators:' header
                    print("{}:".format(k))
                    # For each interpolator type
                    for m, n in zip(v.keys(), v.values()):
                        # Format 'names'
                        # Print interpolator type
                        print("\t{}:".format(m))

                        # Get names of all current interpolators
                        curr_interps = n.keys()

                        # Get the starting number for the interpolator names
                        interps_start = int(list(filter(str.isdigit, curr_interps[0]))[0])

                        # Get the number of interpolators
                        num_interps = len(curr_interps)

                        # Get the interpolator base name by stripping everything
                        # except alphabetical characters
                        interp_name = ''.join(list(filter(str.isalpha, curr_interps[0])))

                        # Combine interpolator name information and print
                        print('\t\tnames: {}[{}-{}]'.format(
                            interp_name, interps_start, num_interps))

                        # Extract dictionary from first interpolator
                        # to retrieve repetitive data
                        first_interp = v[m][curr_interps[0]].__dict__

                        # Print header file name
                        print("\t\theader_file_dir: {}".format(
                            first_interp['header_file_dir']))
                        # Print time file name
                        print("\t\ttime_file: {}".format(first_interp['time_file']))

                        # Format 'data_file'
                        # Get the file number of the first data file
                        data_file_start = list(filter(str.isdigit, first_interp['data_file']))[0]
                        # Get data file base name
                        data_file_name = re.sub(r"\d+", "", first_interp['data_file'])

                        # Combine base name, starting number, ending number,
                        # and file extension to create data_file_str
                        data_file_str = data_file_name[:-4] + '[' \
                            + data_file_start + '-' + str(num_interps) + ']'\
                                + '.' +data_file_name[-3:]

                        # Print data file name(s)
                        print("\t\tdata_files: {}".format(data_file_str))
                        # Print 'default_values'
                        print("\t\tdefault_values: {}".format(
                            first_interp['default_values']))

                        # Print dictionaries of 'default_units' and 'title_names'
                        # Create list of keys to loop over
                        unit_and_titles = ['default_units', 'title_names']

                        # For each key in the list
                        for u_and_t in unit_and_titles:
                            # Print the key
                            print("\t\t{}:".format(u_and_t))

                            # For each key,value pair in the dictionary specified by the key
                            for a, b in zip(first_interp[u_and_t].keys(), first_interp[u_and_t].values()):
                                # If the output is an empty string, set it to None
                                if b == '':
                                    b = None
                                # Print key, value pair
                                print("\t\t\t{}: {}".format(a, b))

                        # Print 'interp_2d'
                        print("\t\tinterp_2d: {}".format(first_interp['interp_2d']))
                        # Print 'created'
                        print("\t\tcreated: {}".format(first_interp['created']))

                        # Print 'time_points'
                        time_start, time_end, time_diff, timestep_check = check_time_steps(
                            first_interp['time_points'])

                        # If time steps are uniform
                        if timestep_check:
                            # Print time points range
                            print("\t\ttime_points: [{}-{}]".format(time_start, time_end))

                            # Print time step
                            print("\t\ttime_step: {}".format(time_diff[0]))

                        else:  # otherwise, if time steps are non-uniform
                            #Print all time points
                            print("\t\ttime_points: {}".format(
                                first_interp['time_points']))

                        # Get num_data_time_points
                        num_data_time_points = first_interp['num_data_time_points']

                        # Get names of data
                        data = first_interp['data'].keys()

                        # Format data_headers
                        # Get data_headers
                        data_headers = first_interp['data_headers']

                        # Instantiate empty list to hold data_headers strings
                        data_headers_list = []

                        # Extract data_headers based on names of data and format strings
                        for name in data:
                            # Create a list containing only those headers with the current name
                            temp_list = [m for m in data_headers if re.search(name, m)]

                            if not temp_list: # if the list is empty
                                continue
                            elif len(temp_list) == 1:  # if there is only one header name
                                # Set 'temp_name' to this name
                                temp_name = name

                            else:
                                # Get first number of header label
                                first_label_num = temp_list[0][temp_list[0].find('_')+1:]
                                # Get last number of header label
                                last_label_num = temp_list[-1][temp_list[-1].find('_')+1:]
                                # Get base name of header label
                                data_header_name = temp_list[0][:temp_list[0].find('_')+1]
                                # Create data header string
                                temp_name = data_header_name + '[' + first_label_num + '-' + last_label_num + ']'

                            # Append data header string to list
                            data_headers_list.append(temp_name)
                            # Remove current data headers from main list
                            data_headers = [n for n in data_headers if n not in temp_list]

                        # Combine the remaining data headers with the list of formatted data headers
                        data_headers_list = data_headers + data_headers_list
                        # Print 'data headers:' header
                        print('\t\tdata headers:')
                        # For each data header in the list
                        for each in data_headers_list:
                            # print the data header
                            print('\t\t\t{}'.format(each))

                        # Get points list size
                        points_y = len(first_interp['points'])
                        points_x = len(first_interp['points'][0])

                        # Print the number of data time points
                        print('\t\tnum_data_time_points: {}'.format(num_data_time_points))
                        # Print data names
                        print('\t\tdata:')

                        # For each item in data
                        for each in data:
                            # Print the item
                            print('\t\t\t{}'.format(each))

                        # Print points list size
                        print('\t\tpoints: list({}, {})'.format(points_y, points_x))
                        # Print num_xy_points
                        print('\t\tnum_xy_points: {}'.format(first_interp['num_xy_points']))

            # If the current key is the interpolator creators ('interp_creators')
            elif k == 'interp_creators':
                # If there are no interpolator creators to report
                if len(v) == 0:
                    # Print 'interp_creators:' as None
                    print("{}: None".format(k))

                else:
                    # Print the 'interp_creators:' header
                    print("{}:".format(k))
                    # For each key, value pair in 'interp_creators' value
                    for m, n in zip(v.keys(), v.values()):
                        # Print the key as header
                        print("\t{}:".format(m))
                        # Expand the values dictionary
                        n = n.__dict__
                        # For each key, value pair in the values dictionary
                        for o, p in zip(n.keys(), n.values()):
                            # If the key is the name ('name')
                            if o == 'name':
                                # print the name and associated value
                                print("\t\t{}: {}".format(o, p))

            # If the current key is collectors ('collectors')
            elif k == 'collectors':
                # If there are no collectors to report
                if len(v) == 0:
                    # print "collectors:" as None
                    print("{}: None".format(k))

                else:   # otherwise
                    # Print the "collectors:" header
                    print("{}:".format(k))
                    # For each key, value pair in the collectors value dictionary
                    for m, n in zip(v.keys(), v.values()):
                        # Print the key m as header
                        print("\t{}:".format(m))
                        # For each key, value pair in the dictionary n
                        for o, p in zip(n.keys(), n.values()):
                            # Print the key o as header
                            print("\t\t{}:".format(o))
                            # For each key, value pair in the dictionary p
                            for q, r in zip(p.keys(), p.values()):
                                # If the current key is connections ('Connection')
                                if q == 'Connection':
                                    # If there are no connections to report
                                    if len(r) == 0:
                                        # Print "Connection:" as None
                                        print("\t\t\t{}: None".format(q))

                                    # If there is exactly one connection to report
                                    elif len(r) == 1:
                                        # Print the "Connection:" header and single connection
                                        print("\t\t\t{}: {}".format(q, r[0]))

                                    else:
                                        # Print the "Connection:" header
                                        print("\t\t\t{}:".format(q))
                                        # For each connection in the values list
                                        for each in r:
                                            # print the connection
                                            print("\t\t\t\t{}".format(each))

                                elif q == "argument":  # If the current key is an argument ('argument')
                                    # Print the key, value pair
                                    print("\t\t\t{}: {}".format(q, r))

                                elif q == "data":  # If the current key is the data ('data')
                                    # If there is no data to report
                                    if len(r) == 0:
                                        # Print the "data:" header as None
                                        print("\t\t\t{}: None".format(q))

                                    else:  # otherwise
                                        # Print the "data:" header
                                        print("\t\t\t{}:".format(q))
                                        # For each item in the data value list
                                        for each in r:
                                            # expand the item dictionary
                                            each = each.__dict__
                                            # for each key, value pair in the item dictionary
                                            for s, t in zip(each.keys(), each.values()):
                                                # If the current key is the name ('_name')
                                                if s == '_name':
                                                    # Print the '_name:' header
                                                    print("\t\t\t\t{}:".format(t))

                                                else:  # otherwise
                                                    # print the current key, value pair
                                                    print("\t\t\t\t\t{}: {}".format(s, t))

            # If the current key does not satisfy the preceding criteria
            # and the value associated with it is a dictionary type
            elif isinstance(v, (dict, collections.OrderedDict,
                                matk.ordereddict.OrderedDict)):
                # Run the key, value pair through a recursive dictionary analysis function
                dict_recursive(k, v, 0)

            # If the current key does not satisfy the preceding criteria
            # and the value associated with it has a dictionary attribute
            elif hasattr(v, '__dict__'):
                # expand the dictionary of the value
                v = v.__dict__
                # Run the key, value pair through a recursive dictionary analysis function
                dict_recursive(k, v, 0)

            # If the key, value pair satisfies none of the preceding criteria
            else:
                # If the value is an empty string
                if v == '':
                    # Set the value to None
                    v = None
                # print the current key, value pair
                print("{}: {}".format(k, v))

    # Reassign the default print location to its original value
    sys.stdout = original_stdout


def component_models_to_text(models, out_dir, analysis):
    """
    Component_models_to_text decomposes, filters, and prints the component models object
    into a human-readable text file for developers.

    Inputs:
    :param models: current component models
    :type models: dictionary

    :param out_dir: current output directory for simulation
    :type out_dir: str

    :param analysis: type of analysis performed
    :type analysis: str

    Output:
    Formatted text files named {analysis}_{component}_component_model_output.txt, where {analysis} is the current
    analysis type as defined in the 'analysis' input parameter and {component} is the current component model as
    defined by the first key in the model. This file is created and stored in a folder named 'components_data' within
    the directory defined by the 'out_dir' input parameter.
    """

    # Get original default print output location
    original_stdout = sys.stdout

    # If the components_data folder does not exist, create it
    if not os.path.exists(os.path.join(out_dir, 'components_data')):
        os.makedirs(os.path.join(out_dir, 'components_data'))

    # For each key, value pair in the models dictionary
    for k, v in zip(models.keys(), models.values()):

        # Create and open the current component model output file in write mode
        with open(os.path.join(out_dir, 'components_data',
                               '{}_{}_component_model_output.txt'.format(analysis, k)), 'w') as y_file:

            # Set the default print location to the currently open file
            sys.stdout = y_file

            # Print the name of the current component model
            print("{}:".format(k))

            # Expand the dictionary contained within the current value
            v = v.__dict__

            # For each key, value pair in the current component model value
            for m, n in zip(v.keys(), v.values()):
                # If the current key can be found in a list of excluded keys, skip it
                if m in COMPONENT_ATTRIBUTE_IGNORE:
                    pass

                # If the current key is a model ('model')
                elif m == 'model':
                    # If the current value is a string
                    if isinstance(n, str):
                        # If the current value is an empty string
                        if n == '':
                            # Set the current value to None
                            n = None
                        # Print the current key, value pair
                        print('\t{}: {}'.format(m, n))

                    else:  # otherwise
                        # Print the current key and the name of the model
                        # taken from the model object
                        print('\t{}: {}'.format(m, n.__qualname__))

                # If the current key is included in the keys list
                elif m in ['model_args', 'model_kwargs', 'workdir', 'grid_obs_keys',
                           'obs_base_names', 'pars_bounds', 'temp_data_bounds',
                           'workdir_index', 'run_frequency']:

                    # Try to see if the current value has zero length, and if so set it to None
                    try:
                        if len(n) == 0:
                            n = None
                    # Otherwise, skip it
                    except:
                        pass

                    # If the current value is a dictionary type
                    if isinstance(n, (dict, collections.OrderedDict,
                                      matk.ordereddict.OrderedDict)):
                        # Print the current key as header
                        print('\t{}:'.format(m))
                        # For each key, value pair in the current value
                        for o, p in zip(n.keys(), n.values()):
                            # Print the current key, value pair
                            print('\t\t{}: {}'.format(o, p))

                    # If the current value is a string or integer
                    elif isinstance(n, (str, int)):
                        # If the current value is an empty string
                        if n == '':
                            # Set the current value to None
                            n = None
                        # Print the current key, value pair
                        print('\t{}: {}'.format(m, n))

                    # If the current value has a dictionary attribute
                    elif hasattr(n, '__dict__'):
                        # Print the current key as header
                        print('\t{}:'.format(m))
                        # Expand the dictionary of the current value
                        n = n.__dict__
                        # For each key, value pair in the current value dictionary
                        for o, p in zip(n.keys(), n.values()):
                            # Print the key, value pair
                            print('\t\t{}: {}'.format(o, p))

                    elif isinstance(n, list):  # if the current value is a list
                        # If the current value is a list with zero elements
                        if n is None:
                            # Print the current key, value pair, where value is equal to None
                            print("\t{}: {}".format(m, n))

                        else:  # otherwise
                            # Print the current key as header
                            print("\t{}:".format(m))
                            # For each element in the current value list
                            for o in n:
                                # Print the current element
                                print("\t\t{}".format(o))

                    # If the current key satisfies none of the preceding criteria
                    else:
                        # Print the current key, value pair
                        print('\t{}: {}'.format(m, n))

                # If the current key is a parameters key (ends in 'pars')
                elif m[-4:] == 'pars':
                    # If the current value is an empty string
                    if n == '':
                        # Set the value to None
                        n = None
                        # Print the key,value pair
                        print('\t{}: {}'.format(m, n))

                    else:  # otherwise
                        # If the length of the current value is zero
                        if len(n) == 0:
                            # Set the value to None
                            n = None
                            # Print the key, value pair
                            print('\t{}: {}'.format(m, n))

                        else:  # otherwise
                            # Print the 'parameters:' header
                            print('\t{}:'.format(m))
                            # For each key, value pair in the values dictionary
                            for o, p in zip(n.keys(), n.values()):
                                # If the current value is a string or integer
                                if isinstance(p, (str, int)):
                                    # If the current value is an empty string
                                    if p == '':
                                        # Set the current value to None
                                        p = None
                                    # Print the current key, value pair
                                    print("\t\t{}: {}".format(o, p))

                                # If the current value is a Parameter object
                                elif isinstance(p, matk.parameter.Parameter):
                                    # Extract the dictionary attribute of the parameter
                                    p = vars(p)
                                    # Print the current key as header
                                    print('\t\t{}:'.format(o))
                                    # For each key, value pair in the current value
                                    for q, r in zip(p.keys(), p.values()):
                                        # If the current key is not in the COMPONENT_PAR_IGNORE list
                                        if q not in COMPONENT_PAR_IGNORE:
                                            # If the current key is 'from_internal'
                                            if q == 'from_internal':
                                                # Print the current key and the extracted name of the value
                                                print('\t\t\t{}: {}'.format(q, r.__qualname__))
                                            # If the current key is '_discrete_vals'
                                            elif q == '_discrete_vals':
                                                # Set a value string starting with double open brackets
                                                val_string = '[['
                                                # For each item in the current values list
                                                for s in r:
                                                    # For each item in the current item list
                                                    for t in s:
                                                        # Add the current item to the values string
                                                        val_string = val_string + '{},'.format(t)

                                                    # After adding all items in the current item list,
                                                    # close the current list and open and new one in the string
                                                    val_string = val_string[:-1] + '],['

                                                # After adding all items in the current values list,
                                                # close the list in the string
                                                val_string = val_string[:-2] + ']'
                                                # Print formatted string
                                                print('\t\t\t{}: {}'.format(q, val_string))

                                            else:  # otherwise
                                                # Print current key, value pair
                                                print('\t\t\t{}: {}'.format(q, r))

                                else:  # otherwise
                                    # Print current key, value pair
                                    print("\t\t{}: {}".format(o, p))

                # If the current key is a set of keyword arguments (ends in 'kwargs')
                elif m[-6:] == 'kwargs':

                    # Check if the current value length is zero, and if so, set it to None
                    try:
                        if len(n) == 0:
                            n = None
                    except:  # otherwise, skip it
                        pass

                    # If the current value is an empty string
                    if n == '':
                        # Set the current value to None
                        n = None

                    # If the current value is a type of dictionary
                    if isinstance(n, (dict, collections.OrderedDict,
                                      matk.ordereddict.OrderedDict)):
                        # Print the current key as header
                        print('\t{}:'.format(m))
                        # For each key, value pair in the current value
                        for o, p in zip(n.keys(), n.values()):
                            # Print the current key, value pair
                            print('\t\t{}: {}'.format(o, p))

                    # If the current value is a string or integer
                    elif isinstance(n, (str, int)):
                        # Print the current key, value pair
                        print('\t{}: {}'.format(m, n))

                    # If the current value has a dictionary attribute
                    elif hasattr(n, '__dict__'):
                        # Print the current key as header
                        print('\t{}:'.format(m))
                        # Expand the dictionary for the current value
                        n = n.__dict__
                        # For each key, value pair in the current value dictionary
                        for o, p in zip(n.keys(), n.values()):
                            # Print the key, value pair
                            print('\t\t{}: {}'.format(o, p))

                    # If the current value does not satisfy any of the preceding criteria
                    else:
                        # Print the current key, value pair
                        print('\t{}: {}'.format(m, n))

                # If the current key is found in the associated list
                elif m in ['linkobs', 'accumulators', 'grid_obs', 'local_obs']:
                    # Check if the current value length is zero, and if so, set it to None
                    try:
                        if len(n) == 0:
                            n = None
                    except:  # otherwise, skip it
                        pass

                    # If the current value is an empty string, set it to None
                    if n == '':
                        n = None

                    # If the current value is a dictionary type
                    if isinstance(n, (dict, collections.OrderedDict,
                                      matk.ordereddict.OrderedDict)):
                        # Print the current key as header
                        print('\t{}:'.format(m))
                        # For each key, value pair in the current value dictionary
                        for o, p in zip(n.keys(), n.values()):
                            # If the current value is a string or integer
                            if isinstance(p, (str, int)):
                                # If the current value is an empty string, set it to None
                                if p == '':
                                    p = None
                                # Print the current key, value pair
                                print("\t\t{}: {}".format(o, p))

                            # If the current value is an Observation object
                            elif isinstance(p, matk.observation.Observation):
                                # Extract the dictionary from the current value
                                p = vars(p)
                                # Print the current key as header
                                print('\t\t{}:'.format(o))
                                # For each key, value pair in the current value dictionary
                                for q, r in zip(p.keys(), p.values()):
                                    # If the current key is not in the COMPONENT_PAR_IGNORE list
                                    if q not in COMPONENT_PAR_IGNORE:
                                        # If the current key is 'from_internal'
                                        if q == 'from_internal':
                                            # Print the current key and the name
                                            # of the current value object
                                            print('\t\t\t{}: {}'.format(q, r.__qualname__))

                                        else:  # otherwise
                                            # Print the current key, value pair
                                            print('\t\t\t{}: {}'.format(q, r))

                            else:  # otherwise
                                # Print the current key, value pair
                                print("\t\t{}: {}".format(o, p))

                    # If the current value is either a string or integer
                    elif isinstance(n, (str, int)):
                        # Print the current key, value pair
                        print('\t{}: {}'.format(m, n))

                    # If the current value has a dictionary attribute
                    elif hasattr(n, '__dict__'):
                        # Print the current key as header
                        print('\t{}:'.format(m))
                        # Expand the dictionary of the current value
                        n = n.__dict__
                        # For each key, value pair in the current value dictionary
                        for o, p in zip(n.keys(), n.values()):
                            # Print the current key, value pair
                            print('\t\t{}: {}'.format(o, p))

                    else:  # otherwise
                        # Print the current key, value pair
                        print('\t{}: {}'.format(m, n))

                # If the current value is a string or an integer
                elif isinstance(n, (str, int)):
                    # Check if the current value length is zero, and if so, set it to None
                    try:
                        if len(n) == 0:
                            n = None
                    except:  # otherwise, skip it
                        pass

                    # If the current value is an empty string
                    if n == '':
                        # Set the current value to None
                        n = None

                    # Print the current key, value pair
                    print("\t{}: {}".format(m, n))

                # If the current value is a 'time_array' or 'time_points'
                elif m in ['time_array', 'time_points']:
                    time_start, time_end, time_diff, timestep_check = check_time_steps(n)

                    # If all time steps are the same
                    if timestep_check:
                        # Print the current key as header and define a list using the start and end times
                        print("\t{}: [{}-{}]".format(m, time_start, time_end))
                        # Print the time_step
                        print("\ttime_step: {}".format(time_diff[0]))

                    else:  # if there are different time steps
                        # Print the whole time array
                        print("\t{}: {}".format(m, n))

                # If the current key is observations ('obs')
                elif m == 'obs':
                    # Print the location where the observation CSV files are stored
                    print("\t{}: {}".format(m, os.path.join(
                        out_dir, 'csv_files', 'simulation_results') + ".csv"))

                # If the current key contains parameter values ('parameter_values')
                elif m == 'parameter_values':
                    # Print the size of the parameter value array
                    print('\t{}: Array({}, {})'.format(m, len(n), len(n[0])))

                # If the current key is either 'intr_names' or 'filenames'
                elif m in ['intr_names', 'filenames']:
                    # If the current value is an empty string
                    if n == '':
                        # Print the current key and value of None
                        print("{}: None".format(m))

                    else:  # otherwise
                        # Create lists of keys and values from the current value
                        n_keys = list(n.keys())
                        n_vals = list(n.values())
                        # Format 'intr_names' or 'filenames'
                        # Get the file number of the first data file
                        names_start = n_keys[0]
                        # Get the file number of the last data file
                        names_end = n_keys[-1]
                        # Get 'intr_names' base name
                        names_name = re.sub(r"\d+", "", n_vals[0])

                        if "." in names_name:
                            # Combine base name, starting number, ending number,
                            # and file extension to create data_file_str
                            names_str = names_name[:names_name.find('.')] + '[' \
                                + str(names_start) + '-' + str(names_end) + ']'\
                                    + names_name[names_name.find('.'):]

                        else:
                            # Combine base name, starting number, ending number,
                            # and file extension to create data_file_str
                            names_str = names_name + '[' + str(names_start) \
                                + '-' + str(names_end) + ']'

                        # Print data file name(s)
                        if m == 'intr_names':
                            print("\tintr_names: {}".format(names_str))

                        elif m == 'filenames':
                            print("\tfilenames: {}".format(names_str))

                # If the current key does not satisfy the preceding criteria
                else:
                    # Print the current key as header
                    print('\t{}:'.format(m))
                    # Pass the current value to a recursive interpreter function
                    parse_recursive(n, 2)

            # Add an empty line to the output
            print('\n')

        # Reset the default print location to its original value for each component
        sys.stdout = original_stdout


def parse_recursive(v, layer_num):
    """
    Parse_recursive decomposes and prints an input based on the input type

    Inputs:
    :param v: current component models
    :type v: multiple possible types

    :param layer_num: number of tabs to add to a print string for output
    :type layer_num: int

    Output:
    Formatted print strings output to a text file.
    """
    # Set tab string to the number of current tabs specified by layer_num
    tab_string = '\t' * layer_num

    # If the input is a dictionary type
    if isinstance(v, (dict, collections.OrderedDict,
                      matk.ordereddict.OrderedDict)):
        # For each key, value in the dictionary
        for m, n in zip(v.keys(), v.values()):
            # If the current key is '_parent', skip it
            if m == '_parent':
                pass
            # If the current value is a Parameter object
            elif isinstance(n, matk.parameter.Parameter):
                # Print the current key as header
                print(tab_string + "{}:".format(m))
                # Extract the dictionary attribute from the parameter object
                n = vars(n)
                # Run the Parameter dictionary through the parse_recursive
                # function with a layer number increased by 1
                parse_recursive(n, layer_num + 1)
            else:  # otherwise
                # Print the current key as header
                print(tab_string + "{}:".format(m))
                # Run the dictionary through the parse_recursive function
                # with a layer number increased by 1
                parse_recursive(n, layer_num + 1)

    elif hasattr(v, '__dict__'):  # if the input has a dictionary attribute
        # Expand the dictionary from the input
        v = v.__dict__
        # For each key, value pair in the input
        for m, n in zip(v.keys(), v.values()):
            # If the current key is '_parent', skip it
            if m == '_parent':
                pass
            else:  # otherwise
                # Print the current key as header
                print(tab_string + "{}:".format(m))
                # Run the dictionary through the parse_recursive function
                # with a layer number increased by 1
                parse_recursive(n, layer_num + 1)

    elif isinstance(v, list):  # if the input is a list
        # For each item in the list
        for p in v:
            # If the item is a string or integer
            if isinstance(p, (str, int)):
                # Print the item
                print(tab_string + "\t{}".format(p))
            else:  # otherwise
                # If the current item has a dictionary attribute
                if hasattr(p, '__dict__'):
                    # Expand the current item dictionary
                    p = p.__dict__
                    # Run the dictionary through the parse_recursive function
                    # with a layer number increased by 1
                    parse_recursive(p, layer_num + 1)
                else:  # otherwise
                    # If the current item is an empty string
                    if p == '':
                        # Set the current item as None
                        p = None
                    # Print the current item
                    print(tab_string + "\t{}:".format(p))
    else:  # otherwise
        # If the input is an empty string
        if v == '':
            # Set the input to None
            v = None
        # Print the input
        print(tab_string + "{}".format(v))


def dict_recursive(k, v, layer_num):
    """
    Dict_recursive decomposes and prints a recursive dictionary

    Inputs:
    :param k: current dictionary key
    :type k: str

    :param v: current dictionary value
    :type v: dict

    :param layer_num: number of tabs to add to a print string for output
    :type layer_num: int

    Output:
    Formatted print strings output to a text file.
    """
    # Set tab string to the number of current tabs specified by layer_num
    tab_string = '\t' * layer_num

    # If the length of the value dictionary is zero
    if len(v) == 0:
        # Set the value to None
        v = None
        # Print the current key, value pair
        print(tab_string + "{}: {}".format(k, v))
    else:  # otherwise
        # Print the current key as header
        print(tab_string + "{}:".format(k))
        # For each key, value pair
        for m, n in zip(v.keys(), v.values()):
            # If the current key is not '_parent'
            if m != '_parent':
                # If the current value is a blank string
                if n == '':
                    # Set the current value to None
                    n = None
                    # Print current key, value pair
                    print(tab_string + "\t{}: {}".format(m, n))

                # If the current value is a dictionary type
                elif type(n) in [dict, collections.OrderedDict,
                                 matk.ordereddict.OrderedDict]:
                    # Run the current key, value pair through the dictionary_recursive function
                    # and increment the layer number by 1
                    dict_recursive(m, n, layer_num + 1)

                elif hasattr(n, '__dict__'): # if the current value has a dictionary attribute
                    # Expand the current value dictionary
                    n = n.__dict__
                    # Run the current key, value pair through the dictionary_recursive function
                    # and increment the layer number by 1
                    dict_recursive(m, n, layer_num + 1)

                else: # if the current value does not satisfy any of the preceding criteria
                    # Print the current key, value pair
                    print(tab_string + "\t{}: {}".format(m, n))


def check_time_steps(time_points):
    """
    Check whether time steps formed by time_points are the same.
    """
    # Get time points start and end
    time_start, time_end = time_points[0], time_points[-1]
    # Take first differential of time points
    time_diff = np.diff(time_points)
    # Compare differential to determine if time steps are uniform
    check_flag = np.all(time_diff == time_diff[0])

    return time_start, time_end, time_diff, check_flag
