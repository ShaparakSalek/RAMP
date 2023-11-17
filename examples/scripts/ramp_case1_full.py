# -*- coding: utf-8 -*-
"""
Combined scripts into one:
1. download_folder_content_on_edx.py
Script analyzes folder content (seismic_data or vp_model) and downloads selected
data files. User needs to copy their API-key into line 18 obtained from EDX to use
this script.

2. ramp_sys_seismic_monitoring_optimization_data.py

3. array_construction_nrms_processing.ipynb

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL

4. Added optimization code as well as json/yaml input/output files
@author: Alexander Hanna (alexander.hanna@pnnl.gov)
"""

import os
import sys
import re
import requests
import zipfile
import shutil
import json
import yaml
import pickle

import scipy
import scipy.spatial
from scipy.spatial import distance_matrix

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0

import itertools
import time

mpl.use('Agg')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.sep.join(['..', '..', 'source']))
sys.path.insert(0, os.sep.join(['..', '..', 'src']))

from openiam import SystemModel

sys.path.insert(0, os.sep.join(['..', '..', 'ramp']))
sys.path.insert(0, os.sep.join(['..', '..', 'ramp', 'components','base']))

from ramp.utilities.data_readers import default_bin_file_reader

from ramp import SeismicDataContainer
from ramp import SeismicSurveyConfiguration
from ramp import SeismicMonitoring
from ramp.components.seismic.seismic_configuration import SeismicSurveyConfiguration, five_n_receivers_array_creator
from ramp.optimize.ttd_det_optimization import *

#from ramp import five_n_receivers_array_creator

#from ramp.data_container import default_bin_file_reader
#from ramp.seismic_data_container import SeismicDataContainer
#from ramp.seismic_configuration import SeismicSurveyConfiguration
#from ramp.seismic_monitoring import SeismicMonitoring
#from ramp.seismic_configuration import five_n_receivers_array_creator

def subsample_to_n_points(points, n):
    """
    Subsample a set of points to the most uniformly-distributed n points,
    by returning the indexes of the sparsified points.

    :param points: List of (x, y) coordinates
    :param n: Desired number of points
    :return: List of indexes of subsampled points
    """
    points_array = np.array(points)
    points_array[:, 0] -= np.min(points_array[:, 0])
    points_array[:, 0] /= np.max(points_array[:, 0])
    points_array[:, 1] -= np.min(points_array[:, 1])
    points_array[:, 1] /= np.max(points_array[:, 1])
    indexes = np.arange(len(points_array))  # Create an array of indexes

    while len(points_array) > n:
        dists = distance_matrix(points_array, points_array)
        np.fill_diagonal(dists, np.inf)
        min_dist_idx = np.argmin(dists)
        delete_idx = np.unravel_index(min_dist_idx, dists.shape)[0]

        # Delete the point with the minimum distance and its index
        points_array = np.delete(points_array, delete_idx, axis=0)
        indexes = np.delete(indexes, delete_idx)

    return list(indexes)


if len(sys.argv) == 1:
  raise Exception('''Please include an input argument specifying the YAML or JSON filename.
Example:
>> python3 %s inputs.json
'''%sys.argv[0])

try:
    inputs = json.load(open(sys.argv[1], 'r'))
except:
    try:
        inputs = yaml.safe_load(open(sys.argv[1], 'r'))
    except: ValueError

# ====================================
# =========== Setup Step =============
# ====================================
# Duplicate options commented out in detailed setup below
# Use your API-Key
#api_key = "db3f43a7-a871-4608-b349-48c8af2b3be2"
#api_key = '2fb3645b-5657-4986-bf16-4fad657fbb45'
api_key = inputs['edx_api_key']

# Choose what data needs to be downloaded
data_case = inputs['data_case']  # 1 is seismic data, 2 is velocity data

# Choose scenarios to download
#scenario_indices = list(range(1, 992))  # e.g., list(range(51, 201)) requests files from 51 to 200
#scenario_indices = list(range(11, 14))  # e.g., list(range(51, 201)) requests files from 51 to 200
#print(inputs['scenarios'])
#print(type(inputs['scenarios']))
#if '-' in inputs['scenarios']: print(inputs['scenarios'])
if isinstance(inputs['scenarios'], int):
    scenario_indices = list(range(1, inputs['scenarios']+1))
elif isinstance(inputs['scenarios'], list):
    scenario_indices = inputs['scenarios']
elif isinstance(inputs['scenarios'], str):
    if '-' in inputs['scenarios']:
        scenario_indices = list(range(int(inputs['scenarios'].split('-')[0]),
                                      1+int(inputs['scenarios'].split('-')[1])))
    else:
        raise Exception('Error, scenarios list is not formatted correctly')
else:
    raise Exception('Error, scenarios list is not formatted correctly')

#print(scenario_indices)

#scenario_indices = inputs['scenarios']  # e.g., list(range(51, 201)) requests files from 51 to 200

# Setup whether downloaded files should be unzipped
to_unzip = True  # False means do not unzip, True means unzip after download

# Setup whether archives will be deleted after unzipping
to_delete = True  # False: do not delete archive, True: delete archives
# ====================================
# =========== End Setup ==============
# ====================================

if __name__ == "__main__":
    # Start of script from download_folder_content_on_edx.py
    # Use your API-Key
    # api_key = ""

    headers = {"EDX-API-Key": api_key}

    workspace_id = 'nrap-task-4-monitoring'
    # Kimb1.2 for RAMP folder (https://edx.netl.doe.gov/workspace/resources/nrap-task-4-monitoring?folder_id=b1ecb785-e1f9-46bc-9d04-7b943d3fe379)
    # has 2 subfolders: seismic_data and vp_model
    # Both subfolders have 986 archives with data files
    # Switching between the subfolders allows to download the needed files
    # using scripts rather than manually

    # ====================================
    # =========== Setup Step 1 ===========
    # ====================================
    # Choose what data needs to be downloaded
    # data_case = 1  # 1 is seismic data, 2 is velocity data
    if data_case == 1:
        folder_id = '9a032568-5977-494c-a5b1-a903704104e4'  # seismic_data folder id
    elif data_case == 2:
        folder_id = '2e5d0a00-281f-45e2-bc97-c6fef29d9e9b'  # vp_model folder id
    else:
        err_msg = 'Script is not setup for a data_case {}'.format(data_case)
        raise ValueError(err_msg)

    # ====================================
    # =========== Setup Step 2 ===========
    # ====================================
    # Setup indices of data files to be downloaded
    # scenario_indices = [12, 109, 302, 622, 141, 318, 881, 986, 22, 76, 269]
    # scenarios with incomplete data [449, 518, 136, 970, 397, 590, 150, 598, 863, 37, 935, 937, 749, 302, 686, 500, 245, 182, 118, 312, 313, 315, 316]
    # missing_scenario_indices = [312, 313, 315, 316, 500]
    # scenario_indices = [12, 109, 622, 141, 318, 881, 986, 22, 76, 269] #[37, 118, 136, 150, 182, 245, 302, 303] #
    # scenario_indices = list(range(555,556))  # e.g., list(range(51, 201)) requests files from 51 to 200
    # scenario_indices = [37, 118, 136, 150, 182, 245, 397, 449, 452, 454, 455, 458, 503, 505, 508, 509, 516, 526, 539, 542, 545, 548, 550, 555, 559, 560, 562, 564, 566, 567, 576, 580, 583, 588, 589, 598, 590, 605, 607, 609, 612, 614, 615, 617, 621, 625, 640, 646, 648, 649, 650, 653, 686, 743, 744, 746, 747, 748, 749, 831, 832, 833, 834, 835, 836, 837, 839, 840, 842, 843, 863, 935, 937, 970, 991]
    # Print names of the requested files
    # Define file name format
    if data_case == 1:
        base_name = 'data_sim{:04}'
    elif data_case == 2:
        base_name = 'vp_sim{:04}'
    file_name = base_name + '.zip'
    print('The names of the requested files:')
    for ind in scenario_indices:
        print(file_name.format(ind))
    print('')

    # ====================================
    # =========== Setup Step 3 ===========
    # ====================================
    # Setup whether downloaded files should be unzipped
    # to_unzip = True  # False means do not unzip, True means unzip after download

    # ====================================
    # =========== Setup Step 4 ===========
    # ====================================
    # Setup output directory where the downloaded and unzipped files will be stored
    #output_directory = os.sep.join(['..', '..', 'data', 'user'])
    if data_case == 1:
        output_directory = os.sep.join([inputs['directory_seismic_data']])
    elif data_case == 2:
        output_directory = os.sep.join([inputs['directory_velocity_data']])

    step_ind = 1
    if inputs['download_data']:
        print('Step {}: Downloading data from EDX...'.format(step_ind))
        step_ind = step_ind + 1
        # ====================================
        # =========== Setup Step 5 ===========
        # ====================================
        # Setup whether archives will be deleted after unzipping
        # to_delete = True  # False: do not delete archive, True: delete archives

        data = {
            "workspace_id": workspace_id,
            "folder_id": folder_id,
            # "folder_id": ['7c3e598c-3486-468d-9d5b-9eee6da7637a', '4d932b54-57be-404a-b38e-6e9caa6e3b23']  list of folders format
            # "only_show_type": 'folders' # Uncomment this line if you wish to only return folders
            # "only_show_type": 'resources' # Uncomment this line if you wish to only return resources
        }

        # The following link stays the same even for different workspaces
        # This is an URL to API endpoint
        url = 'https://edx.netl.doe.gov/api/3/action/folder_resources'

        # Get data associated with folder
        r = requests.post(
            url, # URL to API endpoint
            headers=headers, # Headers dictionary
            data=data, # Dictionary of data params
            )

        # Convert data into dictionary format
        json_data = r.json()
        #print(json_data)
        #print(json_data.keys())

        # Get folder resources: files names, their urls, etc.
        resources = json_data['result']['resources']

        # Print number of resources to see expected number of 986 resources
        print('Total number of resources', len(resources), '\n')
        # for res in resources:
        #     print(res['name'])

        # Get URL of files to be downloaded
        urls = {}
        # Go over all resources in the folder
        for res in resources:
            if data_case == 1:
                scen_ind = int(res['name'][8:12])  # valid for seismic_data
            elif data_case == 2:
                scen_ind = int(res['name'][6:10])  # valid for vp_model
            if scen_ind in scenario_indices:
                urls[scen_ind] = res['url']

        # Download files
        for scen_ind, url_link in urls.items():
            print(f'scen_ind: {scen_ind}')
            if os.path.exists(os.path.join(output_directory, base_name.format(scen_ind)+'.zip')) or \
                    os.path.exists(os.path.join(output_directory, base_name.format(scen_ind))):
                print('Skipping file', base_name.format(scen_ind))
                continue

            print('Downloading file:', file_name.format(scen_ind))
            print('---')

            # print("Getting resource...")
            r = requests.get(url_link, headers=headers)

            fname = ''

            if "Content-Disposition" in r.headers.keys():
                fname = re.findall("filename=(.+)", r.headers["Content-Disposition"])[0]
            else:
                fname = url_link.split("/")[-1]

            if fname.startswith('"'):
                fname = fname[1:]

            if fname.endswith('"'):
                fname = fname[:-1]

            with open(os.sep.join([output_directory, fname]), 'wb') as file:
                file.write(r.content)

        # Unzip all archives if user has requested it
        if to_unzip:
            for scen_ind in urls:
                print('Unzipping {} file ...'.format(file_name.format(scen_ind)))
                path_name = os.sep.join([output_directory, file_name.format(scen_ind)])
                try:
                    with zipfile.ZipFile(path_name, 'r') as zip_ref:
                        folder_to_extract = os.sep.join([output_directory,
                                                         base_name.format(scen_ind)])
                        try:
                            os.mkdir(folder_to_extract)
                            zip_ref.extractall(folder_to_extract)
                        except FileExistsError:
                            pass

                    if to_delete:
                        print('Removing {} file ...'.format(file_name.format(scen_ind)))
                        print('---')
                        os.remove(path_name)
                except:
                    pass

        # Check downloads for complete data
        # Attempt second download for missing/incomplete files
        incomplete = []
        for scen_ind in scenario_indices:
            files = os.listdir(os.sep.join([output_directory,
                                            base_name.format(scen_ind), 'data']))
            if len(files) != 20: # could check for specific files if necessary
                incomplete.append(scen_ind)
        print(f'incomplete scenarios (1st round): {incomplete}')

        # Delete incomplete data folders
        for scen_ind in incomplete:
            files = os.sep.join([output_directory, base_name.format(scen_ind)])
            if os.path.isdir(files):
                shutil.rmtree(files)

        # Run download script again on deleted files
        # Get URL of files to be downloaded
        urls = {}
        # Go over all resources in the folder
        for res in resources:
            if data_case == 1:
                scen_ind = int(res['name'][8:12])  # valid for seismic_data
            elif data_case == 2:
                scen_ind = int(res['name'][6:10])  # valid for vp_model
            if scen_ind in incomplete: ### Only change to download script is list of files ###
                urls[scen_ind] = res['url']

        # Download files
        for scen_ind, url_link in urls.items():
            print(f'scen_ind: {scen_ind}')
            if os.path.exists(os.path.join(output_directory, base_name.format(scen_ind)+'.zip')) or \
                    os.path.exists(os.path.join(output_directory, base_name.format(scen_ind))):
                print('Skipping file', base_name.format(scen_ind))
                continue

            print('Re-downloading file:', file_name.format(scen_ind))
            print('---')

            # print("Getting resource...")
            r = requests.get(url_link, headers=headers)

            fname = ''

            if "Content-Disposition" in r.headers.keys():
                fname = re.findall("filename=(.+)", r.headers["Content-Disposition"])[0]
            else:
                fname = url_link.split("/")[-1]

            if fname.startswith('"'):
                fname = fname[1:]

            if fname.endswith('"'):
                fname = fname[:-1]

            with open(os.sep.join([output_directory, fname]), 'wb') as file:
                file.write(r.content)

        # Unzip all archives if user has requested it
        if to_unzip:
            for scen_ind in urls:
                print('Unzipping {} file ...'.format(file_name.format(scen_ind)))
                path_name = os.sep.join([output_directory, file_name.format(scen_ind)])
                try:
                    with zipfile.ZipFile(path_name, 'r') as zip_ref:
                        folder_to_extract = os.sep.join([output_directory,
                                                         base_name.format(scen_ind)])
                        try:
                            os.mkdir(folder_to_extract)
                            zip_ref.extractall(folder_to_extract)
                        except FileExistsError:
                            pass

                    if to_delete:
                        print('Removing {} file ...'.format(file_name.format(scen_ind)))
                        print('---')
                        os.remove(path_name)
                except:
                    pass

    if inputs['download_data'] or inputs['run_optimization']:
        print('Step {}: Performing check of the downloaded data...'.format(step_ind))
        step_ind = step_ind + 1
        # Check 2nd round of downloads for incomplete data to exclude from NRMS calculations
        excluded = []
        for scen_ind in scenario_indices:
            files = os.listdir(os.sep.join([output_directory, base_name.format(scen_ind), 'data']))
            if len(files) != 20: # could check for specific files if necessary
                excluded.append(scen_ind)
        print(excluded)

        if data_case == 1:
            #Start of code from ramp_sys_seismic_monitoring_optimization_data.py
            # Define keyword arguments of the system model
            time_points = 10*np.arange(1, 21)
            time_array = 365.25*time_points
            sm_model_kwargs = {'time_array': time_array}   # time is given in days

            # Setup required information for data container before creating one
            obs_name = 'seismic'
            #data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
            #output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
            #                                'ramp_sys_seismic_monitoring_optimization_data')
            data_directory = inputs['directory_seismic_data']
            output_directory = inputs['directory_nrms_data']
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            data_reader = default_bin_file_reader
            data_reader_kwargs = {'data_shape': (1251, 101, 9),
                                'move_axis_destination': [-1, -2, -3]}

            num_time_points = len(time_points)
            # excluded = [37, 118, 136, 150, 182, 245]  # 6 scenarios
            # excluded = [37, 118, 136, 150, 182, 245, 397, 449, 456, 457, 468, 469, 498, 499, 500, 590, 598, 686, 749, 831, 832, 833, 834, 835, 836, 837, 839, 840, 842, 843, 839, 863, 935, 937, 970, 991]  # 17 scenarios
            #job 477 = 468, maybe 467 with exclusions # 469?
            #job 508 = 499, maybe 498, 500?

            scenarios = set(scenario_indices).difference(excluded)
            scenarios = list(scenarios)
            num_scenarios = len(scenarios)
            family = 'seismic'
            data_setup = {}
            for scen in scenarios:
                data_setup[scen] = {'folder': os.path.join('data_sim{:04}'.format(scen), 'data')}
                for t_ind, tp in enumerate(time_points):
                    data_setup[scen]['t{}'.format(t_ind+1)] = 'data_sim{:04}_t{}.bin'.format(scen, tp)
            baseline = True

            '''# Define coordinates of sources
            num_sources = 9
            sources = np.c_[4000 + np.array([240, 680, 1120, 1600, 2040, 2480, 2920, 3400, 3840]),
                            np.zeros(num_sources),
                            np.zeros(num_sources)]

            # Define coordinates of receivers
            num_receivers = 101
            receivers = np.c_[4000 + np.linspace(0, 4000, num=num_receivers),
                            np.zeros(num_receivers),
                            np.zeros(num_receivers)]'''

            if 'sources' in inputs.keys():
                num_sources = len(inputs['sources'])
                sources = np.c_[inputs['sources'],
                                np.zeros(num_sources),
                                np.zeros(num_sources)]
            else:
                num_sources = inputs['sourcesNum']
                min_sources = inputs['sourcesMin']
                max_sources = inputs['sourcesMax']
                sources = np.c_[np.linspace(min_sources, max_sources, num=num_sources),
                                np.zeros(num_sources),
                                np.zeros(num_sources)]

            if 'receivers' in inputs.keys():
                num_receivers = len(inputs['receivers'])
                receivers = np.c_[inputs['receivers'],
                                  np.zeros(num_receivers),
                                  np.zeros(num_receivers)]
            else:
                num_receivers = inputs['receiversNum']
                min_receivers = inputs['receiversMin']
                max_receivers = inputs['receiversMax']
                receivers = np.c_[np.linspace(min_receivers, max_receivers, num=num_receivers),
                                  np.zeros(num_receivers),
                                  np.zeros(num_receivers)]

    if inputs['process_data']:
        print('Step {}: Processing seismic data into NRMS values...'.format(step_ind))
        step_ind = step_ind + 1
        # Create survey with defined coordinates
        survey_config = SeismicSurveyConfiguration(sources, receivers, name='Test Survey')

        # ------------- Create system model -------------
        sm = SystemModel(model_kwargs=sm_model_kwargs)

        # ------------- Add data container -------------
        dc = sm.add_component_model_object(
            SeismicDataContainer(name='dc', parent=sm, survey_config=survey_config,
                                total_duration=inputs['seismic_total_duration'],
                                sampling_interval=inputs['seismic_sampling_interval'],
                                family=family, obs_name=obs_name,
                                data_directory=data_directory, data_setup=data_setup,
                                time_points=time_points, baseline=baseline,
                                data_reader=data_reader,
                                data_reader_kwargs=data_reader_kwargs,
                                presetup=True))
        # Add parameters of the container
        dc.add_par('index', value=scenarios[0],
                discrete_vals=[scenarios, num_scenarios*[1/num_scenarios]])
        # Add gridded observation
        # dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
        # dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
        #                 output_dir=output_directory)

        # ------------- Add seismic monitoring technology -------------
        smt = sm.add_component_model_object(
            SeismicMonitoring(name='smt', parent=sm, survey_config=survey_config, time_points=time_points))
        # Add keyword arguments linked to the seismic data container outputs
        smt.add_kwarg_linked_to_obs('data', dc.linkobs['seismic'], obs_type='grid')
        smt.add_kwarg_linked_to_obs('baseline', dc.linkobs['baseline_seismic'], obs_type='grid')
        # Add gridded observation
        smt.add_grid_obs('NRMS', constr_type='matrix', output_dir=output_directory)
        # Add scalar observations
        for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
            smt.add_obs(nm)
            smt.add_obs_to_be_linked(nm)

        print('--------------------------------')
        print('Stochastic simulation started...')
        print('--------------------------------')
        print('Number of scenarios: {}'.format(num_scenarios))
        # Create sampleset varying over scenarios: this is not a typical setup
        # We want to make sure scenarios are not repeated.
        samples = np.array(scenarios).reshape(num_scenarios, 1)
        s = sm.create_sampleset(samples)

        results = s.run(cpus=5, verbose=False)
        print('--------------------------------')
        print('Stochastic simulation finished.')
        print('--------------------------------')

        print('--------------------------------')
        print('Collecting results...')
        print('--------------------------------')
        # Get saved gridded observations from files
        nrms = np.zeros((num_scenarios, num_time_points, num_sources, num_receivers))
        time_indices = list(range(num_time_points))

        for rlzn_number in range(1, num_scenarios+1):
            print('Realization {}'.format(rlzn_number))
            data = sm.collect_gridded_observations_as_time_series(
                smt, 'NRMS', output_directory, indices=time_indices,
                rlzn_number=rlzn_number) # data shape (num_time_points, num_sources, num_receivers)
            nrms[rlzn_number-1] = data

        file_to_save = os.path.join(
            output_directory,
            'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=nrms)

        data_check = 0
        if data_check:
            # Read file
            file_to_read = os.path.join(
                output_directory,
                'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
            d = np.load(file_to_read)
            # Determine shape of the data
            data_shape = d['data'].shape
            print(data_shape)  # (36, 20, 9, 101)
            nrms = d['data']

            # Check that the data makes sense
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111)
            ax_im = ax.imshow(nrms[0, 8, :, :], aspect='auto')

            # Set title
            title = ax.set_title('NRMS at {} years'.format(90))

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
            cbar = plt.colorbar(ax_im, label='Percentage, %')
            fig.tight_layout()

    if inputs['run_optimization']:
        print('Step {}: Running optimization...'.format(step_ind))
        step_ind = step_ind + 1
        # Start of array_construction_nrms_processing.ipynb
        # Setup directories
        #data_directory = os.path.join('..', 'user', 'output', 'ramp_sys_seismic_monitoring_optimization_data')
        #output_directory = os.path.join('..', 'user', 'output', 'ramp_sys_seismic_monitoring_optimization_data')
        data_directory = inputs['directory_nrms_data']
        output_directory = inputs['directory_nrms_data']

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # Create survey configuration with defined coordinates
        array_creator_kwargs = {'source_coords': sources,
                                'receiver_coords': receivers}
        configuration = SeismicSurveyConfiguration(
            sources, receivers, name='Test Survey', create_arrays=True,
            array_creator=five_n_receivers_array_creator,
            array_creator_kwargs=array_creator_kwargs)
        print('Number of created arrays:', configuration.num_arrays)

        # Load NRMS data
        nrms_data_file = os.path.join(data_directory, 'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
        d = np.load(nrms_data_file)
        # Determine shape of the data
        data_shape = d['data'].shape
        print(data_shape)  # (300, 20, 9, 101)  scenarios, time points, sources, receivers
        nrms = d['data']

        # Setup data that will hold results of processing NRMS data for all created arrays
        arrays_nrms = np.zeros((configuration.num_arrays,
                                num_scenarios,
                                num_time_points,
                                3)) # 3 is for number of largest NRMS values

        # Process NRMS data for each array in the set
        for array_ind in configuration.arrays:
            sind = configuration.arrays[array_ind]['source']
            rind = configuration.arrays[array_ind]['receivers']
            # Get subset of NRMS data for a given array
            subset_nrms = nrms[:, :, sind, rind]
            # Sort numbers in increasing order
            sorted_subset_nrms = np.sort(subset_nrms)
            # Keep the largest three nrms associated with a given array
            arrays_nrms[array_ind, :, :, :] = sorted_subset_nrms[:, :, -3:]

        # Save arrays_nrms data
        file_to_save = os.path.join(
            output_directory,
            'arrays_nrms_data_3max_values_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=arrays_nrms)

        sub_arrays_nrms = arrays_nrms[:, :, :, 0]
        file_to_save = os.path.join(
            output_directory,
            'arrays_nrms_data_3rd_max_value_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=sub_arrays_nrms)
        #print(arrays_nrms.shape)
        #print(sub_arrays_nrms.shape)

        stage1 = inputs['stage1']
        stage2 = inputs['stage2']

        num_arrays = configuration.num_arrays
        produced_arrays = configuration._arrays
        #print(len(produced_arrays))
        threshold = inputs['threshold_nrms']
        nrms = sub_arrays_nrms
        nrmsBool = np.array(sub_arrays_nrms>threshold, dtype='bool')
        #print(nrmsBool.shape)

        det_best = 0
        ttd_best = []
        for iScenario in range(nrmsBool.shape[1]):
            if np.any(nrmsBool[:, iScenario, :]):
                det_best += 1
            if len(np.where(np.any(nrmsBool[:, iScenario, :], axis=0))[0]) > 0:
                ttd_best += [np.min(np.where(np.any(nrmsBool[:, iScenario, :], axis=0))[0])]

        plans1 = list(single_array_timings(nrmsBool[:, :, :stage1]))
        plans1up = find_unique_pareto(plans1)
        plans1up = list(set(plans1up).union(
            find_different_density_same_timestep(plans1, produced_arrays)))

        plans2 = list(additional_array_timings(plans1up, nrmsBool[:, :, :stage1]))
        plans2up = find_unique_pareto(plans2)
        plans2up = list(set(plans2up).union(
            find_different_density_same_timestep(plans2, produced_arrays)))

        plans3 = list(additional_array_timings(plans2up, nrmsBool[:, :, :stage1]))
        plans3up = find_unique_pareto(plans3)
        plans3up = list(set(plans3up).union(
            find_different_density_same_timestep(plans3, produced_arrays)))

        x1 = np.array([plan[1] for plan in plans1])
        y1 = np.array([plan[2] for plan in plans1])
        x2 = np.array([plan[1] for plan in plans2])
        y2 = np.array([plan[2] for plan in plans2])
        x3 = np.array([plan[1] for plan in plans3])
        y3 = np.array([plan[2] for plan in plans3])
        r3 = pareto(x3, y3)

        candidates = np.where(x3==np.max(x3))[0]
        selectedPlan = np.random.choice(
            candidates[np.where(y3[candidates]==np.min(y3[candidates]))[0]])

        detected=set()

        #print(detected,len(detected))
        for iArray in plans3[selectedPlan][0]:
            detected.update(np.where(nrmsBool[iArray[0], :, iArray[1]])[0])
        #print(detected,len(detected))

        undetected=np.array(list(set(range(nrmsBool.shape[1]))-detected))

        plans4 = list(single_array_timings(nrmsBool[:, undetected, stage1:stage2]))
        plans4up = find_unique_pareto(plans4)
        plans4up = list(set(plans4up).union(
            find_different_density_same_timestep(plans4, produced_arrays)))

        x4 = np.array([plan[1] for plan in plans4])
        y4 = np.array([plan[2] for plan in plans4])
        r4 = pareto(x4, y4)

        plans5 = list(additional_array_timings(
            plans4up, nrmsBool[:, undetected, stage1:stage2]))
        plans5up = find_unique_pareto(plans5)
        plans5up = list(set(plans5up).union(
            find_different_density_same_timestep(plans5, produced_arrays)))

        x5 = np.array([plan[1] for plan in plans5])
        y5 = np.array([plan[2] for plan in plans5])
        r5 = pareto(x5, y5)

        plans6 = list(additional_array_timings(
            plans5up, nrmsBool[:, undetected, stage1:stage2]))
        plans6up = find_unique_pareto(plans6)
        plans6up = list(set(plans6up).union(
            find_different_density_same_timestep(plans6, produced_arrays)))

        x6 = np.array([plan[1] for plan in plans6])
        y6 = np.array([plan[2] for plan in plans6])
        r6 = pareto(x6, y6)

        x4 += len(detected)
        x5 += len(detected)
        x6 += len(detected)
        y4 += stage1
        y5 += stage1
        y6 += stage1

        candidates = np.where(x6==np.max(x6))[0]
        selectedPlan = np.random.choice(
            candidates[np.where(y6[candidates]==np.min(y6[candidates]))[0]])

        for iArray in plans6[selectedPlan][0]:
            detected.update(np.where(nrmsBool[iArray[0], :, iArray[1]+stage1])[0])
        undetected = np.array(list(set(range(nrmsBool.shape[1]))-detected))

        plans7 = list(single_array_timings(nrmsBool[:, undetected,stage2:]))
        plans7up = find_unique_pareto(plans7)
        plans7up = list(set(plans7up).union(
            find_different_density_same_timestep(plans7, produced_arrays)))

        plans8 = list(additional_array_timings(plans7up, nrmsBool[:, undetected, stage2:]))
        plans8up = find_unique_pareto(plans8)
        plans8up = list(set(plans8up).union(
            find_different_density_same_timestep(plans8, produced_arrays)))

        plans9 = list(additional_array_timings(plans8up, nrmsBool[:, undetected, stage2:]))
        plans9up = find_unique_pareto(plans9)
        plans9up = list(set(plans9up).union(
            find_different_density_same_timestep(plans9, produced_arrays)))

        x7 = np.array([plan[1] for plan in plans7])
        y7 = np.array([plan[2] for plan in plans7])
        r7 = pareto(x7, y7)
        x8 = np.array([plan[1] for plan in plans8])
        y8 = np.array([plan[2] for plan in plans8])
        r8 = pareto(x8, y8)
        x9 = np.array([plan[1] for plan in plans9])
        y9 = np.array([plan[2] for plan in plans9])
        r9 = pareto(x9, y9)

        plans = {'stage1': [plans1, plans2, plans3],
                 'stage2':[plans4,plans5,plans6],
                 'stage3':[plans7,plans8,plans9] }
        json.dump({'arrays':configuration.arrays, 'plans':plans},
                  open('output.json','w'))
        yaml.dump({'arrays':configuration.arrays, 'plans':plans},
                  open('output.yaml','w'))
        pickle.dump({'arrays':configuration.arrays, 'plans':plans},
                    open('output.dat', 'wb'))

        arrays_selected = []
        plans_selected = {}
        plans_selected['stage1'] = []
        plans_selected['stage2'] = []
        plans_selected['stage3'] = []

        xx = np.array(x1.tolist()+x2.tolist()+x3.tolist())
        yy = 10*np.array(y1.tolist()+y2.tolist()+y3.tolist())
        rr = pareto(xx, yy)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])), inputs['number_proposals'])
        plans = plans1+plans2+plans3
        for i in ii:
            plans_selected['stage1'] += [convertPlan_tuple2dict(plans[np.where(rr==1)[0][i]])]
            plans_selected['stage1'][-1]['plan_number'] = int(i)
            for deployment in plans[np.where(rr==1)[0][i]][0]:
                arrays_selected += [deployment[0]]

        xx = np.array(x4.tolist()+x5.tolist()+x6.tolist())
        yy = 10*np.array(y4.tolist()+y5.tolist()+y6.tolist())
        rr = pareto(xx, yy)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])),
                                   inputs['number_proposals'])
        plans = plans4+plans5+plans6
        for i in ii:
            plans_selected['stage2'] += [convertPlan_tuple2dict(plans[np.where(rr==1)[0][i]])]
            plans_selected['stage2'][-1]['plan_number'] = int(i)
            plans_selected['stage2'][-1]['time_to_detection'] += 10.0*inputs['stage1']
            for i in range(len(plans_selected['stage2'][-1]['deployments'])):
                plans_selected['stage2'][-1]['deployments'][i]['time'] += 10.0*inputs['stage1']
            for deployments in plans[np.where(rr==1)[0][i]][0]:
                arrays_selected += [deployments[0]]

        xx = np.array(x7.tolist()+x8.tolist()+x9.tolist())
        yy = 10*np.array(y7.tolist()+y8.tolist()+y9.tolist())
        rr = pareto(xx, yy)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])), inputs['number_proposals'])
        plans = plans7+plans8+plans9
        for i in ii:
            plans_selected['stage3'] += [convertPlan_tuple2dict(plans[np.where(rr==1)[0][i]])]
            plans_selected['stage3'][-1]['plan_number'] = int(i)
            plans_selected['stage3'][-1]['time_to_detection'] += 10.0*inputs['stage2']
            for i in range(len(plans_selected['stage3'][-1]['deployments'])):
                plans_selected['stage3'][-1]['deployments'][i]['time'] += 10.0*inputs['stage2']
            for deployments in plans[np.where(rr==1)[0][i]][0]:
                arrays_selected += [deployments[0]]

        arrays_selected = list(set(arrays_selected))
        arrays_summary = {}
        for i in arrays_selected:
            arrays_summary[i] = configuration.arrays[i]

        print('plans_selected', plans_selected)

        json.dump({'arrays':arrays_summary, 'plans':plans_selected},
                  open('output_summary.json','w'))
        yaml.dump({'arrays':arrays_summary, 'plans':plans_selected},
                  open('output_summary.yaml','w'))
        pickle.dump({'arrays':arrays_summary, 'plans':plans_selected},
                    open('output_summary.dat','wb'))

    if inputs['plot_results']:

        output = json.load(open('output.json', 'r'))
        plans1 = output['plans']['stage1'][0]
        plans2 = output['plans']['stage1'][1]
        plans3 = output['plans']['stage1'][2]
        plans4 = output['plans']['stage2'][0]
        plans5 = output['plans']['stage2'][1]
        plans6 = output['plans']['stage2'][2]
        plans7 = output['plans']['stage3'][0]
        plans8 = output['plans']['stage3'][1]
        plans9 = output['plans']['stage3'][2]

        x1 = np.array([plan[1] for plan in plans1])
        y1 = np.array([plan[2] for plan in plans1])
        r1 = pareto(x1, y1)
        x2 = np.array([plan[1] for plan in plans2])
        y2 = np.array([plan[2] for plan in plans2])
        r2 = pareto(x2, y2)
        x3 = np.array([plan[1] for plan in plans3])
        y3 = np.array([plan[2] for plan in plans3])
        r3 = pareto(x3, y3)
        x4 = np.array([plan[1] for plan in plans4])
        y4 = np.array([plan[2] for plan in plans4])
        r4 = pareto(x4, y4)
        x5 = np.array([plan[1] for plan in plans5])
        y5 = np.array([plan[2] for plan in plans5])
        r5 = pareto(x5, y5)
        x6 = np.array([plan[1] for plan in plans6])
        y6 = np.array([plan[2] for plan in plans6])
        r6 = pareto(x6, y6)
        x7 = np.array([plan[1] for plan in plans7])
        y7 = np.array([plan[2] for plan in plans7])
        r7 = pareto(x7, y7)
        x8 = np.array([plan[1] for plan in plans8])
        y8 = np.array([plan[2] for plan in plans8])
        r8 = pareto(x8, y8)
        x9 = np.array([plan[1] for plan in plans9])
        y9 = np.array([plan[2] for plan in plans9])
        r9 = pareto(x9, y9)

        x4 += np.max([np.max(x1), np.max(x2), np.max(x3)])
        x5 += np.max([np.max(x1), np.max(x2), np.max(x3)])
        x6 += np.max([np.max(x1), np.max(x2), np.max(x3)])
        x7 += np.max([np.max(x4), np.max(x5), np.max(x6)])
        x8 += np.max([np.max(x4), np.max(x5), np.max(x6)])
        x9 += np.max([np.max(x4), np.max(x5), np.max(x6)])

        plt.figure(figsize=(16, 5))

        plt.subplot(131)
        plt.title('Stage 1 (0-%i years)'%((inputs['stage1']-1)*10), fontsize=14)
        plt.scatter(x1, 10*y1, s=10, c='blue', label='1 array, 1 time')
        plt.scatter(x2, 10*y2, s=10, c='red', zorder=1, label='2 arrays/times')
        plt.scatter(x3, 10*y3, s=10, c='green', zorder=1, label='3 arrays/times')

        xx = np.array(x1.tolist()+x2.tolist()+x3.tolist())
        yy = 10*np.array(y1.tolist()+y2.tolist()+y3.tolist())
        rr = pareto(xx, yy)
        print(rr)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])), inputs['number_proposals'])
        #plt.scatter(xx[rr==1],yy[rr==1], s=40, c='lightgray', zorder=0, label='%i selected'%5)
        plt.scatter(xx[rr==1][ii],yy[rr==1][ii], s=80, c='gray', zorder=0,
                    label='%i selected'%inputs['number_proposals'])

        plt.locator_params(axis='x', integer=True, tight=True)
        plt.xlabel('Number of Leaks Detected/Detectable', fontsize=14)
        plt.ylabel('Average Time to First Detection [years]', fontsize=14)
        plt.legend(bbox_to_anchor=(-0.23,+1))

        plt.subplot(132)
        plt.title('Stage 2 (%i-%i years)'%((inputs['stage1'])*10, (inputs['stage2']-1)*10),
                  fontsize=14)
        plt.scatter(x4, 10*inputs['stage1']+10*y4, s=10, c='blue', label='1 array, 1 time')
        plt.scatter(x5, 10*inputs['stage1']+10*y5, s=10, c='red', zorder=1, label='2 arrays/times')
        plt.scatter(x6, 10*inputs['stage1']+10*y6, s=10, c='green', zorder=1, label='3 arrays/times')

        xx = np.array(x4.tolist()+x5.tolist()+x6.tolist())
        yy = 10*np.array(y4.tolist()+y5.tolist()+y6.tolist())
        rr = pareto(xx, yy)
        print(rr)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])),
                                 inputs['number_proposals'])
        #plt.scatter(xx[rr==1],10*inputs['stage1']+yy[rr==1], s=40, c='lightgray', zorder=0, label='%i selected'%5)
        plt.scatter(xx[rr==1][ii], 10*inputs['stage1']+yy[rr==1][ii], s=80,
                    c='gray', zorder=0, label='%i selected'%inputs['number_proposals'])

        plt.locator_params(axis='x', integer=True, tight=True)
        plt.xlabel('Number of Leaks Detected/Detectable', fontsize=14)
        plt.ylabel('Average Time to First Detection [years]', fontsize=14)

        plt.subplot(133)
        plt.title('Stage 3 (%i-200 years)'%(inputs['stage2']*10),fontsize=14)
        plt.scatter(x7, 10*inputs['stage2']+10*y7, s=10, c='blue',
                    label='1 array, 1 time')
        plt.scatter(x8, 10*inputs['stage2']+10*y8, s=10, c='red', zorder=1,
                    label='2 arrays/times')
        plt.scatter(x9, 10*inputs['stage2']+10*y9, s=10, c='green', zorder=1,
                    label='3 arrays/times')

        xx = np.array(x7.tolist()+x8.tolist()+x9.tolist())
        yy = 10*np.array(y7.tolist()+y8.tolist()+y9.tolist())
        rr = pareto(xx, yy)
        print(rr)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])), inputs['number_proposals'])
        #plt.scatter(xx[rr==1],10*inputs['stage2']+yy[rr==1], s=40, c='lightgray', zorder=0, label='%i selected'%5)
        plt.scatter(xx[rr==1][ii],10*inputs['stage2']+yy[rr==1][ii], s=80,
                    c='gray', zorder=0, label='%i selected'%inputs['number_proposals'])

        plt.locator_params(axis='x', integer=True, tight=True)
        plt.xlabel('Number of Leaks Detected/Detectable', fontsize=14)
        plt.ylabel('Average Time to First Detection [years]', fontsize=14)

        plt.savefig('%s/multi_stage_optimization.png'%inputs['directory_plots'],
                    format='png', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(16,5))

        plt.subplot(131)
        plt.title('Stage 1 (0-%i years)'%((inputs['stage1']-1)*10), fontsize=14)
        #plt.scatter(x1, 10*y1, s=10, c='blue', label='1 array, 1 time')
        #plt.scatter(x2, 10*y2, s=10, c='red', zorder=1, label='2 arrays/times')
        #plt.scatter(x3, 10*y3, s=10, c='green', zorder=1, label='3 arrays/times')

        xx = np.array(x1.tolist()+x2.tolist()+x3.tolist())
        yy = 10*np.array(y1.tolist()+y2.tolist()+y3.tolist())
        rr = pareto(xx, yy)
        print(rr)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])),
                                   inputs['number_proposals'])
        #plt.scatter(xx[rr==1], yy[rr==1], s=40, c='lightgray', zorder=0, label='%i selected'%5)
        plt.scatter(xx[rr==1][ii], yy[rr==1][ii], s=80, c='gray', zorder=0)

        plt.locator_params(axis='x', integer=True, tight=True)
        plt.xlabel('Number of Leaks Detected/Detectable', fontsize=14)
        plt.ylabel('Average Time to First Detection [years]', fontsize=14)
        plt.legend(bbox_to_anchor=(-0.23,+1))

        plt.subplot(132)
        plt.title('Stage 2 (%i-%i years)'%((inputs['stage1'])*10, (inputs['stage2']-1)*10), fontsize=14)
        #plt.scatter(x4, 10*inputs['stage1']+10*y4, s=10, c='blue', label='1 array, 1 time')
        #plt.scatter(x5, 10*inputs['stage1']+10*y5, s=10, c='red', zorder=1, label='2 arrays/times')
        #plt.scatter(x6, 10*inputs['stage1']+10*y6, s=10, c='green', zorder=1, label='3 arrays/times')

        xx = np.array(x4.tolist()+x5.tolist()+x6.tolist())
        yy = 10*np.array(y4.tolist()+y5.tolist()+y6.tolist())
        rr = pareto(xx, yy)
        print(rr)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])), inputs['number_proposals'])
        #plt.scatter(xx[rr==1], 10*inputs['stage1']+yy[rr==1], s=40, c='lightgray', zorder=0, label='%i selected'%5)
        plt.scatter(xx[rr==1][ii], 10*inputs['stage1']+yy[rr==1][ii], s=80, c='gray', zorder=0)

        plt.locator_params(axis='x', integer=True, tight=True)
        plt.xlabel('Number of Leaks Detected/Detectable', fontsize=14)
        plt.ylabel('Average Time to First Detection [years]', fontsize=14)

        plt.subplot(133)
        plt.title('Stage 3 (%i-200 years)'%(inputs['stage2']*10), fontsize=14)
        #plt.scatter(x7, 10*inputs['stage2']+10*y7, s=10, c='blue', label='1 array, 1 time')
        #plt.scatter(x8, 10*inputs['stage2']+10*y8, s=10, c='red', zorder=1, label='2 arrays/times')
        #plt.scatter(x9, 10*inputs['stage2']+10*y9, s=10, c='green', zorder=1, label='3 arrays/times')

        xx = np.array(x7.tolist()+x8.tolist()+x9.tolist())
        yy = 10*np.array(y7.tolist()+y8.tolist()+y9.tolist())
        rr = pareto(xx, yy)
        print(rr)
        ii = subsample_to_n_points(list(zip(xx[rr==1], yy[rr==1])),
                                   inputs['number_proposals'])
        #plt.scatter(xx[rr==1], 10*inputs['stage2']+yy[rr==1], s=40, c='lightgray', zorder=0, label='%i selected'%5)
        plt.scatter(xx[rr==1][ii], 10*inputs['stage2']+yy[rr==1][ii], s=80, c='gray', zorder=0)

        plt.locator_params(axis='x', integer=True, tight=True)
        plt.xlabel('Number of Leaks Detected/Detectable', fontsize=14)
        plt.ylabel('Average Time to First Detection [years]', fontsize=14)

        plt.savefig('%s/multi_stage_optimization_summary.png'%inputs['directory_plots'],
                    format='png', bbox_inches='tight')
        plt.close()

        candidates = np.where(x3==np.max(x3))[0]
        stage1_best = np.random.choice(candidates[np.where(y3[candidates]==np.min(y3[candidates]))[0]])
        print('stage1_best', plans3[stage1_best])

        xmin = +np.inf
        xmax = -np.inf
        for deployment in plans3[stage1_best][0]:
            iArray = deployment[0]
            iSource = configuration.arrays[iArray]['source']
            iReceivers = configuration.arrays[iArray]['receivers']
            xyz = np.array(configuration.sources.coordinates[iSource])
            xmin = np.min([xmin, np.min(xyz)])
            xmax = np.max([xmax, np.max(xyz)])
            xyz = np.array(configuration.receivers.coordinates[iReceivers])
            xmin = np.min([xmin, np.min(xyz)])
            xmax = np.max([xmax, np.max(xyz)])

        fig = plt.figure(figsize=(24, 8))
        kk = 1
        for deployment in plans3[stage1_best][0]:
            ax = fig.add_subplot(1, len(plans3[stage1_best]), kk, projection='3d')
            iArray = deployment[0]
            iTime = deployment[1]
            print(iArray, configuration.arrays[iArray])

            iSource = configuration.arrays[iArray]['source']
            iReceivers = configuration.arrays[iArray]['receivers']

            xyz = np.array(configuration.sources.coordinates[iSource])
            ax.scatter(xyz[0], xyz[1], xyz[2], s=50, c='b', marker='*',
                       zorder=1, label='Sources')

            xyz = np.array(configuration.receivers.coordinates[iReceivers])
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=10, c='r',
                       zorder=0, label='Receivers')

            plt.legend()
            ax.set_xlim([xmin,xmax])
            ax.set_xlabel('Easting [m]', fontsize=14)
            ax.set_ylabel('Northing [m]', fontsize=14)
            ax.set_zlabel('Depth [m]', fontsize=14)
            kk+=1

        plt.savefig('%s/arrays_stage1.png'%inputs['directory_plots'],
                    format='png', bbox_inches='tight')
        plt.close()

        candidates = np.where(x6==np.max(x6))[0]
        stage2_best = np.random.choice(
            candidates[np.where(y6[candidates]==np.min(y6[candidates]))[0]])
        print('stage2_best', plans6[stage2_best])

        xmin = +np.inf
        xmax = -np.inf
        for deployment in plans6[stage2_best][0]:
            iArray = deployment[0]
            iSource = configuration.arrays[iArray]['source']
            iReceivers = configuration.arrays[iArray]['receivers']
            xyz = np.array(configuration.sources.coordinates[iSource])
            xmin = np.min([xmin, np.min(xyz)])
            xmax = np.max([xmax, np.max(xyz)])
            xyz = np.array(configuration.receivers.coordinates[iReceivers])
            xmin = np.min([xmin, np.min(xyz)])
            xmax = np.max([xmax, np.max(xyz)])

        fig = plt.figure(figsize=(24, 8))
        kk = 1
        for deployment in plans6[stage2_best][0]:
            ax = fig.add_subplot(1, len(plans6[stage2_best]), kk, projection='3d')
            iArray = deployment[0]
            iTime  = deployment[1]
            print(iArray, configuration.arrays[iArray])

            iSource = configuration.arrays[iArray]['source']
            iReceivers = configuration.arrays[iArray]['receivers']

            xyz = np.array(configuration.sources.coordinates[iSource])
            ax.scatter(xyz[0], xyz[1], xyz[2], s=50, c='b', marker='*',
                       zorder=1, label='Sources')

            xyz = np.array(configuration.receivers.coordinates[iReceivers])
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=10, c='r',
                       zorder=0, label='Receivers')

            plt.legend()
            ax.set_xlim([xmin, xmax])
            ax.set_xlabel('Easting [m]', fontsize=14)
            ax.set_ylabel('Northing [m]', fontsize=14)
            ax.set_zlabel('Depth [m]', fontsize=14)
            kk += 1

        plt.savefig('%s/arrays_stage2.png'%inputs['directory_plots'],
                    format='png', bbox_inches='tight')
        plt.close()

        candidates = np.where(x9==np.max(x9))[0]
        stage3_best = np.random.choice(
            candidates[np.where(y9[candidates]==np.min(y9[candidates]))[0]])
        print('stage3_best', plans9[stage3_best])

        xmin = +np.inf
        xmax = -np.inf
        for deployment in plans9[stage3_best][0]:
            iArray = deployment[0]
            iSource = configuration.arrays[iArray]['source']
            iReceivers = configuration.arrays[iArray]['receivers']
            xyz = np.array(configuration.sources.coordinates[iSource])
            xmin = np.min([xmin, np.min(xyz)])
            xmax = np.max([xmax, np.max(xyz)])
            xyz = np.array(configuration.receivers.coordinates[iReceivers])
            xmin = np.min([xmin, np.min(xyz)])
            xmax = np.max([xmax, np.max(xyz)])

        fig = plt.figure(figsize=(24, 8))
        kk = 1
        for deployment in plans9[stage3_best][0]:
            ax = fig.add_subplot(1, len(plans9[stage3_best]), kk, projection='3d')
            iArray = deployment[0]
            iTime  = deployment[1]
            print(iArray, configuration.arrays[iArray])

            iSource = configuration.arrays[iArray]['source']
            iReceivers = configuration.arrays[iArray]['receivers']

            xyz = np.array(configuration.sources.coordinates[iSource])
            ax.scatter(xyz[0], xyz[1], xyz[2], s=50, c='b', marker='*',
                       zorder=1, label='Sources')

            xyz = np.array(configuration.receivers.coordinates[iReceivers])
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=10, c='r',
                       zorder=0, label='Receivers')

            plt.legend()
            ax.set_xlim([xmin, xmax])
            ax.set_xlabel('Easting [m]', fontsize=14)
            ax.set_ylabel('Northing [m]', fontsize=14)
            ax.set_zlabel('Depth [m]', fontsize=14)
            kk+=1

        plt.savefig('%s/arrays_stage3.png'%inputs['directory_plots'], format='png',
                    bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))

        #tt = []
        #dd = []
        #for i in range(nrmsBool.shape[2]):
        #    print(i, np.any(nrmsBool[:, :, :i], axis=(0, 2)), np.sum(np.any(nrmsBool[:, :, :i], axis=(0,2))))
        #    tt += [10*i]
        #    dd += [np.sum(np.any(nrmsBool[:,:,:i],axis=(0,2)))]
        #tt, dd = scatter2step(tt, dd)
        #plt.plot(tt, dd)

        print('nrmsBool.shape', nrmsBool.shape)

        print(plans3[stage1_best])
        tt  = [0]
        dd  = [0]
        for deployment in plans3[stage1_best][0]:
            tt += [10*deployment[1]]

        detections = nrmsBool[plans3[stage1_best][0][0][0], :,
                              plans3[stage1_best][0][0][1]]
        #print(plans3[stage1_best][0][0][0],plans3[stage1_best][0][0][1])
        #print(detections)
        #print(np.sum(detections))
        #exit()
        dd += [np.sum(detections)]
        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans3[stage1_best][0][1][0], :, plans3[stage1_best][0][1][1]])]
        dd += [np.sum(detections)]
        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans3[stage1_best][0][2][0], :, plans3[stage1_best][0][2][1]])]
        dd += [np.sum(detections)]

        tt += [200]
        dd += [dd[-1]]
        tt, dd = scatter2step(tt, dd)
        plt.plot(tt, dd, label='Stage 1')

        print(plans6[stage2_best])
        tt = [0]
        dd = [0]
        for deployment in plans6[stage2_best][0]:
            tt += [10*inputs['stage1']+10*deployment[1]]

        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans6[stage2_best][0][0][0], :, plans6[stage2_best][0][0][1]])]
        dd += [np.sum(detections)]
        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans6[stage2_best][0][1][0], :, plans6[stage2_best][0][1][1]])]
        dd += [np.sum(detections)]
        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans6[stage2_best][0][2][0], :, plans6[stage2_best][0][2][1]])]
        dd += [np.sum(detections)]

        tt += [200]
        dd += [dd[-1]]
        tt, dd = scatter2step(tt, dd)
        plt.plot(tt, dd,label='Stage 2')

        print(plans9[stage3_best])
        tt = [0]
        dd = [0]
        for deployment in plans9[stage3_best][0]:
            tt += [10*inputs['stage2']+10*deployment[1]]

        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans9[stage3_best][0][0][0], :, plans9[stage3_best][0][0][1]])]
        dd += [np.sum(detections)]
        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans9[stage3_best][0][1][0], :, plans9[stage3_best][0][1][1]])]
        dd += [np.sum(detections)]
        detections = [a or b for a, b in zip(
            detections,
            nrmsBool[plans9[stage3_best][0][2][0], :, plans9[stage3_best][0][2][1]])]
        dd += [np.sum(detections)]

        tt += [200]
        dd += [dd[-1]]
        tt, dd = scatter2step(tt, dd)
        plt.plot(tt, dd, label='Stage 3')

        plt.legend()
        plt.xlabel('Time [years]', fontsize=14)
        plt.ylabel('Number of Leaks Detectable', fontsize=14)
        plt.savefig('%s/detection_breakthrough_curve.png'%inputs['directory_plots'],
                    format='png', bbox_inches='tight')
        plt.close()
