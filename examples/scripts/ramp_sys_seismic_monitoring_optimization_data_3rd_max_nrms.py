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
"""

# ====================================
# =========== Setup Step =============
# ====================================
# Duplicate options commented out in detailed setup below
# Use your API-Key
api_key = ""

# Choose what data needs to be downloaded
data_case = 1  # 1 is seismic data, 2 is velocity data

# Choose scenarios to download
scenario_indices = list(range(1, 992))  # e.g., list(range(51, 201)) requests files from 51 to 200

# Setup whether downloaded files should be unzipped
to_unzip = True  # False means do not unzip, True means unzip after download

# Setup whether archives will be deleted after unzipping
to_delete = True  # False: do not delete archive, True: delete archives
# ====================================
# =========== End Setup ==============
# ====================================

import os
import re
import requests
import zipfile
import shutil
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import time
import pickle

mpl.use('Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.sep.join(['..', '..', 'source']))
from openiam import SystemModel

sys.path.insert(0, os.sep.join(['..', '..', 'ramp']))
from ramp.data_container import default_bin_file_reader
from ramp.seismic_data_container import SeismicDataContainer
from ramp.seismic_configuration import SeismicSurveyConfiguration
from ramp.seismic_monitoring import SeismicMonitoring
from ramp.seismic_configuration import five_n_receivers_array_creator

         
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
    output_directory = os.sep.join(['..', '..', 'data', 'user'])
    if data_case == 1:
        output_directory = os.sep.join([output_directory, 'seismic'])
    elif data_case == 2:
        output_directory = os.sep.join([output_directory, 'velocity'])

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
    print(json_data)
    print(json_data.keys())

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
                    folder_to_extract = os.sep.join([output_directory, base_name.format(scen_ind)])
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
        files = os.listdir(os.sep.join([output_directory, base_name.format(scen_ind), 'data']))
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
                    folder_to_extract = os.sep.join([output_directory, base_name.format(scen_ind)])
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
        data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
        output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
                                        'ramp_sys_seismic_monitoring_optimization_data')
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
            data_setup[scen] = {'folder': os.path.join('data_sim{:04}'.format(scen),
                                                    'data')}
            for t_ind, tp in enumerate(time_points):
                data_setup[scen]['t{}'.format(t_ind+1)] = 'data_sim{:04}_t{}.bin'.format(scen, tp)
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
        dc.add_par('index', value=scenarios[0],
                discrete_vals=[scenarios, num_scenarios*[1/num_scenarios]])
        # Add gridded observation
        # dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
        # dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
        #                 output_dir=output_directory)

        # ------------- Add seismic monitoring technology -------------
        smt = sm.add_component_model_object(
            SeismicMonitoring(name='smt', parent=sm, survey_config=survey_config,
                            time_points=time_points))
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
        nrms = np.zeros((num_scenarios, num_time_points,
                        num_sources, num_receivers))
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
        
        # Start of array_construction_nrms_processing.ipynb
        # Setup directories
        data_directory = os.path.join('..', 'user', 'output', 'ramp_sys_seismic_monitoring_optimization_data')
        output_directory = os.path.join('..', 'user', 'output', 'ramp_sys_seismic_monitoring_optimization_data')
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
        arrays_nrms = np.zeros((configuration.num_arrays, num_scenarios, num_time_points, 3)) # 3 is for number of largest NRMS values
        
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
        file_to_save = os.path.join(output_directory,'arrays_nrms_data_3max_values_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=arrays_nrms)
        
        sub_arrays_nrms = arrays_nrms[:, :, :, 0]
        file_to_save = os.path.join(output_directory,'arrays_nrms_data_3rd_max_value_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=sub_arrays_nrms)