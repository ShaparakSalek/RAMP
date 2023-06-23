# -*- coding: utf-8 -*-
"""
Script analyzes folder content (seismic_data or vp_model) and downloads selected
data files. User needs to copy their API-key into line 18 obtained from EDX to use
this script.


@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import re
import requests
import zipfile

if __name__ == "__main__":
    # Use your API-Key
    api_key = ""
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
    data_case = 2  # 1 is seismic data, 2 is velocity data
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
    scenario_indices = list(range(1, 3))  # e.g., list(range(51, 201)) requests files from 51 to 200
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
    to_unzip = True  # False means do not unzip, True means unzip after download

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
    to_delete = True  # False: do not delete archive, True: delete archives

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

    # print(json_data.keys())

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
