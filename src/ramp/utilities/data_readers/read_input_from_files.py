
# -*- coding: utf-8 -*-
"""

@author: Yuan Tian @ LLNL

"""
import yaml

import numpy as np
import requests
import glob

def read_yaml_parameters(file_path):
    """
    Reads and flattens parameters from a given YAML file.

    Parameters:
        file_path (str): Path to the input YAML file.

    Returns:
        flattened_params (dict): Flattened dictionary of parameters read from the YAML file.
    """
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    # Flattening the restructured YAML content
    flattened_params = {}
    for section, values in params.items():
        if isinstance(values, dict):  # Adding a check for dictionary items
            for key, value in values.items():
                flattened_params[key] = value
        else:
            flattened_params[section] = values
    return flattened_params


def read_sens_from_segy(filename,sen_nor):
    """
    Reads sensitivity values from a SEGY file and normalizes them if required.

    Parameters:
        filename (str): Path to the input SEGY file.
        sen_nor (int): Flag to determine if sensitivity values need to be normalized.

    Returns:
        sens (np.array): 2D array of sensitivity values.
    """
    from obspy.io.segy.core import _read_segy
    st1 = _read_segy(filename)
    sens2d_read=[]
    for i,tr in enumerate(st1):
        sens2d_read.append(tr.data)
    sens=np.array(sens2d_read)
    if sen_nor==1:
        maxSens = np.max(sens)
        minSens = np.min(sens)
        dsens = maxSens - minSens
        sens = (sens - minSens) / dsens
    return sens

def download_data_from_edx(workspace_id,folder_id,api_key,outdir):
    headers = {"EDX-API-Key": api_key}

    data = {
        "workspace_id": workspace_id,
        "folder_id": folder_id,
    }

    #outdir ='./'

    # The following link stays the same even for different workspaces
    # This is a URL to API endpoint
    url = 'https://edx.netl.doe.gov/api/3/action/folder_resources'

    # Get data associated with folder
    r = requests.post(
        url,  # URL to API endpoint
        headers=headers,  # Headers dictionary
        data=data,  # Dictionary of data params
    )

    # Convert data into dictionary format
    json_data = r.json()
    print(json_data.keys())

    # Get files names, their urls in the folder resources
    resources = json_data['result']['resources']
    print('Total number of files', len(resources), '\n')

    # download all files in the folder
    for res in resources:
        file_name = res['name']
        print(file_name)
        file_url = res['url']
        print(file_url)
        r = requests.get(file_url, headers=headers)
        with open(outdir + file_name, 'wb') as file:
            file.write(r.content)


def get_all_h5_filenames(rootdir):
    h5_dirs=glob.glob(rootdir+'sim*')
    h5_dirs=sorted(h5_dirs)
    for i,h5dir in enumerate(h5_dirs):
        h5files = glob.glob(h5dir + '/*.h5')
        h5files=sorted(h5files)
        if i==0:
            all_gra_sims_fn=h5files
        else:
            all_gra_sims_fn=all_gra_sims_fn+h5files
    return all_gra_sims_fn


def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data
