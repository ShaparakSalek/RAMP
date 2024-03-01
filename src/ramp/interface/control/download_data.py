# -*- coding: utf-8 -*-
"""
Methods present in this module allow to download data sets/files from EDX
using the control file setup.


@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
import re
import requests
import zipfile
import logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

DEFAULT_EDX_DATA_DIRECTORY =  os.sep.join([BASE_DIR, 'data', 'user', 'edx'])


def process_indices(input_vals):
    """
    Convert string input for indices into corresponding list of integers.

    Parameters
    ----------
    input_vals : str
        String containing list of integer indices to be converted into list.

    Returns
    -------
    list_indices : TYPE
        DESCRIPTION.

    """
    str_indices = input_vals.split(',')
    list_indices = []
    for item in str_indices:
        if '-' in item:
            ind1, ind2 = (int(ind) for ind in item.split('-'))
            list_indices = list_indices + list(range(ind1, ind2+1))
        else:
            list_indices = list_indices + [int(item)]

    return list_indices


def process_download_data(setup_data, logger=None, to_debug=False):
    """
    Download data from EDX using setup dictionary provided as argument to the method.

    Parameters
    ----------
    setup_data : dict
        Dictionary containing setup needed to download data from EDX.

    Returns
    -------
    None.

    """
    if logger is None:
        if to_debug:
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO
        logger = logging.getLogger('')
        logger.handlers.clear()
        logger.setLevel(log_level)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(log_level)
        # logging formatter for console output
        log_formatter = logging.Formatter(
            fmt='%(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M')
        console.setFormatter(log_formatter)
        logger.addHandler(console)

    logger.info('Setup of data to download from EDX is being processed.')
    to_run = setup_data.get('Run', False)
    if not to_run:
        logger.info('No download of data is requested.')
        return 2

    api_key = setup_data.get('EDXAPIKey', '')
    if not api_key:
        logger.error('User EDXAPIKey entry is not provided in the setup.')
        return 0

    headers = {"EDX-API-Key": api_key}
    workspace_id = setup_data.get('WorkspaceID', 'nrap-task-4-monitoring')

    folder_id = setup_data.get('DataFolderID', None)
    if folder_id is None:
        logger.error('Data folder ID is not provided in the setup.')
        return 0

    output_directory = setup_data.get('OutputDirectory', DEFAULT_EDX_DATA_DIRECTORY)
    output_directory = os.sep.join([BASE_DIR, output_directory])
    if not os.path.exists(output_directory):
        logger.debug(f'Created data directory {output_directory}')
        os.mkdir(output_directory)

    instr_data = setup_data.get('Instructions', {'FileNameTemplate': None})
    to_unzip = instr_data.get('UnzipArchive', False)
    to_delete = instr_data.get('DeleteArchive', False)
    file_name = instr_data.get('FileNameTemplate', None)

    # Process indices if provided
    indices_str_input = instr_data.get('Indices', None)
    if indices_str_input is not None:
        indices = process_indices(indices_str_input)
    else:
        indices = []

    # Process to be excluded indices if provided
    to_exclude_indices_str_input = instr_data.get('ToExclude', None)
    if to_exclude_indices_str_input is not None:
        to_exclude_indices = process_indices(to_exclude_indices_str_input)
    else:
        to_exclude_indices = []

    data = {"workspace_id": workspace_id,
            "folder_id": folder_id}

    # The following link stays the same even for different workspaces
    # This is an URL to API endpoint
    url = 'https://edx.netl.doe.gov/api/3/action/folder_resources'

    # Get data associated with folder
    r = requests.post(
        url,             # url to API endpoint
        headers=headers, # deaders dictionary
        data=data) # dictionary of data params

    # Convert data into dictionary format
    json_data = r.json()

    logger.debug(json_data.keys())

    # Get folder resources: files names, their urls, etc.
    resources = json_data['result']['resources']

    # Find number of resources
    num_resources = len(resources)
    logger.debug('Total number of resources', num_resources, '\n')
    resources_names = []
    name_to_url_dict = {}
    for res in resources:
        resources_names.append(res['name'])
        name_to_url_dict[res['name']] = res['url']
        logger.debug('Resource {} is found'.format(res['name']))

    # Get url of files to be downloaded
    if file_name is not None:
        urls = {}
        if indices:
            for ind in indices:
                if ind not in to_exclude_indices:
                    file_name_frmted = file_name.format(ind)
                    if file_name_frmted in resources_names:
                        urls[file_name_frmted] = name_to_url_dict[file_name_frmted]
        else:
            if file_name in resources_names:
                urls[file_name] = name_to_url_dict[file_name]
    else: # all files will be downloaded
        urls = name_to_url_dict

    # Download files
    for file_id, url_link in urls.items():
        if os.path.exists(os.path.join(output_directory, file_id)):
            logger.info(f'Skipping file {file_id}')
            continue

        logger.info(f'Downloading file {file_id}')

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
        for file_id in urls:
            path_name = os.sep.join([output_directory, file_id])
            file_name, file_extension = os.path.splitext(file_id)
            if file_extension == '.zip':
                logger.info(f'Unzipping file {file_id}')
                try:
                    with zipfile.ZipFile(path_name, 'r') as zip_ref:
                        folder_to_extract, _ = os.path.splitext(
                            os.sep.join([output_directory, file_id]))
                        try:
                            os.mkdir(folder_to_extract)
                            zip_ref.extractall(folder_to_extract)
                        except FileExistsError:
                            pass

                    if to_delete:
                        logger.info(f'Removing file {file_id}')
                        os.remove(path_name)
                except:
                    pass

    return 1
