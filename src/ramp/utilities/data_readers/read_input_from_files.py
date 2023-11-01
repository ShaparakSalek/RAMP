
# -*- coding: utf-8 -*-
"""

@author: Yuan Tian @ LLNL

"""
import yaml
from obspy.io.segy.core import _read_segy
import numpy as np

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