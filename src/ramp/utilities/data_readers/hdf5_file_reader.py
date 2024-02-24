# -*- coding: utf-8 -*-
"""

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import numpy as np
import h5py


def default_h5_file_reader(file_name, time_index=0, obs_name='pressure'):
    """
    Read data from provided *.h5 file.

    Parameters
    ----------
    file_name : str
        Path to the file containing data to be read.
    time_index : int
        Index of a time point for which data is to be exported and which
        determines what group within h5 file it belongs to
    obs_names : str or list of str
        Names of observations to export from h5 files

    Raises
    ------
    FileNotFoundError
        DESCRIPTION.

    Returns
    -------
    data : numpy.ndarray
        Data loaded from the provided file.

    """

    if os.path.isfile(file_name):
        if isinstance(obs_name, str):
            obs_name = [obs_name]

        # Initialize dictionary data to be exported
        data = {}
        # Determine which group will be read
        key_to_read = 't{}'.format(time_index)
        # Read data and save into dictionary
        with h5py.File(file_name, 'r') as hf:
            for nm in obs_name:
                data[nm] = hf[key_to_read][nm][()]

                if nm == 'gravity':
                    data[nm] = data[nm].T
        return data

    raise FileNotFoundError('File {} is not found.'.format(file_name))
