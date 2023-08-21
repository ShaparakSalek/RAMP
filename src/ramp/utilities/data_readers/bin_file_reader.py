# -*- coding: utf-8 -*-
"""

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import numpy as np


def default_bin_file_reader(file_name, data_shape=None, r_order='F',
                            move_axis_destination=None):
    """

    Parameters
    ----------
    file_name : str
        Path to the file containing data to be read.
    data_shape : tuple, optional
        Tuple describing into what shape the data should be reshaped after load.
        The default is None, i.e., no reshaping of data is needed
    r_order : str, optional
        The order in which the data is to be read. The default is 'F', and in most
        case it would be the most appropriate.
    move_axis_destination : list, optional
        Destination positions for each of the original axes. The original axes
        are given starting from 0. The default is None, i.e., no moving of axes
        is needed.

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
        data = np.fromfile(file_name, dtype=np.float32)
        if data_shape is not None:
            data = data.reshape(data_shape, order=r_order)

        if move_axis_destination is not None:
            num_dims = len(data.shape)
            data = np.moveaxis(data, list(range(num_dims)), move_axis_destination)

        return data

    raise FileNotFoundError('File {} is not found.'.format(file_name))
