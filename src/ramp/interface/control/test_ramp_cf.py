# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 09:16:07 2024

@author: veron
"""

import os
import sys
from datetime import datetime
import logging

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))))

from ramp.interface.control import process_download_data

EDXAPIKey = ''


def test_download_data_seismic():

    data = {
        'Run': True,
        'EDXAPIKey':  EDXAPIKey,
        'WorkspaceID': 'nrap-task-4-monitoring',
        # Do not change DataFolderID entry unless modifying the file for different data set
        'DataFolderID': '9a032568-5977-494c-a5b1-a903704104e4',
        # We do not recommend users to change OutputDirectory entry
        'OutputDirectory': 'data/user/seismic',
        'Instructions':  {  # might be specific to each data set case
            # We recommend to keep UnzipArchive entry True unless users
            # want to do archive unzipping manually
            'UnzipArchive': True,   # applies to all downloaded zip files
            'DeleteArchive': True,  # applies to all downloaded zip files
            # FileNameTemplate should be setup to None if all files
            # within the directory with DataFolderID are to be downloaded.
            # Indices and ToExclude entries are not used in this case
            'FileNameTemplate': 'data_sim{:04}.zip',
            'Indices': '2,3,4',   # entered in quotes '1-100, 123', or None if not applicable or all possible indices
            # ToExclude are indices that are within the range specified by Indices entry
            # but specifying files that should not be downloaded
            'ToExclude': None}} # '2' provided in quotes

    flag_var = process_download_data(data)


def test_download_data_velocity():

    data = {
        'Run': True,
        'EDXAPIKey':  EDXAPIKey,
        'WorkspaceID': 'nrap-task-4-monitoring',
        # Do not change DataFolderID entry unless modifying the file for different data set
        'DataFolderID': '2e5d0a00-281f-45e2-bc97-c6fef29d9e9b',
        # We do not recommend users to change OutputDirectory entry
        'OutputDirectory': 'data/user/velocity',
        'Instructions':  {  # might be specific to each data set case
            # We recommend to keep UnzipArchive entry True unless users
            # want to do archive unzipping manually
            'UnzipArchive': True,   # applies to all downloaded zip files
            'DeleteArchive': True,  # applies to all downloaded zip files
            # FileNameTemplate should be setup to None if all files
            # within the directory with DataFolderID are to be downloaded.
            # Indices and ToExclude entries are not used in this case
            'FileNameTemplate': 'vp_sim{:04}.zip',
            'Indices': '1-6',   # entered in quotes '1-100, 123', or None if not applicable or all possible indices
            # ToExclude are indices that are within the range specified by Indices entry
            # but specifying files that should not be downloaded
            'ToExclude': '4'}} # '2' provided in quotes

    flag_var = process_download_data(data)


def test_download_data_pressure():

    data = {
        'Run': True,
        'EDXAPIKey':  EDXAPIKey,
        'WorkspaceID': 'nrap-task-4-monitoring',
        # Do not change DataFolderID entry unless modifying the file for different data set
        'DataFolderID': '306fcd78-b271-4a51-b576-cc6348f3b3af',
        # We do not recommend users to change OutputDirectory entry
        'OutputDirectory': 'data/user/pressure',
        'Instructions':  {  # might be specific to each data set case
            # We recommend to keep UnzipArchive entry True unless users
            # want to do archive unzipping manually
            'UnzipArchive': True,   # applies to all downloaded zip files
            'DeleteArchive': True,  # applies to all downloaded zip files
            # FileNameTemplate should be setup to None if all files
            # within the directory with DataFolderID are to be downloaded.
            # Indices and ToExclude entries are not used in this case
            'FileNameTemplate': None,
            'Indices': None,   # entered in quotes '1-100, 123', or None if not applicable or all possible indices
            # ToExclude are indices that are within the range specified by Indices entry
            # but specifying files that should not be downloaded
            'ToExclude': None}} # '2' provided in quotes

    flag_var = process_download_data(data)


if __name__ == "__main__":

    test_case = 3
    if test_case == 1:      # seismic data
        test_download_data_seismic()
    elif test_case == 2:    # velocity
        test_download_data_velocity()
    elif test_case == 3:    # pressure, saturation, TDS and gravity
        test_download_data_pressure()
