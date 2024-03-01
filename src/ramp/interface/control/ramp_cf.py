# -*- coding: utf-8 -*-
"""

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
from datetime import datetime
import logging

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

sys.path.append(os.sep.join([BASE_DIR, 'src']))

from ramp.interface.control import process_download_data

DEFAULT_OUTPUT_DIRECTORY =  os.sep.join([BASE_DIR, 'output'])
PROCESSING_START_MGS = 'Processing of file {} is started.'
PROCESSING_COMPLETE_MSG = 'Processing of file {} is completed.'

# TODO place ramp_version variable as one of the module attributes
ramp_version = '0.0.1'
OUTPUT_HEADER = 'RAMP: {ramp_version}\nRuntime: {start_time_frmtd} \n'

LOG_DICT = {'all': logging.NOTSET,
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL}

def setup_logger(logging_file_name, analysis_log, log_level=logging.INFO):
    logger = logging.getLogger('')
    # Remove default existing handlers
    logger.handlers.clear()
    logger.setLevel(log_level)
    # logging formatter for log files with more details
    log_formatter1 = logging.Formatter(
        fmt='%(levelname)s %(module)s.%(funcName)s: %(message)s',
        datefmt='%m-%d %H:%M')
    # logging formatter for console output
    log_formatter2 = logging.Formatter(
        fmt='%(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M')

    # Setup logging to log file
    file_handler = logging.FileHandler(filename=logging_file_name, mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_formatter1)
    logger.addHandler(file_handler)

    # Setup logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(log_formatter2)
    logger.addHandler(console)

    return logger, file_handler


def process_control_file(cf_name, log_level='info'):
    """
    Reads in yaml data control file to create RAMP workflow and run it.

    Parameters
    ----------
    cf_name : str
        Full path to the yaml formatted RAMP control file
    log_level : str
        Level of logging messages to be shown during run of the control file

    Returns
    -------
    bool
        Indicates whether processing of the control file was successful
    """
    start_time = datetime.now()
    start_time_frmtd = start_time.strftime('%Y-%m-%d_%H.%M.%S')
    # Load yaml file data
    with open(cf_name, 'r') as yaml_cf:
        yaml_data = yaml.load(yaml_cf, Loader=yaml.SafeLoader)

    model_data = yaml_data.get('ModelParams', {})
    model_data_not_setup = (model_data == {})
    if 'Logging' not in model_data:
        model_data['Logging'] = log_level
    log_level = log_dict[model_data['Logging'].lower()]

    logger, file_handler = setup_logger(logging_file_name, analysis_log, log_level=log_level)
    logger.info(PROCESSING_START_MGS)

    # Process download data setup if provided
    ddata = yaml_data.get('DownloadData', {})
    dd_flag = 2  # to indicate that everything is ok with DownloadData entry setup
    if ddata:
        dd_flag = process_download_data(ddata)

    # If ModelParams entry is not provided, there is no need to continue
    if model_data_not_setup:
        logger.info('ModelParams entry is not setup.')
        logger.info(PROCESSING_COMPLETE_MSG.format(cf_name))
        return False
    else:
        workflow_data = yaml_data.get('Workflow', {})
        if not workflow_data:
            logger.warn('Workflow entry is not setup.')
            logger.warn(PROCESSING_COMPLETE_MSG.format(cf_name))
            return False
        else:
            to_run = workflow_data.get('Run', True)
            if not to_run:
                logger.info('Workflow is setup to not be run.')
                logger.info(PROCESSING_COMPLETE_MSG.format(cf_name))
                return False

    if dd_flag not in [1, 2]:
        # It is assumed that data will be needed for specified workflow
        # If data is not available,
        logger.error('Processing of DownloadData entry did not result in the data download')
        logger.info(PROCESSING_COMPLETE_MSG.format(cf_name))
        return False
