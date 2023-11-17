# -*- coding: utf-8 -*-
"""

@author: Yuan Tian @ LLNL
"""

import os
import sys
from obspy import Trace, Stream, UTCDateTime
from obspy.core import AttribDict
from obspy.io.segy.segy import SEGYTraceHeader, SEGYBinaryFileHeader
import numpy as np

import yaml


def convert_matrix_to_segy(sens2d,filename):
    """
    Converts a 2D matrix into a SEGY file format.

    Parameters:
        sens2d (np.array): 2D array representing the sensitivity matrix.
        filename (str): Name of the output SEGY file.

    Returns:
        None. Writes the converted data to a SEGY file.
    """
    stream = Stream()

    for i in range(sens2d.shape[0]):
        # Create some random data.
        data = sens2d[i][:]
        data = np.require(data, dtype=np.float32)
        trace = Trace(data=data)

        # Attributes in trace.stats will overwrite everything in
        # trace.stats.segy.trace_header
        trace.stats.delta = 0.01
        # SEGY does not support microsecond precision! Any microseconds will
        # be discarded.
        trace.stats.starttime = UTCDateTime(2011, 11, 11, 11, 11, 11)

        # If you want to set some additional attributes in the trace header,
        # add one and only set the attributes you want to be set. Otherwise the
        # header will be created for you with default values.
        if not hasattr(trace.stats, 'segy.trace_header'):
            trace.stats.segy = {}
        trace.stats.segy.trace_header = SEGYTraceHeader()
        trace.stats.segy.trace_header.trace_sequence_number_within_line = i + 1
        trace.stats.segy.trace_header.receiver_group_elevation = 444

        # Add trace to stream
        stream.append(trace)

    # A SEGY file has file wide headers. This can be attached to the stream
    # object.  If these are not set, they will be autocreated with default
    # values.
    stream.stats = AttribDict()
    stream.stats.textual_file_header = 'Textual Header!'
    stream.stats.binary_file_header = SEGYBinaryFileHeader()
    stream.stats.binary_file_header.trace_sorting_code = 5
    stream.write(filename, format="SEGY", data_encoding=1,
                 byteorder=sys.byteorder)




def write_optimal_design_to_yaml(design2,yaml_output_path):
    """
    Writes an optimal design configuration into a YAML file.

    Parameters:
        design2 (list): List of design configurations.
        yaml_output_path (str): Path to the output YAML file.

    Returns:
        None. Writes the optimal design configuration to a YAML file.
"""
    # Convert design2 to dictionary format
    design2_dict = {}
    for i in range(0, len(design2), 2):
        key = design2[i]
        values = design2[i + 1]
        design2_dict[key] = eval(str(values))
    with open(yaml_output_path, 'w') as outfile:
        yaml.dump(design2_dict, outfile, default_flow_style=False)
