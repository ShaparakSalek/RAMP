import os

from .point import Point
from .seismic_configuration import Source, Receiver, SeismicSurveyConfiguration
from .ramp_base_classes import RAMPSystemModel
from .data_container import DataContainer
from .monitoring_technology import MonitoringTechnology

RAMP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

__version__ = 'alpha_0.0.1-23.04.30'

__all__ = ['RAMP_DIR',
           'RAMPSystemModel',
           'Point',
           'Source',
           'Receiver',
           'SeismicSurveyConfiguration',
           'DataContainer',
           'MonitoringTechnology',
           ]
