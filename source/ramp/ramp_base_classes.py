# -*- coding: utf-8 -*-
"""
Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matk import matk
from matk.lmfit.asteval import Interpreter
from matk.parameter import Parameter
from matk.observation import Observation
from matk.ordereddict import OrderedDict
from openiam import SystemModel, ComponentModel, SampleSet


class RAMPSystemModel(SystemModel):  # another option for name RAMPDriver
    """
    This class is supposed to handle system models with specific setup relevant
    to RAMP workflows.
    We will add methods allowing handling of optimization related tasks,
    location-based components, etc.
    """
    def __init__(self, model_kwargs=None):
        """
        Constructor method of RAMPSystemModel class.

        Parameters
        ----------
        model_kwargs : dict, optional
            dictionary of additional optional keyword arguments of the system
            model. Possible keys are 'time_array' or 'time_point'.
            time_array and time_point data are assumed to be given in days.
            time_array is assumed to be of numpy.array type and to have
            at least two elements, otherwise an error message is shown.
            The default value for model_kwargs dict is None.

        Returns
        -------
        RAMPSystemModel class object

        """

        super().__init__(model_kwargs=model_kwargs)
        # Attribute that will keep references to point_sets associated or not with different
        # components
        self.point_sets = OrderedDict()
        self.configurations = OrderedDict()

    def add_point_set_object(self, point_set_object):
        """
        Add instance of Layout class to system model.

        Parameters
        ----------
        point_set_object : Layout class instance
            point_set to be added to the system model


        Returns
        -------
        Layout class object
        """
        if point_set_object.name in self.point_sets:
            self.point_sets[point_set_object.name] = point_set_object
        else:
            self.point_sets.__setitem__(point_set_object.name, point_set_object)

    def add_configuration(self, cm_object, config_object):
        """


        Parameters
        ----------
        cm_object : ComponentModel object
            Component model to be added to the system model

        Returns
        -------
        None.

        """
        if cm_object.name in self.configurations:
            self.configurations[cm_object.name] = config_object
        else:
            self.configurations.__setitem__(cm_object.name, config_object)
