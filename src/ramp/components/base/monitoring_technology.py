# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 13:30:55 2023

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from openiam import ComponentModel


class MonitoringTechnology(ComponentModel):
    """
    This class is a base class for components modeling any kind of monitoring
    technology.
    """
    def __init__(self, name, parent, model_kwargs=None):
        """
        Constructor method of MonitoringTechnology class.

        Parameters
        ----------
        name : str
            Name of MonitoringTechnology instance under which it will be known
            to its parent.
        parent : SystemModel class instance
            System model to which the instance belongs

        Returns
        -------
        Instance of MonitoringTechnology class

        """
        # Setup keyword arguments of the 'self.model' method provided by the system model
        if model_kwargs is None:
            model_kwargs = {'time_point': 0.0}  # default value of 0 days
        elif 'time_point' not in model_kwargs:
            model_kwargs['time_point'] = 0.0

        super().__init__(name=name, parent=parent, model=self.process_data,
                         model_args=None, model_kwargs=model_kwargs)

        # Add type attribute
        self.class_type = 'MonitoringTechnology'

    def process_data(self, p, time_point=None, data=None, baseline=None, **kwargs):
        """


        Parameters
        ----------
        p : dict
            Parameters of component
        data : numpy.ndarray
            Data to be processed.

        Returns
        -------
        None.

        """
        # data is what this component keeps track of
        pass
