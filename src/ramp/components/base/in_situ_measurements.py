# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:36:20 2023

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from ramp.components.base.monitoring_technology import MonitoringTechnology


# Purpose of the class is monitoring well simulation which can track pressure,
# TDS, pH, etc at a given location and compare them to a threshold
class InSituMeasurements(MonitoringTechnology):

    def __init__(self, parent, name, loc_x=None, loc_y=None, loc_z=None):
        """


        Parameters
        ----------
        parent : TYPE
            DESCRIPTION.
        name : TYPE
            DESCRIPTION.
        loc_x : TYPE, optional
            DESCRIPTION. The default is None.
        loc_y : TYPE, optional
            DESCRIPTION. The default is None.
        loc_z : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Instance of InSituMeasurements class.

        """

        super().__init__(self, name, parent, loc_x=loc_x, loc_y=loc_y, loc_z=loc_z,
                         model_args=None, model_kwargs=None)

        # Add type attribute
        self.class_type = 'InSituMeasurements'
        # self.model is self.collect_data
        # might need accumulators to keep track of detection time

    def collect_data(self, p, data):
        """


        Parameters
        ----------
        p : TYPE
            DESCRIPTION.
        data : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # data is what this component keeps track of
        pass
