# -*- coding: utf-8 -*-
"""
Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from ramp.configuration.point import Point
from ramp.configuration.receiver import Source, Receiver
from ramp.configuration.point_set import PointSet


POINTS_CLASS = {'point': Point,
                'source': Source,
                'receiver': Receiver}


class BaseConfiguration():
    def __init__(self, sources=None, receivers=None, name='Unnamed',
                 sc_name='source', rc_name='receiver'):
        """
        Parameters
        ----------
        sources : numpy.array of shape (nsources, 3)
            Array of x, y, z coordinates of sources of a given configuration
        receivers : numpy.array of shape (nreceivers, 3)
            Array of x, y, z coordinates of sources of a given configuration

        name : string, optional
            Name of configuration

        Returns
        -------
        Instance of BaseConfiguration class containing information
        about the configuration sources and receivers
        """
        # Set name attribute
        self.name = name

        # Set source and receiver class names
        self.source_class = sc_name
        self.receiver_class = rc_name

        # Check whether both sources and receivers are provided
        if sources is None and receivers is None:
            # Create empty SeismicSurvey
            self._sources = []
            self.num_sources = 0
            self._receivers = []
            self.num_receivers = 0

        else:
            if sources is not None:
                # Create set of sources
                self.num_sources = sources.shape[0]
                self._sources = self.create_point_set(
                    'sources', sources, self, POINTS_CLASS[sc_name])

            if receivers is not None:
                # Create set of receivers
                self.num_receivers = receivers.shape[0]
                self._receivers = self.create_point_set(
                    'receivers', receivers, self, POINTS_CLASS[rc_name])

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, new_sources):
        if type(new_sources) is np.ndarray:  # numpy.array with coordinates
            self._sources = self.create_point_set(
                'sources', new_sources, self, POINTS_CLASS[self.source_class])
            self.num_sources = len(new_sources)
        elif hasattr(new_sources, 'points'):  # set of sources
            self._sources = new_sources
            self.num_sources = len(new_sources.points)

    @property
    def receivers(self):
        return self._receivers

    @receivers.setter
    def receivers(self, new_receivers):
        if type(new_receivers) is np.ndarray:  # numpy.array with coordinates
            self._receivers = self.create_point_set(
                'receivers', new_receivers, self, POINTS_CLASS[self.receiver_class])
            self.num_receivers = len(new_receivers)
        elif hasattr(new_receivers, 'points'):  # set of receivers
            self._receivers = new_receivers
            self.num_receivers = len(new_receivers.points)

    @staticmethod
    def create_point_set(name, point_coords, config, point_class):
        point_set = PointSet(name, point_coords, point_class=point_class)

        for point in point_set.elements:
            point.config = config

        return point_set
