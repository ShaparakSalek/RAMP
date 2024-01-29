# -*- coding: utf-8 -*-
"""
Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
from math import sqrt

import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from ramp.configuration.point import Point


class Source(Point):
    def __init__(self, x=0, y=0, z=0, index=0, name='', config=None):
        """
        Constructor of Source class object.

        Parameters
        ----------
        x : int or float, optional
            x-coordinate of a new source. The default is 0.
        y : int or float, optional
            y-coordinate of a new source. The default is 0.
        z : int or float, optional
            z-coordinate of a new source. The default is 0.
        index : int, optional
            integer index assigned to a new source. The default is 0.
        name : string, optional
            Name of the newly created source. the default is ''.
        config : Configuration class instance, optional
            Configuration to which the source belongs

        Returns
        -------
        Instance of Source class
        """
        if name == '':
            name = str(index)
        super().__init__(x=x, y=y, z=z, name=name)
        self.index = index
        self.config = config

    def copy(self, name=''):
        """
        Create a new Source with the same coordinates (but possibly different name).

        Parameters
        ----------
        name : string, optional
            Name of the newly created point. The default is ''.

        Returns
        -------
        Instance of Source class

        """
        return Source(self.x, self.y, self.z, name=name)

    def __repr__(self):
        """
        Return string representation of a source point.

        Returns
        -------
        String representation of a source point.

        """
        if self.name:
            str_repr = 'Source {} at (x, y, z) = ({}, {}, {})'.format(
                self.name, self.x, self.y, self.z)
        else:
            str_repr = 'Source at (x, y, z) = ({}, {}, {})'.format(
                self.x, self.y, self.z)

        return str_repr


class Receiver(Point):
    def __init__(self, x=0, y=0, z=0, index=0, name='', config=None):
        """
        Constructor of Receiver class object.

        Parameters
        ----------
        x : int or float, optional
            x-coordinate of a new receiver. The default is 0.
        y : int or float, optional
            y-coordinate of a new receiver. The default is 0.
        z : int or float, optional
            z-coordinate of a new receiver. The default is 0.
        index : int, optional
            integer index assigned to a new source. The default is 0.
        name : string, optional
            Name of the newly created receiver. the default is ''.
        config : Configuration class instance, optional
            Configuration to which the source belongs

        Returns
        -------
        Instance of Receiver class
        """
        if name == '':
            name = str(index)
        super().__init__(x=x, y=y, z=z, name=name)
        self.index = index
        self.config = config

    def copy(self, name=''):
        """
        Create a new Receiver with the same coordinates (but possibly different name).

        Parameters
        ----------
        name : string, optional
            Name of the newly created point. The default is ''.

        Returns
        -------
        Instance of Receiver class

        """
        return Receiver(self.x, self.y, self.z, name=name)

    def __repr__(self):
        """
        Return string representation of receiver.

        Returns
        -------
        String representation of the receiver.

        """
        if self.name:
            str_repr = 'Receiver {} at (x, y, z) = ({}, {}, {})'.format(
                self.name, self.x, self.y, self.z)
        else:
            str_repr = 'Receiver at (x, y, z) = ({}, {}, {})'.format(
                self.x, self.y, self.z)

        return str_repr
