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
        sc_name : string, optional
            Shortcut to the source class name. Possible options: 'source'
            for Source class and 'point' for Point class
        rc_name : string, optional
            Shortcut to the receiver class name. Possible options: 'receiver'
            for Receiver class and 'point' for Point class

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

        # Initialize configuration attributes
        self._sources = None
        self.num_sources = 0
        self._receivers = None
        self.num_receivers = 0

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
        """
        Sets _sources attribute of the class using provided argument

        Parameters
        ----------
        new_sources : numpy.array or PointSet instance
            Coordinates of the new sources provided as an array or as a PointSet
            instance

        Returns
        -------
        None.

        """
        if type(new_sources) is np.ndarray:  # numpy.array with coordinates
            self._sources = self.create_point_set(
                'sources', new_sources, self, POINTS_CLASS[self.source_class])
            self.num_sources = len(new_sources)
        elif hasattr(new_sources, 'elements'):  # set of sources
            self._sources = new_sources
            self.num_sources = len(new_sources.elements)

    @property
    def receivers(self):
        return self._receivers

    @receivers.setter
    def receivers(self, new_receivers):
        """
        Sets _receivers attribute of the class using provided argument

        Parameters
        ----------
        new_receivers : numpy.array or PointSet instance
            Coordinates of the new receivers provided as an array or as a PointSet
            instance

        Returns
        -------
        None.

        """
        if type(new_receivers) is np.ndarray:  # numpy.array with coordinates
            self._receivers = self.create_point_set(
                'receivers', new_receivers, self, POINTS_CLASS[self.receiver_class])
            self.num_receivers = len(new_receivers)
        elif hasattr(new_receivers, 'elements'):  # set of receivers
            self._receivers = new_receivers
            self.num_receivers = len(new_receivers.elements)

    @staticmethod
    def create_point_set(name, point_coords, config, point_class):
        """
        Create point set from the provided arguments.

        Parameters
        ----------
        name : string
            Name of the new configuration
        point_coords : numpy.array of shape (npoints, 3)
            Array of x, y, z coordinates of points to be added to the point set
        config : Instance of BaseConfiguration class
            Configuration to which the point set will belong
        point_class : class Point or inherited from Point
            This is a class not an instance of class which will be used to create
            members of the point set

        Returns
        -------
        point_set : Instance of PointSet class
            Point set.

        """
        point_set = PointSet(name, point_coords, point_class=point_class)

        for point in point_set.elements:
            point.config = config

        return point_set

    def plot_configuration(self):
        """
        Visualize location of sources and receivers.

        Returns
        -------
        None.

        """
        if self.receivers is None:
            warn_msg = ''.join(['Method plot_configuration cannot visualize ',
                                'configuration "{}" since receivers ',
                                '(and, possibly, sources) ',
                                'are not defined.']).format(self.name)
            logging.warning(warn_msg)
        else:
            # Initialize figure
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            fig_title = 'Distribution of receivers for {}'

            # Check whether sources are available
            if self.sources is not None:
                if self.sources.elements:
                    fig_title = 'Distribution of sources and receivers for {}'
                    # Plot the first source
                    ax.scatter(self._sources.elements[0].x,
                               self._sources.elements[0].y,
                               self._sources.elements[0].z,
                               marker='o', c='red', label='sources')

                    # Plot the rest of the sources
                    for ind in range(1, self.num_sources):
                        ax.scatter(self._sources.elements[ind].x,
                                   self._sources.elements[ind].y,
                                   self._sources.elements[ind].z,
                                   marker='o', c='red')

            # Plot the first receiver
            ax.scatter(self._receivers.elements[0].x,
                       self._receivers.elements[0].y,
                       self._receivers.elements[0].z,
                       marker='s', c='blue', label='receivers')

            # Plot the rest of the receivers
            for ind in range(1, self.num_receivers):
                ax.scatter(self._receivers.elements[ind].x,
                           self._receivers.elements[ind].y,
                           self._receivers.elements[ind].z,
                           marker='s', c='blue')

            ax.set_xlabel('x, [m]')
            ax.set_ylabel('y, [m]')
            ax.set_zlabel('z, [m]')
            ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))

            fig.suptitle(fig_title.format(self.name))

            plt.show()


def test_base_configuration1():
    # Define coordinates of sources
    num_sources = 9
    sources = np.c_[4000 + np.array([240, 680, 1120, 1600, 2040, 2480, 2920, 3400, 3840]),
                    np.zeros(num_sources),
                    np.zeros(num_sources)]

    # Define coordinates of receivers
    num_receivers = 101
    receivers = np.c_[4000 + np.linspace(0, 4000, num=num_receivers),
                      np.zeros(num_receivers),
                      np.zeros(num_receivers)]

    # Create survey configuration with defined coordinates
    configuration = BaseConfiguration(sources, receivers, name='Config 1')

    # Plot survey to see if it is created correctly
    configuration.plot_configuration()


def test_base_configuration2():
    # Define coordinates of receivers
    nx = 7
    ny = 12
    nz = 5
    num_p = nx*ny*nz

    xyz_coords = np.zeros((num_p, 3))
    x = np.linspace(0, 700, num=nx)
    y = np.linspace(0, 600, num=ny)
    z = np.linspace(0, 2000, num=nz)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    xyz_coords[:, 0] = xx.reshape((num_p, ))
    xyz_coords[:, 1] = yy.reshape((num_p, ))
    xyz_coords[:, 2] = zz.reshape((num_p, ))

    # Create survey configuration with defined coordinates
    configuration = BaseConfiguration(
        sources=None, receivers=xyz_coords, name='Config 2')

    # Plot survey to see if it is created correctly
    configuration.plot_configuration()


if __name__ == "__main__":

    test_case = 2
    available_tests = {1: test_base_configuration1,
                       2: test_base_configuration2}
    available_tests[test_case]()
