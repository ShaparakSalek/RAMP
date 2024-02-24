# -*- coding: utf-8 -*-
"""
Last modified:

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
from math import sqrt
import numpy as np


class Point():
    """
    Class is designed to be flexible with respect to the number of dimensions.
    Points can be defined as 1d, 2d, and 3d.
    Any axis (x, y, z) can be chosen as primary for 1d point. In this case,
    the rest of the coordinates would assume 0 values, by default.
    For 2d points any two axes can be chosen as primary. The remaining coordinate
    would assume 0 value, by default.
    """
    def __init__(self, x=0, y=0, z=0, name=''):
        """
        Constructor of Point class object.

        Parameters
        ----------
        x : int or float, optional
            x-coordinate of a new point. The default is 0.
        y : int or float, optional
            y-coordinate of a new point. The default is 0.
        z : int or float, optional
            z-coordinate of a new point. The default is 0.
        name : string, optional
            Name of the newly created point. the default is ''.

        Returns
        -------
        Instance of Point class
        """
        self.x = x
        self.y = y
        self.z = z
        self.name = name

        # Additional attributes
        self.index = None
        self.config = None

    def __repr__(self):
        """
        Return string representation of Point instance.

        Returns
        -------
        String representation of the point.

        """
        if self.name:
            str_repr = 'Point {} at (x, y, z) = ({}, {}, {})'.format(
                self.name, self.x, self.y, self.z)
        else:
            str_repr = 'Point at (x, y, z) = ({}, {}, {})'.format(
                self.x, self.y, self.z)

        return str_repr

    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

    def shift(self, **kwargs):
        """
        Shift location of the point by the specified offset in each dimension.

        Parameters
        ----------
        kwargs : optional keyword arguments
            Optional keyword arguments specifying how the location of the point
            changes. Possible keys:
            dx : int or float, optional
                offset in x-direction. The default is 0.
            dy : int or float, optional
                offset in y-direction. The default is 0.
            dz : int or float, optional
                offset in z-direction. The default is 0.

        Returns
        -------
        None.

        """
        for key in ['dx', 'dy', 'dz']:
            if key in kwargs:
                # Get previous value assigned to a given coordinate
                prev_value = getattr(self, key[1])
                # Assigns new value taking offset into account
                setattr(self, key[1], prev_value+kwargs[key])

    def move(self, x=None, y=None, z=None):
        """
        Change one or all coordinates (x, y, and z) of the point.

        Parameters
        ----------
        x : int or float, optional
            new x location. The default is None.
        y : int or float, optional
            DESCRIPTION. The default is None.
        z : int or float, optional
            DESCRIPTION. The default is None.

        If a particular coordinate is not specified in the call of the method,
        it will stay the same.

        Returns
        -------
        None.

        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z

    def distance_from(self, point2):
        """
        Calculate distance from the specified point.

        Parameters
        ----------
        point2 : instance of Point class
            Point from which the distance to be calculated.

        Returns
        -------
        Distance from the specified point.

        """
        distance = sqrt((self.x - point2.x)**2
                        + (self.y - point2.y)**2
                        + (self.z - point2.z)**2)
        return distance

    def copy(self, name=''):
        """
        Create a new Point with the same coordinates (but possibly different name).

        Parameters
        ----------
        name : string, optional
            Name of the newly created point. The default is ''.

        Returns
        -------
        Instance of Point class

        """
        return Point(self.x, self.y, self.z, name=name)

    def at_the_same_location(self, point2):
        """
        Check whether another point is placed at the same location.

        Parameters
        ----------
        point2 : instance of Point class
            Point whose coordinates will be compared to the coordinates of the
            point.

        Returns
        -------
        True or False.
        True if another point have the same coordinates, and False otherwise.

        """
        if np.all(np.equal(self.coordinates, point2.coordinates)):
            return True
        else:
            return False

if __name__ == "__main__":
    # Create two points
    point1 = Point(4.0, 5.0, name='A')  # z=0
    point2 = Point(7.0, 9.0, name='B')  # z=0

    # Print string representation of both points
    print(point1)
    print(point2)

    # Calculate distance between points and print it
    print('Distance between points A and B:', point1.distance_from(point2))
    print('--------------')

    # Shift point A to other location and print it
    print('Shift point A by dx=3, dy=3:')
    point1.shift(dx=3, dy=3)
    print(point1)

    # Calculate distance between points and print it
    print('Distance between points A and B:', point1.distance_from(point2))
    print('--------------')

    # Move point B to another location: only y-coordinate will be changed
    # and print it
    print('Move point B to a new location:')
    point2.move(y=7.0)
    print(point2)

    # Calculate distance between points and print it
    print('Distance between points A and B:', point2.distance_from(point1))
    print('--------------')

    # Create point with the same coordinates as point A
    point3 = point1.copy(name='C')
    print(point3)
    print('Is point A at the same location as point C?',
          point1.at_the_same_location(point3))
    print('Coordinates of point A', point1.coordinates)
    print('Coordinates of point C', point3.coordinates)
