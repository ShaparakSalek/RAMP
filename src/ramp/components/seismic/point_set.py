# -*- coding: utf-8 -*-
"""
PointSet is a set of points combined by a common topic: e.g., sources,
receivers, or sensors

Last modified: March 8th, 2023

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL
"""
import os
import sys
import logging
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from ramp.components.seismic.point import Point

class PointSet():
    def __init__(self, name, xyz_coords=None, regular=False, point_class=Point,
                 **kwargs):
        """
        Parameters
        ----------
        xyz_coords : numpy.array of shape (npoints, 3)
            Array of x, y, z coordinates of points to be added to the point set, optional
            DESCRIPTION. The default is None.
        regular : boolean, optional
            DESCRIPTION. The default is True.
        **kwargs : dictionary
            Dictionary of optional keyword arguments allowing to create a point set with
            different options.
            Possible options:
                'x', 'y', 'z' - np.array of shapes (nx, ), (ny, ), (nz, ).
                In this case, the code will use numpy.meshgrid to create the points.
                'x_min', 'x_max', 'nx', 'y_min', 'y_max', 'ny', 'z_min', 'z_max', 'nz'
                - scalars providing extent of the domain in x, y, z direction and
                number of points in x, y, and z-directions
                'center', 'rx', 'ry', 'rz', 'nx', 'ny', 'nz' - produces cylinder
                with base being a circle in xy-plane

        Returns
        -------
        Instance of PointSet class.

        """
        # Initialize instance attributes
        self.name = name
        self.point_class = point_class
        self._elements = []
        self.regular = regular
        self.num_elements = 0
        self.unique = {}
        self.extent = {}
        self.coordinates = None

        if xyz_coords is not None:
            self.num_elements = xyz_coords.shape[0]
            self.coordinates = xyz_coords
            self._elements = self.create_points_list(xyz_coords, point_class)
            self.set_additional_attributes(xyz_coords)

        elif kwargs:  # additional arguments are provided
            self.process_constructor_arguments(**kwargs)

    @property
    def elements(self):
        return self._elements

    @elements.setter
    def elements(self, new_points, point_class=None):
        if type(new_points) is np.ndarray:  # numpy.array with coordinates
            self._elements = self.create_points_list(new_points, self.point_class)
        elif isinstance(new_points, list):  # list of points
            self._elements = [point.copy(name=str(ind+1))
                            for ind, point in enumerate(new_points)]
            if point_class is not None:
                self.point_class = point_class
        self.num_elements = len(new_points)

    @staticmethod
    def create_points_list(point_coords, point_class):
        points = []
        num_points = point_coords.shape[0]
        for ind in range(num_points):
            points.append(point_class(x=point_coords[ind, 0],
                                      y=point_coords[ind, 1],
                                      z=point_coords[ind, 2],
                                      index=ind, name=str(ind+1)))

        return points

    def process_constructor_arguments(self, **kwargs):
        """
        Checks what additional arguments are provided and create point set based on
        the inputs.

        Parameters
        ----------
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Start with checking first option: whether x and y are provided
        required_keys = ['x', 'y', 'z']
        missing_keys = check_missing_keys(kwargs, required_keys)
        if not missing_keys:  # if all keys ('x', 'y' and 'z') missing_keys is empty list
            self.setup_meshgrid_coordinates(**kwargs)
        elif 1 <= len(missing_keys) <= 2:
            raise_missing_keys_error(self.name, missing_keys)
        else:
            required_keys = ['nx', 'ny', 'nz',
                             'x_min', 'x_max',
                             'y_min', 'y_max',
                             'z_min', 'z_max']
            missing_keys = check_missing_keys(kwargs, required_keys)
            if not missing_keys:  # if all keys are present
                self.setup_linspace_coordinates(**kwargs)
            elif 1 <= len(missing_keys) <= 2:
                raise_missing_keys_error(self.name, missing_keys)
            else:
                # TODO cylinder based set of points
                pass

    def setup_meshgrid_coordinates(self, **kwargs):
        """


        Parameters
        ----------
        **kwargs : TYPE
            'x', 'y', 'z' - np.array of shapes (nx, ), (ny, ), (nz, ).
            In this case, the code will use numpy.meshgrid to create the points.

        Returns
        -------
        None.

        """
        self.regular = True
        nx = len(kwargs['x'])
        ny = len(kwargs['y'])
        nz = len(kwargs['z'])
        self.num_elements = nx*ny*nz

        self.unique = {nm: kwargs[nm] for nm in ['x', 'y', 'z']}
        self.extent = {nm: [self.unique[nm][0], self.unique[nm][-1]]
                           for ind, nm in enumerate(['x', 'y', 'z'])}

        x, y, z = np.meshgrid(kwargs['x'], kwargs['y'], kwargs['z'],
                              indexing='ij')
        self.coordinates = np.zeros((self.num_elements, 3))
        self.coordinates[:, 0] = x.reshape(x, (self.num_elements, ))
        self.coordinates[:, 1] = y.reshape(y, (self.num_elements, ))
        self.coordinates[:, 2] = z.reshape(z, (self.num_elements, ))

        self._elements = self.create_points_list(self.coordinates)

    def setup_linspace_coordinates(self, **kwargs):
        """


        Parameters
        ----------
        **kwargs : TYPE
            'x_min', 'x_max', 'nx', 'y_min', 'y_max', 'ny', 'z_min', 'z_max', 'nz'
            - scalars providing extent of the domain in x, y, z direction and
            number of points in x, y, and z-directions

        Returns
        -------
        None.

        """
        new_kwargs = {
            nm: np.linspace(
                kwargs['{}_min'.format(nm)], kwargs['{}_max'.format(nm)],
                num=kwargs['n{}'.format(nm)]) for nm in ['x', 'y', 'z']}
        self.setup_meshgrid_coordinates(**new_kwargs)

    def set_additional_attributes(self, xyz_coords=None):
        """
        Determine unique x, y, and z coordinates for a given point set,
        as well as the extent of the set in each direction.

        Parameters
        ----------
        xyz_coords : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if xyz_coords is None:
            if self.elements:  # non-empty list of instances of Point class
                xyz_coords = np.zeros((len(self.elements, 3)))
                for ind, point in enumerate(self.elements):
                    xyz_coords[ind, :] = self.point.coordinates
            else:
                warn_msg = ''.join([
                    'Attribute "elements" of point set {} is not setup. Attributes ',
                    '"unique" and "extent" were not updated. ',
                    'Add points or provide x, y, z coordinates.']).format(self.name)
                logging.warning(warn_msg)

        if xyz_coords is not None:
            # If coordinates are provided/updated calculate unique values of x, y, and z coordinates
            self.unique = {nm: np.unique(xyz_coords[:, ind])
                           for ind, nm in enumerate(['x', 'y', 'z'])}
            self.extent = {nm: [self.unique[nm][0], self.unique[nm][-1]]
                               for ind, nm in enumerate(['x', 'y', 'z'])}

    def create_point_set(self, name, indices=None, **kwargs):
        """
        Create and return a point set with a subset of points from the original point set.
        If an original point set is irregular, the returned point set might be degenerate
        even though the original point set is not degenerate. Plotting the resulting
        point set is recommended to check that the resulting point set is sufficient
        for the intended application.
        For a regular point set a selection of points based on indices for x, y, and z
        is recommended. In this case additional arguments with kwargs need to be used.
        Possible options:
            'x_indices', 'y_indices', 'z_indices': list of indices for x, y, and
            z coordinates. This option can be used only for regular point set.
            In both cases attribute unique will be used to extract
            corresponding points.

        Parameters
        ----------
        indices : list.
            if indices are None the copy of the survey is returned. The points
            of the new point set are copied as well.

        Returns
        -------
        None.

        """
        if not kwargs:  # if dictionary of additional attributes is empty
            # Create empty point set
            new_point_set = PointSet(name)
            if indices is None:
                indices = list(range(self.num_elements))
            new_point_set.elements = [point for ind, point in enumerate(self.elements) \
                                    if ind in indices]
            new_xyz_coords = self.coordinates[indices, :]
            new_point_set.set_additional_attributes(new_xyz_coords)
        else:
            if self.regular:  # given point set is regular
                # Use dictionary attribute unique to extract points
                required_keys = ['x_indices', 'y_indices', 'z_indices']
                missing_keys = check_missing_keys(kwargs, required_keys)
                if not missing_keys:
                    x_inds = kwargs['x_indices']
                    y_inds = kwargs['y_indices']
                    z_inds = kwargs['z_indices']
                    new_point_set = PointSet(name,
                                             x=self.unique['x'][x_inds],
                                             y=self.unique['y'][y_inds],
                                             z=self.unique['z'][z_inds])
                elif 1 <= len(missing_keys) <= 2:
                    raise_missing_keys_error(name, missing_keys)
            else:
                warn_msg = ''.join([
                    'PointSet {} is irregular. The point set cannot create ',
                    'a subset of points using provided arguments.']).format(self.name)
                logging.warning(warn_msg)

        return new_point_set


def check_missing_keys(data_dict, required_keys):
    """
    Check whether dictionary contains all required keys.

    Returns list of missing keys or empty list if all are present.
    """
    missing_keys = []

    for key in required_keys:
        if key not in data_dict:
            missing_keys.append(key)

    return missing_keys


def raise_missing_keys_error(name, missing_keys):
    """
    Raise error if some or all required parameters of the setup of points
    are missing.

    Parameters
    ----------

    Raises
    ------
    KeyError


    Returns
    -------
    None.

    """
    err_msg = ''.join(['Incomplete information is provided for the setup ',
                       'of PointSet {}. Keys {} are missing.']).format(
                           name, missing_keys)
    logging.error(err_msg)
    raise KeyError(err_msg)


if __name__ == "__main__":
    # Create a point set
    xyz_coords = np.zeros((6, 3))
    xyz_coords[:, 0] = np.arange(1, 7)
    xyz_coords[:, 1] = np.arange(9, 15)

    point_set1 = PointSet('test', xyz_coords)
