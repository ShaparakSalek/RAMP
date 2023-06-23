# -*- coding: utf-8 -*-
"""
This module contains methods and classes intended for use in interpolation
problems.
"""
import logging
import os
import sys
import numpy as np

from scipy.spatial import Delaunay

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matk.observation import Observation

def interp_weights(triangulation, xyz):
    """
    Calculate simplex vertices and their weights for the location of interest.

    Determine indices of simplex vertices belonging to the interpolator
    triangulation and weights of the data associated with the found vertices.

    :param xyz: x and y (and z) coordinates of the location of interest
    :type xyz: numpy.array of shape (1, 2) or shape (1, 3)

    :returns: list of vertices indices and weights associated with the location
        of interest; both indices and weights are numpy.array of shape (1, 3) or (1, 4)

    Note: The code in this method as well as in the method `interpolate`
          is based on the code found at:

            https://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids

    The code helped to speedup the interpolation process.
    """
    simplex = triangulation.find_simplex(xyz)
    vertices = np.take(triangulation.simplices, simplex, axis=0)
    temp = np.take(triangulation.transform, simplex, axis=0)

    d = xyz.shape[1]
    delta = xyz - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def interpolate(data, vtx, wts):
    """
    Compute interpolated values.

    Compute interpolated values for desired location using data values
    (and their weights) at the vertices of simplex which encloses
    the location of interest.

    :param data: array with data to use for interpolation; array should
        have a shape (1,N) where N is a number of different data points
    :type data: numpy.array

    :param vtx: array of indices of simplex vertices which enclose
        the location of interest; array should have a shape (1,3); indices'
        should not exceed N
    :type vtx: numpy.array of int

    :param wts: array of weights of data at simplex vertices which enclose
        the location of interest; array should have a shape (1,3); weights
        should be between 0 and 1, and sum up to 1
    :type wts: numpy.array of floats

    returns: interpolated value, interp_val, as numpy.array of shape (1,)
    """
    interp_val = np.einsum('nj,nj->n', np.take(data, vtx), wts)
    return interp_val


class GriddedObservation(Observation):
    """ NRAP-Open-IAM GriddedObservation class. """
    def __init__(self, name, constr_type='', coordinates=None, output_dir='',
                 save_type='npz', index=None, sim=None, weight=1.0, value=None):
        """
        Constructor method of the GriddedObservation class.

        :param name: observation name
        :type name: str

        :param constr_type: type of the gridded observation. Possible values:
            'array','matrix',''. The parameter indicates what type of gridded
            observation, a given component will provide. 'array' option means that
            the returned observation obs will be an array-like object and
            every element can be accessed as obs[ind]; 'matrix' means that the returned
            observation is a matrix-like object and every element can be accessed
            by providing 2 (or 3) indices of elements: either like
            obs[ind1][ind2] or obs[ind1,ind2] for 2d case, or
            obs[ind1][ind2][ind3] or obs[ind1,ind2,ind3] for 3d case;
            '' option means that constr_type is not relevant, suitable for
            defining gridded obs to be linked.
        :type: str

        :param coordinates: x, y (and z) coordinates of the grid associated
            with the observation; by default, it is None. In general,
            coordinates should be a dictionary with keys 'x', 'y' (and 'z').
            coordinates[key] should be an array-like object
            for any key in coordinates.
        :type coordinates: dict of array-like objects

        :param output_dir: directory where the observations will be stored
                           as compressed binary files
        :type output_dir: str

        :param save_type: file format for save of gridded observation;
                          currently supports 'npz', 'csv', and 'npy' formats
        :type save_type: str

        :param index: indices of time points at which observations values are
            of interest. Default value of None means that no time points should be saved
        :type index: array-like

        :param sim: simulated value
        :type sim: fl64

        :param weight: observation weight
        :type weight: fl64

        :param value: measured/initial value of observation
        :type value: fl64

        :returns: Observation object
        """
        super().__init__(name, sim=sim, weight=weight, value=value)

        # Check whether constr_type has the right value
        if constr_type in ['array', 'matrix', '']:
            self.constr_type = constr_type
        else:
            raise ValueError(''.join([
                'Constructor of the GriddedObservation class ',
                'gets the unexpected value of the argument constr_type.']))

        # Set the coordinates arguments
        self.coordinates = {}
        if coordinates is not None:
            # Transform x, y (and z) coordinates to np.array
            for key in coordinates:
                self.coordinates[key] = np.array(coordinates[key])

        # Set output directory
        try:
            self.output_dir = os.path.abspath(output_dir)
            # Check whether path to the folder containing output_dir exists
            if not os.path.exists(os.path.dirname(self.output_dir)):
                os.mkdir(os.path.dirname(self.output_dir))
            if not os.path.exists(self.output_dir):
                os.mkdir(self.output_dir)
        except:
            raise Exception(''.join([
                'Output directory cannot be created: ',
                'path to its parental folder does not exist.']))

        if index is None:
            self.time_indices = []
        else:
            self.time_indices = index

        if save_type not in ['npz', 'csv', 'npy']:
            warn_msg = ''.join(["File extension {} is not yet supported. ",
                                "Saving will default to 'npz' format"]).format(save_type)
            logging.warning(warn_msg)
            self.save_type = 'npz'
        else:
            self.save_type = save_type


class DataInterpolator():
    """ NRAP-Open-IAM DataInterpolator class. """
    def __init__(self, name, parent, header_file_dir, data_file, triangulation=None):
        """
        Constructor method of DataInterpolator

        :param name: name of interpolator
        :type name: str

        :param parent: name of the system model interpolator belongs to
        :type parent: SystemModel object

        :param header_file_dir: location (directory) of the data that will
             be used for interpolation
        :type header_file_dir: string

        :param data_file: name of *.csv file to read data from
        :type filename: string

        :returns: DataInterpolator class object
        """

        # Set up attributes of the object
        self.name = name             # name of interpolator
        self.parent = parent         # a system model interpolator belongs to
        self.header_file_dir = header_file_dir  # path to the directory with simulation data files
        self.data_file = data_file   # name of file with data

        # Create data
        data = np.genfromtxt(os.path.join(self.header_file_dir, self.data_file),
                             delimiter=",", dtype='f8', skip_header=1)

        # Determine number of data points
        self.num_data_points = data.shape[0]

        # Save data in the class attributes
        self.points = data[:, 0:2]   # x,y coordinates
        self.data = np.transpose(data[:, 2])

        # Check whether triangulation is precomputed and provided as argument
        if triangulation is None:
            # Determine triangulation for the data points
            self.triangulation = Delaunay(self.points)
        else:
            self.triangulation = triangulation

        # Log message about creating the interpolator
        debug_msg = 'DataInterpolator created with name {}'.format(name)
        logging.debug(debug_msg)

    def __call__(self, uvw):
        """
        Return data interpolated at the point of interest.

        :param uvw: x and y coordinates of the point of interest
        :type uvw: numpy array of shape (M,2); M is a number of points

        :returns: data interpolated at the point uvw as np.array of shape (M,)
        """

        num_points = uvw.shape[0]
        out = np.zeros((num_points, ))

        for ind in range(uvw.shape[0]):
            vtx, wts = interp_weights(self.triangulation, uvw[[ind], :])
            out[ind] = interpolate(self.data, vtx, wts)

        return out

if __name__ == "__main__":
    # Define logging level
    logging.basicConfig(level=logging.WARNING)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from openiam import SystemModel
    except ImportError as err:
        print('Unable to load IAM class module: '+str(err))

    # Create system model
    sm = SystemModel()

    int1 = DataInterpolator(
        name='int1', parent=sm, header_file_dir=os.path.join(
            '..', 'components', 'reservoir',
            'lookuptables', 'Test_2d'),
        data_file='depth.csv')

    # Setup location of interest
    xy = np.array([[37478.0, 48333.0]])
#    xy = np.array([[37478.0, 48333.0], [37450.0, 48200.0], [37300.0, 48400.0], [37200.0, 48200.0]])

    out = int1(xy)
    print(out)
