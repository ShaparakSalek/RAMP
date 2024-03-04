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
from ramp.configuration.configuration import POINTS_CLASS, BaseConfiguration


class SeismicSurveyConfiguration(BaseConfiguration):
    def __init__(self, sources, receivers, name='Unnamed', create_arrays=False,
                 array_creator=None, array_creator_kwargs=None):
        """
        SeismicSurveyConfiguration helps to describe the seismic survey configuration.

        Parameters
        ----------
        sources : numpy.array of shape (nsources, 3)
            Array of x, y, z coordinates of sources of a given seismic survey
        receivers : numpy.array of shape (nreceivers, 3)
            Array of x, y, z coordinates of sources of a given seismic survey
        name : string, optional
            Name of seismic survey
        create_arrays : boolean, optional
            Flag indicating whether arrays consisting of sources and receivers
            of the given configuration are to be created; default is False
        array_creator : method name, optional
            Method to be used to create arrays of the configuration sources
            and receivers; default is None
        array_creator_kwargs : dict, optional
            Dictionary containing arguments of the method defined in array_creator

        Returns
        -------
        Instance of SeismicSurveyConfiguration class containing information
        about the survey's sources and receivers
        """
        super().__init__(sources=sources, receivers=receivers, name=name,
                         sc_name='source', rc_name='receiver')

        if create_arrays:
            self.array_creator = array_creator
            if array_creator_kwargs is not None:
                self.array_creator_kwargs = array_creator_kwargs
            else:
                self.array_creator_kwargs = dict()
            self.create_arrays(self.array_creator, **self.array_creator_kwargs)
        else:
            self.array_creator = None
            self._arrays = None
            self.num_arrays = 0

    @property
    def arrays(self):
        return self._arrays

    @arrays.setter
    def arrays(self, new_arrays):
        if isinstance(new_arrays, dict):
            self._arrays = new_arrays
            self.num_arrays = len(new_arrays.keys())
        else:
            print("Error: Expected dictionary for new_arrays.")
            return

    def create_arrays(self, array_creator, **array_creator_kwargs):
        """
        Create set of arrays according to the predetermined setup.

        Parameters
        ----------
        array_creator : method name, optional
            Method to be used to create arrays of the configuration sources
            and receivers; default is None
        array_creator_kwargs : dict, optional
            Dictionary containing arguments of the method defined in array_creator

        Returns
        -------
        None.

        """
        self.num_arrays, self._arrays = array_creator(**array_creator_kwargs)

    def create_survey_configuration(self, name, source_indices, receiver_indices):
        """
        Produce new SeismicSurveyConfiguration instance based on subset of sources and
        receivers

        Parameters
        ----------
        name : string
            Name of a new seismic survey configuration
        source_indices : list
            Indices of sources that will be used for new configuration.
        receiver_indices : list
            Indices of receivers that will be used for new configuration.

        Returns
        -------
        None.

        """
        # Create new empty instance of configuration
        new_configuration = SeismicSurveyConfiguration(
            sources=None, receivers=None, name=name)

        # Get sources coordinates
        source_coords = self._sources.coordinates[source_indices, :]
        new_configuration._sources = self.create_point_set(
            'sources', source_coords, new_configuration,
            POINTS_CLASS[self.source_class])

        # Get receivers coordinates
        receiver_coords = self._receivers.coordinates[receiver_indices, :]
        new_configuration._receivers = self.create_point_set(
            'receivers', receiver_coords, new_configuration,
            POINTS_CLASS[self.receiver_class])

        return new_configuration


def elementary_array_creator(num_sources=1, num_receivers=10, receiver_step=1,
                             first_receiver_index=0):
    """
    Create a dictionary of seismic arrays.

    Parameters
    ----------
    num_sources : int
        Number of sources in the original seismic survey configuration.
    num_receivers : int
        Number of receivers in the original seismic survey configuration.
    receiver_step : int
        Integer step between indices of consecutive receivers.
    first_receiver_index : int
        Index of the receiver to be used as first for each array.

    Returns
    -------
    num_arrays : int
        Number of produced arrays
    produced_arrays : dict
        Dictionary containing information about created arrays. Possible keys are
        integers from 0 to (num_arrays-1). Each key correspond to one of the
        created arrays and is represented as a dictionary containing
        the following information:
        'source' : int
            Index of source for a given array
        'receivers' : list of int
            List of receivers indices
        'num_sources' : int
            Number of sources in a given array. In this case, it's always 1.
        'num_receivers' : int
            Number of receivers in an given array

    """
    # Default values if no arrays are constructed
    produced_arrays = None
    num_arrays = 0

    if num_sources > 0 and num_receivers > 0:
        if first_receiver_index < num_receivers:
            produced_arrays = {}
            for ind in range(num_sources):
                receiver_indices = list(range(first_receiver_index, num_receivers,
                                              receiver_step))
                produced_arrays[ind] = {
                    'source': ind,
                    'receivers': receiver_indices,
                    'num_sources': 1,
                    'num_receivers': len(receiver_indices)}

            num_arrays = num_sources
        else:
            warn_msg = ''.join(['Arrays cannot be created for the selected values ',
                                'of input arguments. Index of the first receiver ',
                                'is larger than the number of receivers.'])
            logging.warn(warn_msg)
    else:
        warn_msg = ''.join(['Arrays cannot be created for the selected values ',
                            'of input arguments. num_sources and/or num_receivers ',
                            'have invalid values.'])
        logging.warn(warn_msg)

    return num_arrays, produced_arrays


def multi_step_array_creator(num_sources=9, num_receivers=101, steps=None,
                             array_min_receivers=3):
    """
    Create a dictionary of arrays.

    Created arrays will be comprised of all arrays with all the sources
    separated by specified steps.

    Parameters
    ----------
    num_sources : int
        Number of sources in the original seismic survey configuration.
    num_receivers : int
        Number of receivers in the original seismic survey configuration.
    steps : list
        Integer step(s) between indices of consecutive receivers provided as list
    array_min_receivers : int
        Minimum number of receivers to be present in the created arrays

    Returns
    -------
    num_arrays : int
        Number of produced arrays
    produced_arrays : dict
        Dictionary containing information about created arrays. Possible keys are
        integers from 0 to (num_arrays-1). Each key correspond to one of the
        created arrays and is represented as a dictionary containing
        the following information:
        'source' : int
            Index of source for a given array
        'receivers' : list of int
            List of receivers indices
        'num_sources' : int
            Number of sources in a given array. In this case, it's always 1.
        'num_receivers' : int
            Number of receivers in an given array

    """
    # Default values if no arrays are constructed
    produced_arrays = None
    num_arrays = 0

    if num_sources > 0 and num_receivers > 0:
        if steps is None:
            steps = [1]
        elif isinstance(steps, list):
            flag = np.all(np.array(steps) < num_receivers)
            if not flag:
                warn_msg = ''.join(['Some or all values in the provided argument ',
                                   'steps are too large. No arrays will be produced'])
                logging.warn(err_msg)
                return num_arrays, produced_arrays
        else:
            warn_msg = 'Argument steps is wrong type. No arrays will be produced.'
            logging.warn(err_msg)
            return num_arrays, produced_arrays

        a_ind = 0
        produced_arrays = {}
        for ind in range(num_sources):
            for step in steps:
                for rind in range(0, step):
                    receiver_indices = list(range(rind, num_receivers, step))
                    array_num_receivers = len(receiver_indices)
                    if array_num_receivers >= array_min_receivers:
                        produced_arrays[a_ind] = {
                            'source': ind,
                            'receivers': receiver_indices,
                            'num_sources': 1,
                            'num_receivers': array_num_receivers}
                        a_ind = a_ind + 1
        num_arrays = len(produced_arrays.keys())
        print('Check', num_arrays==a_ind)

    else:
        warn_msg = ''.join(['Arrays cannot be created for the selected values ',
                            'of input arguments. num_sources and/or num_receivers ',
                            'have invalid values.'])
        logging.warn(warn_msg)

    return num_arrays, produced_arrays


def density_based_array_creator(num_sources=9, num_receivers=101,
                                array_min_receivers=3, density='dense'):
    """
    Create a dictionary of arrays.

    Created arrays will be comprised of all arrays with all the sources
    separated by specified steps dependent on the provided density.

    Parameters
    ----------
    num_sources : int, optional
        Number of sources in the original seismic survey configuration
    num_receivers : int, optional
        Number of receivers in the original seismic survey configuration
    array_min_receivers : int, optional
        Minimum number of receivers to be present in the created arrays
    density : str, optional
        Type of spacing between the receivers of the arrays

    Returns
    -------
    num_arrays : int
        Number of produced arrays
    produced_arrays : dict
        Dictionary containing information about created arrays. Possible keys are
        integers from 0 to (num_arrays-1). Each key correspond to one of the
        created arrays and is represented as a dictionary containing
        the following information:
        'source' : int
            Index of source for a given array
        'receivers' : list of int
            List of receivers indices
        'num_sources' : int
            Number of sources in a given array. In this case, it's always 1.
        'num_receivers' : int
            Number of receivers in an given array

    """
    # Default values if no arrays are constructed
    produced_arrays = None
    num_arrays = 0

    density_to_steps = {'dense': [1],
                        'medium': [2, 3, 4],
                        'sparse': [5, 6, 7, 8]}
    # Check whether the density argument has the right value
    if density in ['dense', 'medium', 'sparse']:
        steps = density_to_steps[density]
        num_arrays, produced_arrays = multi_step_array_creator(
            num_sources=num_sources, num_receivers=num_receivers,
            steps=steps, array_min_receivers=array_min_receivers)
        return num_arrays, produced_arrays
    else:
        warn_msg = 'Argument density has an unrecognized value. No arrays will be produced.'
        logging.warn(warn_msg)
        # Default values are returned
        return num_arrays, produced_arrays


def five_n_receivers_array_creator(source_coords=None, receiver_coords=None):
    """
    Create a dictionary of arrays.

    Created arrays will have number of receivers in multiple of five.

    Parameters
    ----------
    source_coords : numpy.ndarray
        x, y, z coordinates of the sources. Shape of array is (ns, 3) where
        ns - number of sources
    receiver_coords : numpy.ndarray
        x, y, z coordinates of the receivers. Shape of array is (nr, 3) where
        nr - number of receivers

    Returns
    -------
    num_arrays : int
        Number of produced arrays
    produced_arrays : dict
        Dictionary containing information about created arrays. Possible keys are
        integers from 0 to (num_arrays-1). Each key correspond to one of the
        created arrays and is represented as a dictionary containing
        the following information:
        'source' : int
            Index of source for a given array
        'receivers' : list of int
            List of receivers indices
        'num_sources' : int
            Number of sources in a given array. In this case, it's always 1.
        'num_receivers' : int
            Number of receivers in an given array

    """
    # Default values if no arrays are constructed
    produced_arrays = None
    num_arrays = 0

    if source_coords is not None and receiver_coords is not None:
        # Determine number of sources and receivers
        num_sources = len(source_coords)
        num_receivers = len(receiver_coords)

        source_indices = num_sources*[None]
        for ind in range(num_sources):
            source_indices[ind] = np.where(
                receiver_coords[:, 0] == source_coords[ind, 0])[0][0]

        # First, create right arrays for sources for which they can be created
        produced_arrays = {}
        array_ind = 0
        for sind in range(num_sources):
            max_num_5_multiples = (num_receivers - source_indices[sind])//5
            # print(sind, max_num_5_multiples)
            for offset in range(1, max_num_5_multiples+1):
                for n in range(1, max_num_5_multiples+1):  # n - number of multiples of 5
                    if (source_indices[sind] + offset*n*5) < num_receivers:  # add array
                        produced_arrays[array_ind] = {
                            'source': sind,
                            'receivers': list(range(
                                source_indices[sind] + offset,
                                source_indices[sind] + offset*n*5 + 1, offset)),
                            'num_receivers': 5*n}
                        array_ind = array_ind + 1
                    else:
                        break

        # Second, create left arrays
        for sind in range(num_sources):
            max_num_5_multiples = source_indices[sind]//5
            for offset in range(1, max_num_5_multiples+1):
                for n in range(1, max_num_5_multiples+1):  # n - number of multiples of 5

                    if (source_indices[sind] - offset*n*5) >= 0:  # add array
                        produced_arrays[array_ind] = {
                            'source': sind,
                            'receivers': list(range(
                                source_indices[sind] - offset*n*5,
                                source_indices[sind] - offset + 1,
                                offset)),
                            'num_receivers': 5*n}
                        array_ind = array_ind + 1
                    else:
                        break
        num_arrays = array_ind

    else:
        warn_msg = ''.join(['Arrays cannot be created for the selected values ',
                            'of input arguments.'])
        logging.warn(warn_msg)

    return num_arrays, produced_arrays


def create_sources_and_receivers():
    """
    Auxiliary method to produce arrays of sources and receivers

    Returns
    -------
    None.

    """
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

    return sources, receivers


def plot_arrays(conf_arrays, source_coords, receiver_coords, num_plots=1,
                num_arrays_per_plot=100, xlims=[4000, 8000],
                markersize=3):
    """
    Auxiliary method to plot all arrays on several plots

    Parameters
    ----------
    arrays : TYPE
        DESCRIPTION.
    num_plots : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    plt.close('all')
    for plot_ind in range(num_plots):
        fig = plt.figure(figsize=(8, 10))
        ax = fig.add_subplot(111)
        init_ind = plot_ind*num_arrays_per_plot
        last_ind = (plot_ind+1)*num_arrays_per_plot-1
        for ind in range(init_ind, last_ind+1):
            if ind in conf_arrays:
                sind = conf_arrays[ind]['source']
                rind = conf_arrays[ind]['receivers']
                nr = conf_arrays[ind]['num_receivers'] # number of receivers in the array
                if ind == init_ind:
                    ax.plot(receiver_coords[rind, 0], nr*[ind+1], 'sb',
                            label='receivers', markersize=markersize)
                    ax.plot([source_coords[sind, 0]], [ind+1], 'or',
                            label='sources', markersize=markersize)
                else:
                    ax.plot(receiver_coords[rind, 0], nr*[ind+1], 'sb',
                            label=None, markersize=markersize)
                    ax.plot([source_coords[sind, 0]], [ind+1], 'or',
                            label=None, markersize=markersize)

        ax.set_xlim(xlims[0], xlims[1])
        ax.set_xlabel('x, [m]')
        if num_plots == 1:
            ax.set_ylim(0.5, last_ind+1.5)
        else:
            ax.set_ylim(init_ind, last_ind+2)
        ax.set_ylabel('array index')
        ax.invert_yaxis()
        ax.legend()


def test_seismic_configuration1():
    """
    Test work of SeismicSurveyConfiguration class.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    # Create survey configuration with defined coordinates
    configuration = SeismicSurveyConfiguration(sources, receivers, name='Test Survey')

    # Plot survey to see if it is created correctly
    configuration.plot_configuration()


def test_seismic_configuration2():
    """
    Test work of SeismicSurveyConfiguration class.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    # Create survey configuration with defined coordinates
    configuration = SeismicSurveyConfiguration(
        sources, receivers, name='Test Survey')
    array_creator_kwargs = {'num_sources': configuration.num_sources,
                            'num_receivers': configuration.num_receivers}
    configuration.create_arrays(elementary_array_creator,
                                **array_creator_kwargs)
    print('Number of created arrays:', configuration.num_arrays)

    # Plot arrays
    plot_arrays(configuration.arrays, sources, receivers, num_plots=1,
                num_arrays_per_plot=configuration.num_arrays, xlims=[3900, 8100],
                markersize=5)


def test_seismic_configuration3():
    """
    Test work of SeismicSurveyConfiguration class.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    # Create survey configuration with defined coordinates
    array_creator_kwargs = {'source_coords': sources,
                            'receiver_coords': receivers}
    configuration = SeismicSurveyConfiguration(
        sources, receivers, name='Test Survey', create_arrays=True,
        array_creator=five_n_receivers_array_creator,
        array_creator_kwargs=array_creator_kwargs)

    for ind in configuration.arrays:
        print(configuration.arrays[ind]['source'],
              configuration.arrays[ind]['receivers'],
              configuration.arrays[ind]['num_receivers'])
    print('Number of created arrays:', configuration.num_arrays)

    # Plot arrays
    plot_arrays(configuration.arrays, sources, receivers, num_plots=5,
                num_arrays_per_plot=100, xlims=[3900, 8100],
                markersize=3)


def test_five_n_receivers_array_creator():
    """
    Test work of five_n_receivers_array_creator method.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    num_arrays, config_arrays = five_n_receivers_array_creator(sources, receivers)

    print('Number of created arrays:', num_arrays)

    # Plot arrays
    plot_arrays(config_arrays, sources, receivers, num_plots=5,
                num_arrays_per_plot=100, xlims=[3900, 8100],
                markersize=3)


def test_multi_step_array_creator1():
    """
    Test work of multi_step_array_creator method.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    num_arrays, config_arrays = multi_step_array_creator(
        num_sources=9, num_receivers=101, steps=[1])

    print('Number of created arrays:', num_arrays)

    # Plot arrays
    plot_arrays(config_arrays, sources, receivers, num_plots=1,
                num_arrays_per_plot=num_arrays, xlims=[3900, 8100],
                markersize=3)

def test_multi_step_array_creator2():
    """
    Test work of multi_step_array_creator method.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    num_arrays, config_arrays = multi_step_array_creator(
        num_sources=9, num_receivers=101, steps=[2, 3, 4])

    print('Number of created arrays:', num_arrays)

    # Plot arrays
    plot_arrays(config_arrays, sources, receivers, num_plots=1,
                num_arrays_per_plot=num_arrays, xlims=[3900, 8100],
                markersize=3)


def test_multi_step_array_creator3():
    """
    Test work of multi_step_array_creator method.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    num_arrays, config_arrays = multi_step_array_creator(
        num_sources=9, num_receivers=101, steps=[5, 6, 7, 8])

    print('Number of created arrays:', num_arrays)

    # Plot arrays
    plot_arrays(config_arrays, sources, receivers, num_plots=5,
                num_arrays_per_plot=52, xlims=[3900, 8100],
                markersize=3)


def test_density_based_array_creator():
    """
    Test work of multi_step_array_creator method.

    Returns
    -------
    None.

    """
    # Get sources and receivers
    sources, receivers = create_sources_and_receivers()

    num_arrays, config_arrays = density_based_array_creator(
        num_sources=9, num_receivers=101, density='medium')

    print('Number of created arrays:', num_arrays)

    # Plot arrays
    plot_arrays(config_arrays, sources, receivers, num_plots=1,
                num_arrays_per_plot=num_arrays, xlims=[3900, 8100],
                markersize=3)


if __name__ == "__main__":

    test_case = 8
    available_tests = {1: test_seismic_configuration1,
                       2: test_seismic_configuration2,
                       3: test_seismic_configuration3,
                       4: test_five_n_receivers_array_creator,
                       5: test_multi_step_array_creator1,
                       6: test_multi_step_array_creator2,
                       7: test_multi_step_array_creator3,
                       8: test_density_based_array_creator}
    available_tests[test_case]()
