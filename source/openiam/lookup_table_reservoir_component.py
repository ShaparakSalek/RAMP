# -*- coding: utf-8 -*-
"""
Last modified: August 4th, 2022

Authors: Seth King, Veronika Vasylkivska
"""
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openiam import (SystemModel, ComponentModel,
                         ReservoirDataInterpolator, IAM_DIR)
    from openiam.iam_gridded_observation import interp_weights
except ImportError as err:
    print('Unable to load IAM class module: {}'.format(err))

try:
    from matk.parameter import Parameter
except ImportError as err:
    print('Unable to load matk module: {}'.format(err))


LUTR_OBSERVATIONS = ['pressure', 'CO2saturation']


def find_weights(loc_xyz, triangulation):
    """ Find vertices of the simplex and calculate corresponding weights.

    :param loc_xyz: list of (locX, locY) or (locX, locY, locZ) where
        locX, locY, and locZ are all array-like
    :type loc_xyz: list()

    :param triangulation: Delaunay tesselation in 2- or 3-d,
        triangulation for which weights and vertices will be found.
    :type triangulation: scipy.spatial.Delaunay

    :returns: list of num_points, tri_vertices, tri_weights
    """
    # Determine number of dimensions: 2d or 3d
    num_dims = len(loc_xyz)

    # Determine number of different points
    num_points = len(loc_xyz[0])
    tri_vertices = np.zeros((num_points, num_dims+1), dtype=np.intp)
    tri_weights = np.zeros((num_points, num_dims+1))

    # Extract coordinates common for both dimension scenarios
    locX = loc_xyz[0]
    locY = loc_xyz[1]

    # Find vertices and corresponding weights for each points
    if num_dims == 2:  # if (x, y) type points are provided
        for ind in range(num_points):
            tri_vertices[[ind]], tri_weights[[ind]] = (
                interp_weights(triangulation,
                               np.array([[locX[ind], locY[ind]]])))
    else:   # if (x, y, z) type points are provided
        locZ = loc_xyz[2]
        for ind in range(num_points):
            tri_vertices[[ind]], tri_weights[[ind]] = interp_weights(
                triangulation,
                np.array([[locX[ind], locY[ind], locZ[ind]]]))

    # Setup machine epsilon to replace comparison with zero
    machine_eps = np.finfo(np.float32).eps  # 1.1920929e-07

    # Check whether weights are reasonable; if they are not,
    # then it means that the point at which we need to interpolate
    # is outside the data range.
    # The first if condition checks whether any of the weights
    # are negative up to machine epsilon. The second condition
    # checks whether any of the weights exceed 1 up to machine epsilon.
    if (np.any(tri_weights < -machine_eps)) or (
            np.any(tri_weights > (1+machine_eps))):
        err_msg = ''.join(['Interpolation is requested at the ',
                           'location outside the data domain.'])
        logging.error(err_msg)
        raise ValueError(err_msg)

    return num_points, tri_vertices, tri_weights

class LookupTableReservoir(ComponentModel):
    """
    The Lookup Table Reservoir component model is a reduced order model based on
    interpolation of data from a set of lookup tables. The lookup tables are based
    on the full-physics scale simulations. Each lookup table is determined by a particular
    set of M input model parameters which define a signature of the given set of lookup
    tables.

    In the NRAP-Open-IAM control file, the type name for the Lookup Table Reservoir component
    is ``LookupTableReservoir``. The component's parameters depend on the M input model
    parameters used to create lookup tables data. The minimum and maximum values
    of lookup table parameters determine boundaries of component parameters.
    Moreover, the component parameters values as a set can only be one of the
    combination of values that went into one of the lookup table linked to the
    component.

    In the NRAP-Open-IAM control file a ``FileDirectory`` keyword must be specified. It
    indicates the directory where files with the simulation data for the lookup tables
    are located. A ``TimeFile`` keyword is a name of the .csv file that stores the
    time points (in years) at which the results in the tables are provided.
    If ``TimeFile`` is not specified then, by default, the name of the file
    with time data is assumed to be *time_points.csv*. The time file
    must be located in the directory specified by ``FileDirectory``.

    A ``ParametersFilename`` keyword can also be specified. It defines the
    names and values of lookup table parameters that were used to create
    the given set of lookup tables. Additionally, it lists the names of the *.csv* files
    containing simulation data for each of the lookup table in the set.
    By default, the name of the file with parameters data is assumed to be named
    *parameters_and_filenames.csv*. The parameters file should be in a comma
    separated values format. The first M entries in the first row of the file
    are the names of the lookup table parameters which were varied for different
    realizations; the (M+1)st entry is a word *filename*. Each subsequent row
    of the parameters file contains the M values of the lookup table parameters
    followed by the name of file (lookup table) with the corresponding realization
    data. The provided filename must match with one of the files in the ``FileDirectory``.

    The user should make sure that the information provided in ``ParametersFilename``
    file on parameters and simulation data files is accurate and complete.
    In general, a given parameter of the Lookup Table Reservoir component
    can have any possible name. At the same time the (possibly random) names specified by user
    in the ``ParametersFilename`` file should be the same names that the user would use
    in the control file for the description of the Lookup Table Reservoir
    component parameters. Due to the way the lookup tables are produced,
    each parameter of the reservoir component can only take on certain
    deterministic values. The possible values of a given parameter should be
    listed after the ``values:`` keyword followed by the list in square brackets ('[').
    The weights for each parameter can be specified with the ``weights:`` keyword
    followed by the list of weights for each value in square brackets. The weights
    should sum to 1, otherwise, they will be normalized. If no weights are provided
    all values are assumed to be equally likely to occur.

    There exists an option to sample the data from the lookup tables without direct
    reference to any of the parameters used for creating the tables. User can use an
    auxiliary parameter ``index`` added to the Lookup Table Reservoir component
    to sample data from a particular lookup table file based on its index in the file
    *parameters_and_filenames.csv*. This option allows to use lookup tables
    in the scenarios where the total number of lookup table data files is less than
    the number of all possible combinations of the lookup table parameters.

    Simulation data files (listed in ``ParametersFilename`` file) in a comma
    separated values format contain the reservoir simulation data,
    e.g., pressure and |CO2| saturation, varying over time. The data is used to build
    the Lookup Table Reservoir component output. Each realization file begins with a header line
    that is ignored by the code. Each subsequent row of the file represents a particular
    spatial location. The first and the second columns are the x- and y-coordinates
    of the location, respectively. The subsequent columns contain reservoir simulation data
    at the location defined in the first two columns. The names of the columns
    should represent the data in them and have the form *base.obs.nm_#* where
    *base.obs.nm* is the name of observation as used in the system model and *#*
    is an index of the time point at which the given observation is provided.
    The indexing of the reservoir simulation data should always start with 1
    not with 0. For example, the pressure data at the first time point
    (even if this time point is 0) should always be indexed as *pressure_1*. Further,
    if the column contains pressure data at the second time point,
    its name should be *pressure_2*, and so on. If the column contains saturation data
    at the 12th time point, its name should be *CO2saturation_12*. The order of
    the columns in the lookup table except the first two x and y columns is arbitrary.
    If some reservoir simulation data does not vary in time then the column name
    should indicate it: in any case its name should not contain underscore symbol *_*
    followed by number (time index). For example, column with name #temperature#
    would indicate that the provided temperature data is constant in time.

    The Lookup Table Reservoir component produces the output using interpolation
    in space and time within the spatio-temporal domain defined by the lookup tables
    simulation model setup. Observations from the Lookup Table Reservoir component are:

    * **pressure** [|Pa|] - pressure at top of the reservoir at the wellbore location(s)

    * **CO2saturation** [-] - |CO2| saturation at the top of the reservoir at the
      wellbore location(s).

    Observations *pressure* and *CO2saturation* are mandatory
    for the Lookup Table Reservoir component which means that the linked lookup tables
    should contain the necessary data to produce them. In addition, the component
    can return any other type of observations provided in the lookup tables.
    """
    def __init__(self, name, parent, intr_family=None,
                 locX=None, locY=None, locZ=None,
                 file_directory=None, time_file='time_points.csv',
                 parameter_filename='parameters_and_filenames.csv',
                 interp_2d=True):
        """
        Constructor method of LookupTableReservoir class.

        :param name: name of component model
        :type name: str

        :param parent: the SystemModel object that the component model
            belongs to
        :type parent: SystemModel object

        :param intr_family: name of interpolator family which belongs to the parent
            system model and whose data will be used for reservoir observations
            (e.g., pressure and saturation) calculations
        :type intr_family: str

        :param locX: x-coordinate of the location at which reservoir observations
            (e.g., pressure and |CO2| saturation) are to be calculated.
            By default, it is None, which means that the whole grid data is requested
        :type locX: float or array-like of floats

        :param locY: y-coordinate of the location at which reservoir observations
            (e.g., pressure and |CO2| saturation) are to be calculated.
            By default, it is None, which means that the whole grid data is requested
        :type locY: float or array-like of floats

        ::param locZ: z-coordinate of the location at which reservoir observations
            (e.g., pressure and |CO2| saturation) are to be calculated.
            By default, it is None, which means that 2d interpolation is needed
        :type locZ: float or array-like of floats

        :param file_directory: location (directory) of the reservoir
            simulation data that will be used to create reservoir
            family of interpolators in the case those are not created before
            the reservoir component
        :type file_directory: str

        :param time_file: name of *.csv file to read information about time
            points (in years) at which observations (e.g., pressure and |CO2|
            saturation) in the data lookup tables were calculated
        :type filename: str

        :param parameter_filename: name of *.csv file containing information
            about sets of model parameters and files corresponding
            to the lookup tables
        :type parameter_filename: str

        :param interp_2d: flag variable indicating whether the data provided
            in the lookup tables should be treated as 2d (or 3d) for the purpose
            of interpolation. By default, the value is True for
            the data to be interpolated as 2d. False value means that
            the data is 3d and 3d interpolation will be used.
        :type interp_2d: boolean

        :returns: LookupTableReservoir class object
        """
        # Set up keyword arguments of the 'model' method provided by the system model
        model_kwargs = {'time_point': 365.25}  # default value of 365.25 days

        super().__init__(name, parent, model=self.simulation_model,
                         model_kwargs=model_kwargs)

        # Add type attribute
        self.class_type = 'LookupTableReservoir'

        # Setup attributes related to interpolator family
        self.intr_family = intr_family
        self.linked_to_intr_family = False
        # Dictionary of pairs (index, name)
        self.intr_names = None

        self.file_directory = file_directory
        self.time_file = time_file
        self.parameter_filename = parameter_filename

        # Set up location attributes
        if locX is not None and locY is not None:
            self.locX = np.asarray(locX).flatten()
            self.locY = np.asarray(locY).flatten()
            if len(self.locX) != len(self.locY):
                err_msg = 'Length of locX and locY arrays are not the same.'
                logging.error(err_msg)
                raise ValueError(err_msg)

            if len(self.locX) > 1:
                self.grid_obs_keys = LUTR_OBSERVATIONS

            self.grid_obs_requested = False
        else:
            self.grid_obs_keys = LUTR_OBSERVATIONS
            self.locX = np.array([None])
            self.locY = np.array([None])
            # Add flag indicating whether whole grid data is needed
            self.grid_obs_requested = True

        # Check whether data is 2d
        self.interp_2d = interp_2d

        # Check whether z-coordinates are provided
        if locZ is not None:
            self.locZ = np.asarray(locZ).flatten()
            self.interp_2d = False
            if len(self.locX) != len(self.locZ):
                err_msg = 'Length of locX and locZ arrays are not the same.'
                logging.error(err_msg)
                raise ValueError(err_msg)
        else:
            self.locZ = None

        # Reserve space for additional attributes
        self.num_points = None
        self.tri_vertices = None
        self.tri_weights = None

        # Add one of the default parameters
        # Parameter value can be between minimum, 1,
        # and maximum, number of linked interpolators
        # -2 means that we didn't assign the value yet
        self.add_default_par('index', value=-2)

        # If interpolator family argument is provided, connect to it,
        # otherwise create an 'empty' reservoir component
        if intr_family:
            self.link_to_interpolators()

        # Log creating the component
        debug_msg = 'LookupTableReservoir component created with name {}'.format(self.name)
        logging.debug(debug_msg)

    def add_par(self, name, value=None, vary=True, discrete_vals=None, **kwargs):
        """
        Add parameter to Lookup Table Reservoir component model.

        :param name: name of parameter
        :type name: str

        :param value: Initial parameter value
        :type value: float

        :param vary: flag indicating whether parameter is deterministic or
            stochastic (varied); by default, parameter is assumed to be varied
            and discrete for Lookup Table Reservoir component
        :type vary: boolean

        :param discrete_vals: tuple of two array_like defining discrete values
            and associated probabilities (probabilities are normalized if they
            do not sum up to 1)
        :type discrete_vals: (lst,lst)

        :param kwargs: additional keyword arguments passed to __init__ method
            of Parameter class
        :type kwargs: any

        """
        # Combine dictionaries of deterministic and stochastic parameters
        comb_pars = {**self.pars, **self.deterministic_pars}

        if comb_pars:
            if 'index' in comb_pars and name != 'index':
                err_msg = ''.join([
                    'Parameter index is already present among component {} parameters. ',
                    'No additional parameters can be added.']).format(self.name)
                logging.error(err_msg)
                raise ValueError(err_msg)

            if name == 'index' and len(comb_pars.keys()) >= 2:
                warn_msg = ''.join([
                    'Parameter index is being added to the existing ',
                    'list {lst} of parameters of component {name}. ',
                    'During sampling of lookup table data, value of parameter ',
                    'index will be chosen over values of other parameters ',
                    'added previously to the component {name}.']).format(
                        name=self.name, lst=comb_pars.keys())
                logging.warning(warn_msg)

        if vary is False:  # if parameter to be added is deterministic
            if value is not None:
                if name in self.deterministic_pars:
                    self.deterministic_pars[name] = Parameter(
                        '.'.join([self.name, name]), parent=self._parent,
                        value=value, vary=False)
                else:
                    self.deterministic_pars.__setitem__(name, Parameter(
                        '.'.join([self.name, name]), parent=self._parent,
                        value=value, vary=False))
            else:
                err_msg = ''.join([
                    "Argument 'value' is not provided to the method 'add_par' ",
                    "in the setup of the Lookup Table Reservoir component ",
                    "{cmpnt_name}."]).format(cmpnt_name=self.name)
                logging.error(err_msg)
                raise ValueError(err_msg)
        else:
            if discrete_vals is not None:
                if '.'.join([self.name, name]) in self._parent.pars:
                    self._parent.pars['.'.join([self.name, name])] = Parameter(
                        '.'.join([self.name, name]), parent=self._parent,
                        value=value, vary=vary,
                        discrete_vals=discrete_vals, **kwargs)
                else:
                    self._parent.pars.__setitem__(
                        '.'.join([self.name, name]),
                        Parameter('.'.join([self.name, name]), parent=self._parent,
                                  value=value, vary=vary,
                                  discrete_vals=discrete_vals, **kwargs))
                self.pars[name] = self._parent.pars['.'.join([self.name, name])]
            else:
                err_msg = ''.join([
                    "'discrete_vals' is not among the arguments provided ",
                    "to the method 'add_par' in the setup of the ",
                    "Lookup Table Reservoir component {cmpnt_name}.\n",
                    "For Lookup Table Reservoir Component any ",
                    "parameter (in this case {par_name}) which does not ",
                    "have a fixed value can have only ",
                    "discrete distribution.\n",
                    "Please update the setup of the parameter ",
                    "by using 'vary=False, value=par_value' or ",
                    "'vary=True, discrete_vals=(values_list, weights_list)'.\n",
                    "Check documentation of the component's add_par method ",
                    "for more information."]).format(
                        cmpnt_name=self.name, par_name=name)
                logging.error(err_msg)
                raise ValueError(err_msg)

    def connect_with_system(self, component_data, *args, **kwargs):
        """
        Code to add LookupTableReservoir to system model for control file interface.

        :param component_data: Dictionary of component data including 'Connection'
        :type component_data: dict

        :returns: None
        """
        if 'FileDirectory' in component_data:
            self.file_directory = os.path.join(IAM_DIR,
                                               component_data['FileDirectory'])
        else:
            err_msg = ''.join(['FileDirectory must be provided for ',
                               'LookupTableReservoir Component.'])
            logging.error(err_msg)
            raise IOError(err_msg)

        if 'TimeFile' in component_data:
            self.time_file = component_data['TimeFile']
        else:
            self.time_file = 'time_points.csv'

        if 'ParameterFilename' in component_data:
            self.parameter_filename = component_data['ParameterFilename']
        else:
            self.parameter_filename = 'parameters_and_filenames.csv'

        # TODO Decide what to do for control file interface in the situations
        # when there are several components linked to the same lookup table
        # reservoir component but requesting gridded and scalar observations
        self.locX = np.asarray(component_data['coordx']).flatten()
        self.locY = np.asarray(component_data['coordy']).flatten()
        if len(self.locX) == 1:
            self.grid_obs_keys = []
        # Try getting z-coordinates if they are provided
        try:
            self.locZ = np.asarray(component_data['coordz']).flatten()
        except:
            pass
        self.grid_obs_requested = False

        # If 'reservoir' interpolator family was not created before,
        # we need to create it first, then link it to the reservoir component(s)
        if 'reservoir' not in self._parent.interpolators:
            # Specify the name of interpolator family to be created to make it explicit
            if 'Interpolation2D' in component_data:
                self.interp_2d = component_data['Interpolation2D']
                self.build_and_link_interpolators(
                    intr_family='reservoir',
                    interp_2d=component_data['Interpolation2D'])
            else:
                self.build_and_link_interpolators(intr_family='reservoir')

            for par_name in component_data['Parameters']:
                if not isinstance(component_data['Parameters'][par_name], dict):
                    component_data['Parameters'][par_name] = {
                        'value': component_data['Parameters'][par_name],
                        'vary': False}
                    self.add_par(par_name, **component_data['Parameters'][par_name])
                    continue

                if 'value' in component_data['Parameters'][par_name]:
                    value = component_data['Parameters'][par_name]['value']
                    self.add_par(par_name, value=value, vary=False)
                    debug_msg = ''.join([
                        'LookupTableReservoir parameter {} ',
                        'created with value {}.']).format(par_name, value)
                    logging.debug(debug_msg)
                    continue

                if 'values' in component_data['Parameters'][par_name]:
                    component_data['Parameters'][par_name]['Values'] = (
                        component_data['Parameters'][par_name]['values'])
                values = component_data['Parameters'][par_name]['Values']
                nv = len(values)

                if 'Weights' not in component_data['Parameters'][par_name]:
                    if 'weights' in component_data['Parameters'][par_name]:
                        weights = component_data['Parameters'][par_name]['weights']
                    else:
                        weights = [1.0/nv for v in values]
                else:
                    weights = component_data['Parameters'][par_name]['Weights']

                self.add_par(par_name, value=values[0],
                             vary=True, discrete_vals=(values, weights))
                debug_msg = ''.join([
                    'LookupTableReservoir parameter {name} created with ',
                    'values {values} and weights {weights}.']).format(
                        name=par_name, values=values, weights=weights)
                logging.debug(debug_msg)

        else:
            # If there are several reservoir components
            # that use the same interpolator family,
            # we need to make sure that we linked the parameters of those
            # to the parameters of the first created reservoir component
            init_res_comp = self._parent.interp_creators['reservoir']
            if 'Interpolation2D' in component_data:
                self.interp_2d = component_data['Interpolation2D']
            self.link_to_interpolators(intr_family='reservoir')
            for par_name in init_res_comp.deterministic_pars:
                self.add_par_linked_to_par(
                    par_name, init_res_comp.deterministic_pars[par_name])
            for par_name in init_res_comp.pars:
                self.add_par_linked_to_par(
                    par_name, init_res_comp.pars[par_name])

    def build_and_link_interpolators(self, file_directory=None, time_file=None,
                                     parameter_filename=None, intr_family=None,
                                     default_values=None,
                                     interp_2d=True,
                                     recompute_triangulation=True,
                                     build_on_the_fly=False):
        """
        Builds interpolators for component model to use.

        :param file_directory: Directory that contains the time file,
            the parameters file, and all realization files.
        :type file_directory: str

        :param time_file: filename of *.csv file containing each of the time steps in years.
        :type time_file: str

        :param parameter_filename: filename of *.csv file containing information
            on the parameter names, values and name of file containing
            the corresponding realization data
        :type parameter_filename: str

        :param intr_family: name of interpolator family whose data is used
            for pressure and saturation calculations
        :type intr_family: str

        :param default_values: dictionary of default values of observations
            which are not provided in the lookup tables. By default,
            it is assumed that data provided in the lookup tables is enough,
            thus, by default, the parameter is None.
            The values provided are assumed to be constant for all time steps
            and all spatial points.
        type default_values: dict()

        :param interp_2d: flag variable indicating whether the data provided
            in the lookup tables should be treated as 2d (or 3d) for the purpose
            of interpolation. By default, the value is True for
            the data to be interpolated as 2d. False value means that
            the data is 3d and 3d interpolation will be used.
        :type interp_2d: boolean

        :param recompute_triangulation: flag variable indicating whether
            the triangulation is to be recomputed for each interpolator
            in the linked family. By default, the triangulation is recomputed for
            each interpolator (value=True) although it is highly expected that
            all lookup tables in the linked family will have the same
            triangulation.
        :type recompute_triangulation: boolean

        :param build_on_the_fly: flag variable indicating whether the data
            from the lookup tables corresponding to the linked family
            of interpolators will be read only if needed, e.g. at the first
            call of model method utilizing a particular interpolator. By default,
            all interpolators are created before the model method of the current
            component (value=False).
        :type build_on_the_fly: boolean

        :returns: None
        """
        if file_directory:
            self.file_directory = file_directory
        if time_file:
            self.time_file = time_file
        if parameter_filename:
            self.parameter_filename = parameter_filename
        if intr_family:
            self.intr_family = intr_family
        else:
            self.intr_family = 'reservoir'

        # Read file with signatures of interpolators and names of files
        # with the corresponding data
        signature_data = np.genfromtxt(os.path.join(self.file_directory,
                                                    self.parameter_filename),
                                       delimiter=",", dtype='str')
        # The first element of the first row is supposed to be named 'index'
        if signature_data[0, 0] != 'index':
            err_msg = "".join([
                "The first column of the file {} should contain the indices ",
                "of the lookup tables specified in the last column (filename) and ",
                "be named 'index'. Please update the file by adding ",
                "the corresponding column and try again. Alternatively, ",
                "one can download the updated data set from its original ",
                "location."]).format(self.parameter_filename)
            logging.error(err_msg)
            raise NameError(err_msg)

        # Find number of interpolators = number of tables in the linked data set
        num_interpolators = signature_data.shape[0]-1  # -1 since the first line is a header

        # Check that the indices are integers
        for ind in range(num_interpolators):
            val = int(signature_data[ind+1, 0])
            if str(val) != signature_data[ind+1, 0]:
                err_msg = ''.join([
                    'The index of the lookup table provided ',
                    'in the column 1, row {} of the file {} is not an integer. ',
                    'Please update the content of the file.']).format(
                        ind+1, self.parameter_filename)
                logging.error(err_msg)
                raise TypeError(err_msg)

        # Check that values provided in the first column of signature_data are unique
        if len(np.unique(signature_data[1:, 0])) != num_interpolators:
            # Check whether number of unique data files equals
            # to the number of rows with data in the parameters_and_filenames file
            if len(np.unique(signature_data[1:, -1])) == num_interpolators:
                err_msg = ''.join([
                    'The indices of the lookup tables provided ',
                    'in the first column of the file {} are not unique. ',
                    'Please check the content of the file.']).format(
                        self.parameter_filename)
                logging.error(err_msg)
                raise ValueError(err_msg)
            else:
                err_msg = ''.join([
                    'The indices and filenames of the lookup tables ',
                    'provided in the file {} are not unique. Please check ',
                    'the content of the file.']).format(
                        self.parameter_filename)
                logging.error(err_msg)
                raise ValueError(err_msg)

        # The first row (except the last element) of the file contains
        # names of the parameters+index.
        # The last element of the first row is supposed to be named 'filename'
        # The last column is for the filenames containing data for a particular realization
        par_names = signature_data[0, 1:-1]
        num_pars = len(par_names)

        # Check whether parameter values represent full tensor product
        # of the unique values of the parameters
        if num_pars > 0:
            max_num_interpolators = 1
            for j in range(num_pars):
                num_unique_vals = len(np.unique(signature_data[1:, j+1]))
                max_num_interpolators = max_num_interpolators*num_unique_vals
            if max_num_interpolators > num_interpolators:
                info_msg = ''.join([
                    'Not all parameters combinations are available ',
                    'in the linked lookup table data set. It is recommended to utilize ',
                    'care when sampling the data through parameters. ',
                    'Instead, in most cases sampling of parameter index ',
                    'might be sufficient.'])
                logging.info(info_msg)

        # If more than just index is present
        if num_pars > 0:
            signature = {par_names[j]: float(signature_data[1, j+1]) for j in range(num_pars)}
        else:
            signature = {}

        int1 = self._parent.add_interpolator(
            ReservoirDataInterpolator(
                name='int1',
                parent=self._parent,
                header_file_dir=self.file_directory,
                time_file=self.time_file,
                data_file=signature_data[1, -1],
                index=int(signature_data[1, 0]), # index of the first interpolator
                signature=signature,
                default_values=default_values,
                interp_2d=interp_2d,
                build_on_the_fly=False),
            intr_family=self.intr_family,
            creator=self)
        if not recompute_triangulation:
            triangulation = int1.triangulation
        else:
            triangulation = None

        # Create and add interpolators to the system model
        for ind in range(1, num_interpolators):
            # If more than just index is present
            if num_pars > 0:
                signature = {
                    par_names[j]: float(signature_data[ind+1, j+1]) for j in range(num_pars)}
            else:
                signature = {}
            # Add interpolator
            self._parent.add_interpolator(
                ReservoirDataInterpolator(
                    name='int'+str(ind+1),
                    parent=self._parent,
                    header_file_dir=self.file_directory,
                    time_file=self.time_file,
                    data_file=signature_data[ind+1, -1],
                    index=int(signature_data[ind+1, 0]),
                    signature=signature,
                    default_values=default_values,
                    interp_2d=interp_2d,
                    triangulation=triangulation,
                    build_on_the_fly=build_on_the_fly),
                intr_family=self.intr_family,
                creator=self)

            debug_msg = 'Signature of the created interpolator is {}'.format(signature)
            logging.debug(debug_msg)

        logging.debug('All interpolators are created')
        # Link just created interpolators to the reservoir component
        self.link_to_interpolators()

    def check_input_parameters(self, p):
        """
        Check whether input parameters fall within specified boundaries.

        :param p: input parameters of component model
        :type p: dict
        """
        debug_msgs = ['Checking input parameters...',
                      'Input parameters {}'.format(p)]
        for ind in [0, 1]:
            logging.debug(debug_msgs[ind])

        if not self.linked_to_intr_family:
            err_msg = ''.join(['Application cannot proceed further. ',
                               'Lookup Table Reservoir component is ',
                               'not linked to any interpolator family.'])
            logging.error(err_msg)
            raise LinkError(err_msg)

        # Save interpolators names
        if self.intr_names is None:
            self.intr_names = {}
        for intr_nm, intr_obj in self._parent.interpolators[self.intr_family].items():
            self.intr_names[intr_obj.index] = intr_nm

        # For lookup table reservoir component we need to make sure that the
        # signature created with input parameters coincide with the signature
        # of one of the interpolators used by component

        # Create signature based on default parameter values
        param_signature = {k: v.value for k, v in self.default_pars.items()}

        # Check whether there are any parameters not belonging to the signature
        for key in p:
            if key not in param_signature:
                msg = ''.join([
                    'Parameter {key} not recognized as ',
                    'a LookupTableReservoir input parameter.']).format(key=key)
                logging.warning(msg)

        # Update default signature with values of input parameters p
        param_signature.update(p)

        # Extract index from updated dictionary
        index = int(param_signature.pop('index'))
        if index != -2:
            if index not in self.intr_names:
                err_msg = ''.join([
                    'Value {} of index parameter does not correspond ',
                    'to any of the linked interpolators.']).format(index)
                logging.error(err_msg)
                raise ValueError(err_msg)
        else:
            # Check for the same signature among all connected interpolators
            signature_found = False
            for interpr in self._parent.interpolators[self.intr_family].values():
                # Compare signature of interpolator with the provided input parameters
                if interpr.signature == param_signature:
                    signature_found = True
                    break

            if not signature_found:
                err_msg = ''.join([
                    'Signature of input parameters do not coincide with ',
                    'signatures of connected interpolators {}.']).format(param_signature)
                logging.error(err_msg)
                raise ValueError(err_msg)

    def link_to_interpolators(self, intr_family=None):
        """
        Link the component to the interpolators to be used.

        The method links the component to the interpolators to be used, check
        interpolation weights, and sets the default parameter values.

        :param intr_family: name of interpolation family to be used.
        :type intr_family: str

        :returns: None
        """
        # Check whether interpolators family intr_family exists
        if intr_family:
            self.intr_family = intr_family
        else:
            intr_family = self.intr_family

        if intr_family in self._parent.interpolators:
            # Save interpolators names
            if self.intr_names is None:
                self.intr_names = {}
            for intr_nm, intr_obj in self._parent.interpolators[intr_family].items():
                self.intr_names[intr_obj.index] = intr_nm

            # Get the first interpolator in the family
            interpr = list(self._parent.interpolators[self.intr_family].values())[0]

            # Get type of data that interpolator is handling and compare with
            # the setup of reservoir component
            interpr_interp_2d = interpr.interp_2d

            options = {True: '2d', False: '3d'}
            if interpr_interp_2d != self.interp_2d:
                msg = ''.join([
                    'Interpolator is linked to {} data but reservoir is setup ',
                    'as being linked to {} data. Reservoir setup will be ',
                    'changed to the one similar to the interpolator.']).format(
                        options[interpr_interp_2d], options[self.interp_2d])
                logging.warning(msg)

                self.interp_2d = interpr_interp_2d

            if not self.grid_obs_requested:
                if self.interp_2d:
                    self.num_points, self.tri_vertices, self.tri_weights = find_weights(
                        (self.locX, self.locY), interpr.triangulation)
                else:
                    self.num_points, self.tri_vertices, self.tri_weights = find_weights(
                        (self.locX, self.locY, self.locZ), interpr.triangulation)

            # The default values of the component parameters are defined
            # by the signature of the first interpolator
            for key in list(interpr.signature.keys()):
                # Set default parameters of the component model
                self.add_default_par(key, value=interpr.signature[key])

            # Setup link flag
            self.linked_to_intr_family = True

        else:
            # Show useful message
            err_msg = ''.join(['Attempt to link reservoir component {} ',
                               'to the family of interpolators failed: ',
                               'family of interpolators {} does ',
                               'not exist.']).format(self.name, intr_family)
            logging.error(err_msg)
            raise KeyError(err_msg)

    def simulation_model(self, p, time_point=365.35,
                         locX=None, locY=None, locZ=None):
        """
        Return pressure and |CO2| saturation at the bottom of leaking well.

        :param p: input parameters of reservoir model determined
            by the signature of interpolators
        :type p: dict

        :param time_point: time point (in days) for which the pressure and
            saturation are to be calculated; by default, its value is 365.25 days
            (1 year in days)
        :type time_point: float

        :param locX: x-coordinate of the location at which reservoir observations
            are to be calculated.
        :type locX: float or array-like of floats

        :param locY: y-coordinate of the location at which reservoir observations
            are to be calculated.
        :type locY: float or array-like of floats

        ::param locZ: z-coordinate of the location at which reservoir observations
            are to be calculated.
        :type locZ: float or array-like of floats

        :returns: out - dictionary of observations of lookup table reservoir
            component model; keys: ['pressure','CO2saturation']
        """
        # Obtain the default values of the parameters from dictionary of default parameters
        actual_p = {k: v.value for k, v in self.default_pars.items()}

        # Update default values of parameters with the provided ones
        actual_p.update(p)

        # Determine initial time point for a given simulation
        if self._parent.time_array is not None:
            init_time_point = self._parent.time_array[0]
        else:
            init_time_point = self._parent.time_point

        # Check whether we deal with initial time points
        if time_point == init_time_point:
            # Check whether locations are provided as keyword arguments
            loc_updated = False
            if locX is not None and locY is not None:
                # Get the provided locations reformatted
                prov_locX = np.array([locX]).flatten()
                prov_locY = np.array([locY]).flatten()

                # If single point is provided update type of observations
                if len(prov_locX) == 1:
                    self.grid_obs_keys = []

                # Check consistency of lengths
                if len(prov_locX) != len(prov_locY):
                    err_msg = ''.join([
                        'Length of locX and locY arrays provided ',
                        'to the model method are not the same.'])
                    logging.error(err_msg)
                    raise ValueError(err_msg)

                # Check if z-coordinates are also provided
                if locZ is not None:
                    prov_locZ = np.array([locZ]).flatten()
                    if len(prov_locX) != len(prov_locZ):
                        err_msg = ''.join([
                            'Length of locX and locZ arrays provided ',
                            'to the model method are not the same.'])
                        logging.error(err_msg)
                        raise ValueError(err_msg)

                    if (len(prov_locZ) != len(self.locZ)) or (
                            np.not_equal(prov_locZ, self.locZ).any()):
                        # Update z-coordinates
                        self.locZ = prov_locZ
                        loc_updated = True

                # Check whether the previously defined locations are the same
                # as the new ones
                if (len(prov_locX) != len(self.locX)) or (
                        np.not_equal(prov_locX, self.locX).any() or np.not_equal(
                            prov_locY, self.locY).any()):
                    # Update locations
                    self.locX = prov_locX
                    self.locY = prov_locY
                    loc_updated = True

                if loc_updated:
                    self.grid_obs_requested = False
                    # Find first linked interpolator
                    interpr = list(self._parent.interpolators[self.intr_family].values())[0]

                    # Recalculate the weights of the simplex and corresponding weights
                    if self.locZ is None:
                        self.num_points, self.tri_vertices, self.tri_weights = find_weights(
                            (self.locX, self.locY), interpr.triangulation)
                    else:
                        self.num_points, self.tri_vertices, self.tri_weights = find_weights(
                            (self.locX, self.locY, self.locZ), interpr.triangulation)

        # Initialize dictionary of leakage rates
        out = dict()

        # Extract index from updated dictionary
        index = int(actual_p.pop('index'))
        if index != -2:   # if we sample datafile/signature indices
            interpr_nm = self.intr_names[index]
            interpr = self._parent.interpolators[self.intr_family][interpr_nm]
        else:
            for interpr_obj in self._parent.interpolators[self.intr_family].values():
                # Compare signature of interpolator with the provided input parameters
                if interpr_obj.signature == actual_p:
                    interpr = interpr_obj
                    break

        if self.grid_obs_requested:
            interpr_out = interpr(time_point)
            for nm in interpr_out:
                out[nm] = interpr_out[nm]
        else:
            if self.num_points == 1:
                interpr_out = interpr(
                    time_point, self.tri_vertices, self.tri_weights)
                # Create dictionary of output
                for nm in interpr_out:
                    out[nm] = interpr_out[nm][0]
            else:
                # Interpolate over all requested points
                for ind in range(self.num_points):
                    interpr_out = interpr(time_point,
                                          self.tri_vertices[[ind]],
                                          self.tri_weights[[ind]])
                    for nm in interpr_out:
                        if not nm in out:
                            out[nm] = np.zeros(self.num_points)
                        out[nm][ind] = interpr_out[nm][0]

            debug_msg = '{} model call output {}'.format(self.name, out)
            logging.debug(debug_msg)

        # Return dictionary of outputs
        return out

    def reset(self):
        pass


class LinkError(Exception):
    """ Class based on Exception class.

    Provides meaningful name to the error for the scenarios when the lookup
    table reservoir component is not linked to any interpolator family.
    """

if __name__ == "__main__":

    logging.basicConfig(level=logging.WARNING)
    test_case = 1
    if test_case == 1:
        data_set_fldr = 'Kimb_54_sims'
        loc_x = 37478.0
        loc_y = 48333.0
        num_years = 50
    elif test_case == 2:
        data_set_fldr = 'Test_2_sims'
        loc_x = 37478.0
        loc_y = 48233.0
#        loc_x = [37478.0, 37478.0]
#        loc_y = [48233.0, 48233.0]
        num_years = 50

    # Define keyword arguments of the system model
    time_array = 365.25*np.arange(0.0, num_years+1)
    sm_model_kwargs = {'time_array': time_array} # time is given in days

    # Create system model
    sm = SystemModel(model_kwargs=sm_model_kwargs)

    # Read file with signatures of interpolators and names of files with the corresponding data
    sign_data = np.genfromtxt(
        os.path.join('..', 'components', 'reservoir', 'lookuptables',
                     data_set_fldr, 'parameters_and_filenames.csv'),
        delimiter=",", dtype='str')

    # The first row (except the last element) of the file contains names of the parameters.
    # The last element of the first row is supposed to be named 'filename'
    # The last column is for the filenames containing data for a particular realization
    par_names = sign_data[0, 1:-1]
    num_pars = len(par_names)

    # Add reservoir component
    ltres = sm.add_component_model_object(
        LookupTableReservoir(name='ltres', parent=sm, locX=loc_x, locY=loc_y))
    file_dir = os.path.join(
        '..', 'components', 'reservoir', 'lookuptables', data_set_fldr)
    ltres.build_and_link_interpolators(
        file_directory=file_dir, intr_family='reservoir',
        default_values={'salinity': 0.1, 'temperature': 50.0},
        recompute_triangulation=False, build_on_the_fly=True)

    # For the Lookup table reservoir component, either set of parameters used
    # for the simulation or only parameter index should be used
    # Add parameters of reservoir component model
    for j in range(num_pars):
        # For the parameters values use arbitrary line data (in this case
        # from the second line) from signature_file
        ltres.add_par(par_names[j], value=float(sign_data[1, j+1]), vary=False)

    # # Or use only index as shown below
    # ltres.add_par('index', value=2, vary=False)

    # Add observations of reservoir component model
    ltres.add_obs('pressure')
    ltres.add_obs('CO2saturation')
    if test_case == 2:
        ltres.add_obs('area')

    # Run system model using current values of its parameters
    sm.forward()

    print('------------------------------------------------------------------')
    print('                  Forward method illustration ')
    print('------------------------------------------------------------------')

    # Collect pressure and saturation observations
    pressure = sm.collect_observations_as_time_series(ltres, 'pressure')
    saturation = sm.collect_observations_as_time_series(ltres, 'CO2saturation')
    print('Pressure:', pressure, sep='\n')
    print('CO2saturation:', saturation, sep='\n')
    if test_case == 4:
        area = sm.collect_observations_as_time_series(ltres, 'area')
        print('Area:', area, sep='\n')

    # Plot results
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121)
    plt.plot(time_array/365.25, pressure/1.0e+6, color="maroon", linewidth=1)
    plt.xlabel('Time, t (years)', fontsize=14)
    plt.ylabel('Pressure, P (MPa)', fontsize=14)
    plt.title('Pressure: leaking well', fontsize=18)
    plt.tight_layout()
    plt.tick_params(labelsize=12)
    plt.xlim([0, 50])
    ax.get_yaxis().set_label_coords(-0.12, 0.5)

    ax = fig.add_subplot(122)
    plt.plot(time_array/365.25, saturation, color="green", linewidth=1)
    plt.xlabel('Time, t (years)', fontsize=14)
    plt.ylabel('Saturation, S (-)', fontsize=14)
    plt.title(r'CO$_2$ saturation: leaking well', fontsize=18)
    plt.tight_layout()
    plt.tick_params(labelsize=12)
    plt.xlim([0, 50])
    ax.get_yaxis().set_label_coords(-0.12, 0.5)
