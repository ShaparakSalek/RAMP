"""
This module contains SystemModel and ComponentModel classes.

You can define your own component model by deriving from ComponentModel class.

@author: Dylan Harp, Veronika Vasylkivska

"""
import os
import sys
import logging
from re import sub

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matk import matk
from matk.lmfit.asteval import Interpreter
from matk.parameter import Parameter
from matk.observation import Observation
from matk.ordereddict import OrderedDict
from openiam.iam_sampleset import SampleSet

try:
    from openiam.iam_gridded_observation import GriddedObservation
except ImportError:
    from .iam_gridded_observation import GriddedObservation

IAM_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))


class SystemModel(matk):
    """ IAM System Model class. """
    def __init__(self, model_kwargs=None):
        """
        Constructor method of SystemModel class.

        :param model_kwargs: dictionary of additional optional
            keyword arguments of the system model; by default, model_kwargs is
            None.
            Possible model_kwargs are 'time_array' and 'time_point'.
            time_array and/or time_point data is assumed to be given in days.
            time_array is assumed to be numpy.array type and to have
            at least two elements. Otherwise an error message is shown.
        :type model_kwargs: dict

        :returns: SystemModel class object
        """
        # Inherent matk class .__init__ method
        super().__init__(model_args=None, model_kwargs={}, cpus=1,
                         workdir_base=None, workdir=None, results_file=None,
                         seed=None, sample_size=10, hosts={})

        # Set up a flag indicating whether more than one point is given.
        # By default single time point is assumed.
        self.single_time_point_flag = True

        # Add job number keyword argument of system_model method
        self.model_kwargs['job_number'] = None

        # Check whether any keyword arguments are provided.
        if model_kwargs is not None:
            # If time array is provided as a keyword argument, set it up as an
            # attribute of the system model.
            if 'time_array' in model_kwargs:
                self.single_time_point_flag = False
                self.time_array = model_kwargs['time_array']
                if self.time_array.size < 2:
                    print('\nERROR: Time array argument of the system model'+
                          ' should have more than one element.\n')
                    sys.exit()
            elif 'time_point' in model_kwargs: # if time point is specified
                # Set up flag that single time point is given
                self.single_time_point_flag = True
                self.time_array = None
                self.time_point = model_kwargs['time_point']
                self.time_step = min(self.time_point, 365.25)  # in days
        else:
            # If time point and step are not provided set up default ones.
            self.time_point = 365.25   # default time point is 1 year in days
            self.time_step = 365.25    # default time step in days
            self.time_array = None

        self.component_models = OrderedDict()
        self.interpolators = OrderedDict()
        self.interp_creators = OrderedDict()
        self.observation2components = OrderedDict()
        # This dictionary would help to save and reuse the information about
        # the models loaded for components
        # the keys are component type and values are links to components of that type
        # that keep the original references to the ML models
        self.ml_models_components = OrderedDict()

        # This dictionary would help to save and reuse the information about
        # the libraries (e.g. dll) loaded for components
        # the keys are component type and values are links to components of that type
        # that keep the original references to the libraries
        self.libraries_components = OrderedDict()

        # Added attribute obs_base_names to keep track on what the observations
        # will have a time index added to the end of name; it helps to deal with
        # these observations in system_model method.
        self.obs_base_names = list()

        # This flag indicates whether simulated values are associated with current parameters
        self._current = False
        self.model = self.system_model

    def add_component_model_object(self, cm_object):
        """
        Add component model object to system model.

        :param cm_object: component model to be added to the system model
        :type name: ComponentModel object

        :returns: ComponentModel object
        """
        if cm_object.name in self.component_models:
            self.component_models[cm_object.name] = cm_object
        else:
            self.component_models.__setitem__(cm_object.name, cm_object)
        return self.component_models[cm_object.name]

    def add_component_model(self, name, model='', model_args=None,
                            model_kwargs=None, workdir=None):
        """
        Create component model object and add it to system model.

        :param name: name of component model
        :type name: str

        :param model: python function. Its first argument is a dictionary
            of parameters. The function should return component model outputs.
        :type model: str

        :param model_args: additional optional arguments of component model
            provided as list; by default, model_args is None
        :type model_args: list

        :param model_kwargs: additional optional keyword arguments of
            component model provided as dictionary; by default, model_kwargs is None
        :type model_kwargs: dict

        :param workdir: name of directory to use for model runs (serial run case)
        :type workdir: str

        :returns: ComponentModel object
        """
        if name in self.component_models:
            self.component_models[name] = ComponentModel(
                name, self, model=model, model_args=model_args,
                model_kwargs=model_kwargs, workdir=workdir)
        else:
            self.component_models.__setitem__(name, ComponentModel(
                name, self, model=model, model_args=model_args,
                model_kwargs=model_kwargs, workdir=workdir))
        return self.component_models[name]

    def add_interpolator(self, intr_object, intr_family, creator=None):
        """
        Add interpolator object to system model.

        :param intr_object: interpolator object to be added to the system model
        :type name: Interpolator class object

        :param intr_family: name of the family under which intr_object will be stored
        :type intr_family: string

        :param creator: component or system model that initializes creation
            of interpolator
        type creator: SystemModel or ComponentModel class object

        :returns: Interpolator class object
        """
        # Check whether family of interpolators was already created
        if intr_family in self.interpolators:
            if intr_object.name in self.interpolators[intr_family]:
                self.interpolators[intr_family][intr_object.name] = intr_object
            else:
                self.interpolators[intr_family].__setitem__(intr_object.name, intr_object)
        else:
            self.interpolators.__setitem__(intr_family, OrderedDict())
            self.interpolators[intr_family].__setitem__(intr_object.name, intr_object)

        if creator:
            self.interp_creators[intr_family] = creator
        else:
            self.interp_creators[intr_family] = self

        return self.interpolators[intr_family][intr_object.name]

    def collect_observations_as_time_series(self, cm_object, obs_nm, indices=None):
        """
        Collect all observations of the component with the provided name.

        :param cm_object: component model of the system model
        :type name: ComponentModel object

        :param obs_nm: name of observation
        :type obs_nm: str

        :param indices: time indices to look for in the list of observations;
            by default it is None which means that all indices of self.time_array
            will be considered
        :type indices: list

        :returns: numpy array of values of observations
        """
        full_obs_nm = '.'.join([cm_object.name, obs_nm])

        # If indices are not specified then we want all indices in time array of system model
        if indices is None:
            ind_list = list(range(len(self.time_array)))
        else:
            ind_list = indices

        # Made obs_list a numpy array to provide opportunity to modify outputs
        # by multiplication, addition etc, right away without changing
        # to numpy array or other suitable type
        obs_list = np.array([self.obs[full_obs_nm+'_'+str(ind)].sim for ind in ind_list])

        return obs_list

    def collect_gridded_observations_as_time_series(
            self, cm_object, obs_nm, output_folder, indices=None,
            rlzn_number=0, save_type='npz'):
        """
        Collect all observations of the component with the provided name.

        :param cm_object: component model of the system model
        :type name: ComponentModel object

        :param obs_nm: name of observation
        :type obs_nm: str

        :param output_folder: absolute path to the folder with data files for
            the gridded observation
        :type output_folder: str

        :param indices: time indices to look for in the list of observations;
            by default, it is None which means that all indices of self.time_array
            will be considered, and the observations will be returned
            which corresponds to these indices
        :type indices: list

        :param rlzn_number: realization number for which the observation
            values need to be returned. By default, the parameter value is 0
            which corresponds to deterministic run with method forward,
            i.e., value of 0 is reserved only for non-stochastic simulations.
            For stochastic simulations the value of the parameter should be
            greater or equal than 1 up to the number of realizations run
            for the given system model.
        :type rlzn_number: int

        :param save_type: file format to be used to save gridded observations.
            Currently supported values are 'npz' (default), 'csv', and 'npy'.
        :type save_type: str

        :returns: numpy array of values of observations
        """
        # If indices are not specified then we want all indices in time array of system model
        if indices is None:
            ind_list = list(range(len(self.time_array)))
        else:
            ind_list = indices

        num_indices = len(ind_list)

        # Check whether output directory and files exist
        # Setup file name
        filename = '_'.join([cm_object.name, obs_nm, 'sim_{}',
                             'time_{}']).format(rlzn_number, ind_list[0])+'.'+save_type
        if not os.path.exists(os.path.dirname(output_folder)):
            err_msg = ''.join([
                'Folder {} is not found. Please check location of the ',
                'data files for gridded observation {}.']).format(output_folder, obs_nm)
            raise FileNotFoundError(err_msg)

        # Create path to the file
        file_path = os.sep.join([output_folder, filename])
        # Check whether the file exists
        if not os.path.isfile(file_path):
            err_msg = ''.join([
                'File {} is not found. Please check location and names of the ',
                'data files for gridded observation {}.']).format(file_path, obs_nm)
            raise FileNotFoundError(err_msg)

        # Get data and determine shape of the data for the first index in ind_list
        d = self.get_gridded_observation_file(file_path, file_extension=save_type)
        data_shape = d.shape

        # Made obs_list a numpy array to provide opportunity to modify outputs
        # by multiplication, addition etc, right away without changing
        # to numpy array or other suitable type
        obs_list = np.empty((num_indices,)+data_shape)
        # Copy data from the first data file corresponding to the first provided index
        obs_list[0, :] = d

        # Loop over remaining files
        for ind in range(1, num_indices):
            # Setup file name
            filename = '_'.join([cm_object.name, obs_nm, 'sim_{}',
                                 'time_{}']).format(rlzn_number, ind_list[ind])+'.'+save_type
            # Create path to the file
            file_path = os.sep.join([output_folder, filename])
            # Check whether the file exists
            if not os.path.isfile(file_path):
                err_msg = ''.join([
                    'File {} is not found. Please check location and names of the ',
                    'data files for gridded observation {}.']).format(file_path, obs_nm)
                raise FileNotFoundError(err_msg)
            # Get data
            d = self.get_gridded_observation_file(file_path, file_extension=save_type)
            # Copy to the obs_list array
            obs_list[ind, :] = d

        return obs_list

    # Override MATK's create_sampleset function
    # so that NRAP-Open-IAM's sampleset can be used instead
    def create_sampleset(self, samples, name=None, responses=None,
                         indices=None, index_start=1):
        """ Add sample set to problem

            :param name: Name of sample set
            :type name: str

            :param samples: Matrix of parameter samples with npar columns
                in order of matk.pars.keys()
            :type samples: list(fl64),ndarray(fl64)

            :param responses: Matrix of associated responses with nobs columns
                in order matk.obs.keys() if observation exists
                (existence of observations is not required)
            :type responses: list(fl64),ndarray(fl64)

            :param indices: Sample indices to use when creating working
                directories and output files
            :type indices: list(int),ndarray(int)
        """
        if not isinstance(samples, (list, np.ndarray)):
            print("Error: Parameter samples are not a list or ndarray")
            return 1
        npar = len(self.pars)
        # If list, convert to ndarray
        if isinstance(samples, list):
            samples = np.array(samples)
        if not samples.shape[1] == npar:
            print(''.join([
                'Error: The number of columns in sample is not ',
                'equal to the number of parameters in the problem']))
            return 1
        if name is None:
            ind = str(len(self.sampleset))
            name = 'ss'+str(ind)
            while name in self.sampleset:
                ind += 1
                name = 'ss'+str(ind)
        if name in self.sampleset:
            self.sampleset[name] = SampleSet(
                name, samples, parent=self, responses=responses,
                indices=indices, index_start=index_start)
        else:
            self.sampleset.__setitem__(name, SampleSet(
                name, samples, parent=self, responses=responses,
                indices=indices, index_start=index_start))
        return self.sampleset[name]

    def reorder_component_models(self, order):
        """
        Reorder execution order of component models.

        :param order: list of component model names in desired execution order
        :type order: lst(str)
        """
        self.component_models = OrderedDict((k, self.component_models[k]) for k in order)

    def swap_components(self, comp1, comp2):
        """
        Swap location of two component models.

        :param comp1: name of first component model to be switched
        :type comp1: str

        :param comp2: name of second component model to be switched
        :type comp2: str
        """

        # Get current list of ordered component models
        order = list(self.component_models.keys())

        # Find indices for component
        comp1_idx, comp2_idx = order.index(comp1), order.index(comp2)

        # Swap locations of comp1 and comp2
        order[comp1_idx], order[comp2_idx] = order[comp2_idx], order[comp1_idx]

        # Pass reordered list to reorder_component_models method
        self.reorder_component_models(self, order)

    def put_in_front_of_component(self, comp1, comp2):
        """
        Change location of component model to before another component model.

        :param comp1: name of reference component model
        :type comp1: str

        :param comp2: name of second component model to be placed before
            reference component model
        :type comp2: str
        """

        # Get current list of ordered component models
        order = list(self.component_models.keys())

        # Find indices for components
        comp1_idx, comp2_idx = order.index(comp1), order.index(comp2)

        # Remove comp2 from list of ordered component models
        order.remove(comp2)

        # Reinsert comp2 before comp1
        order.insert(comp1_idx, comp2)

        # Pass reordered list to reorder_component_models method
        self.reorder_component_models(self, order)

    def single_step_model(self, pardict=None, to_reset=False, sm_kwargs=None, job_number=None):
        """
        Return system model output for a single time step.

        :param pardict: dictionary of parameter values keyed by parameter names
        :type pardict: dict

        :param to_reset: flag indicating whether component models and
            accumulators need to be reset to their default/initial
            states/values.
        :type to_reset: boolean

        :param sm_kwargs: dictionary of keyword arguments (e.g. time step,
            time point) that can be passed from system model to
            component models.
        :type sm_kwargs: dict
        """
        # If parameter dictionary is not provided as an argument get it from self.pars
        if pardict is None:
            pardict = dict([(k, par.value) for k, par in list(self.pars.items())])

        # Initialize output of single_step_model
        total_out = {}

        # Accumulators of the component models are reset for each realization.
        # Each accumulator is assumed to be reset 0 but this behavior
        # can be changed in reset method of component model.
        if to_reset:
            for cm in self.component_models.values():
                # Reset using 'accumulators' dictionary attribute
                for k_ac in list(cm.accumulators.keys()):
                    cm.accumulators[k_ac].sim = 0.0
                # Reset using derived component model class 'reset' method
                # Use of reset method follows the resetting of the accumulators
                # in order to correct zero setting of accumulators if needed.
                cm.reset()

        # Set up parameter dictionary for composite parameter evaluation
        aeval = Interpreter()

        # Composite parameters can depend on random, deterministic and default
        # parameters. Iterate over all component models of the given system model.
        for cm in self.component_models.values():
            # Add default parameter values to aeval
            for k, dpar in cm.default_pars.items():
                aeval.symtable[sub('\.', '_', dpar.name)] = dpar.value
            # Add deterministic parameter values to aeval.  Deterministic
            # parameters should go after default parameters since their names coincide.
            for k, determ_par in cm.deterministic_pars.items():
                aeval.symtable[sub('\.', '_', determ_par.name)] = determ_par.value

        # Add pardict (stochastic) parameters into aeval.  Stochastic
        # parameters should go after default and deterministic parameters
        # since their names coincide.
        for k, v in pardict.items():
            aeval.symtable[sub('\.', '_', k)] = v

        # Iterate over all component models of the given system model.
        for cm in self.component_models.values():
            # Initialize parameter dictionary
            pars = {}

            # Determine deterministic parameters of component cm.
            for k, determ_par in cm.deterministic_pars.items():
                pars[k] = determ_par.value

            # Determine stochastic parameters of component cm
            for k, v in pardict.items():
                if k.startswith(cm.name+'.'):
                    pars[k.split('.')[-1]] = v

            # Determine parameters linked to other parameters
            for k, lpar in cm.parlinked_pars.items():
                # Split into component and parameter name
                lpar_cm, lpar_nm = lpar.split('.')
                # If parameter is linked to default parameter
                if lpar_nm in self.component_models[lpar_cm].default_pars:
                    pars[k] = self.component_models[lpar_cm].default_pars[lpar_nm].value
                # if parameter is linked to deterministic parameter
                if lpar_nm in self.component_models[lpar_cm].deterministic_pars:
                    pars[k] = self.component_models[lpar_cm].deterministic_pars[lpar_nm].value
                # if parameter is linked to stochastic parameter
                if lpar_nm in self.component_models[lpar_cm].pars:
                    pars[k] = self.component_models[lpar_cm].pars[lpar_nm].value
                # if parameter is linked to composite parameter
                elif lpar_nm in self.component_models[lpar_cm].composite_pars:
                    pars[k] = self.component_models[lpar_cm].composite_pars[lpar_nm].value
                # Assign value
                aeval.symtable['_'.join([cm.name, k])] = pars[k]

            # Determine composite parameters
            for k, comp_par in cm.composite_pars.items():
                cm.composite_pars[k].value = aeval(sub('\.', '_', comp_par.expr))
                # Add composite parameter value to aeval
                aeval.symtable[sub('\.', '_', comp_par.name)] = comp_par.value
                pars[k] = comp_par.value

            # Run component's model at frequency defined by the component's run_frequency attribute
            if (cm.run_frequency == 2) or ((cm.run_frequency == 1) and (to_reset is True)):
                # Determine parameters that obtain their values from observations
                for k, lobs in cm.obslinked_pars.items():
                    # Split into cm name and obs name
                    lobs_cm, lobs_nm = lobs.split('.')
                    if lobs_nm in self.component_models[lobs_cm].linkobs:
                        pars[k] = self.component_models[lobs_cm].linkobs[lobs_nm].sim

                # Determine keyword parameters that obtain their values from observations
                for k, lkwargs in cm.obs_linked_kwargs.items():
                    lkwargs_cm, lkwargs_nm = lkwargs.split('.')
                    if lkwargs_nm in self.component_models[lkwargs_cm].linkobs:
                        cm.model_kwargs[k] = (
                            self.component_models[lkwargs_cm].linkobs[lkwargs_nm].sim)

                # Determine keyword parameters that obtain their values from gridded observations
                for k, lkwargs in cm.grid_obs_linked_kwargs.items():
                    lkwargs_cm, lkwargs_nm = lkwargs['name'].split('.')
                    if lkwargs_nm in self.component_models[lkwargs_cm].linkobs:
                        linkobs_data = self.component_models[lkwargs_cm].linkobs[lkwargs_nm].sim
                        # Check whether list of indices is empty
                        if lkwargs['loc_ind']:
                            if len(lkwargs['loc_ind']) > 1:
                                # Setup array keyword argument
                                cm.model_kwargs[k] = np.array(
                                    [linkobs_data[ind] for ind in lkwargs['loc_ind']])
                            else:
                                # Set the scalar keyword argument since there is a single index
                                cm.model_kwargs[k] = linkobs_data[lkwargs['loc_ind'][0]]
                        else:
                            # If list of indices is empty all available data
                            # is copied to the keyword argument
                            # We may want to return np.asarray(linkobs_data)
                            # later but let's leave it as it is for now
                            cm.model_kwargs[k] = linkobs_data

                # Determine dynamic keyword parameters that obtain their values
                # from time series data
                for k, lkwarg_data in cm.dynamic_kwargs.items():
                    # Get the time series data at the time point defined by time_index
                    cm.model_kwargs[k] = lkwarg_data[sm_kwargs['time_index']]

                # Determine keyword parameters that obtain their values
                # from collection of observations
                for k, lkwargs_list in cm.collection_linked_kwargs.items():
                    cm.model_kwargs[k] = list()
                    for lkwargs in lkwargs_list:
                        lkwargs_cm, lkwargs_nm = lkwargs.split('.')
                        if lkwargs_nm in self.component_models[lkwargs_cm].linkobs:
                            cm.model_kwargs[k].append(
                                self.component_models[lkwargs_cm].linkobs[lkwargs_nm].sim)

                # Set up keyword arguments provided by system model
                if cm.model_kwargs is not None:
                    for k in list(cm.model_kwargs.keys()):
                        if k in ['time_point', 'time_step']:
                            cm.model_kwargs[k] = sm_kwargs[k]

                if to_reset: # check whether input parameters need to be checked
                    cm.check_input_parameters(pars)

                # Obtain output of the component model cm
                if cm.model_args is None and cm.model_kwargs is None:
                    out = cm.model(pars)
                elif cm.model_args is not None and cm.model_kwargs is None:
                    out = cm.model(pars, *cm.model_args)
                elif cm.model_args is None and cm.model_kwargs is not None:
                    out = cm.model(pars, **cm.model_kwargs)
                elif cm.model_args is not None and cm.model_kwargs is not None:
                    out = cm.model(pars, *cm.model_args, **cm.model_kwargs)

                # Find keys that needed to be removed from total_out,
                # e.g. corresponding to the gridded observations
                keys_to_be_removed = []

                for k, val in out.items():
                    # Get the name of calculated observation
                    output_name = '.'.join([cm.name, k])

                    # Assign value to the returned observation
                    total_out[output_name] = val

                    # Process linked observations
                    if k in cm.linkobs:
                        cm.linkobs[k].sim = val


                    # Process gridded observations
                    if k in cm.grid_obs:
                        # Get file type to save gridded observations to
                        save_type = cm.grid_obs[k].save_type

                        t_ind = sm_kwargs['time_index']
                        if t_ind in cm.grid_obs[k].time_indices:
                            if job_number is None:
                                filename = '_'.join([cm.name, k, 'sim_0',
                                                     'time_'+str(t_ind)])+'.'+save_type
                            else:
                                filename = '_'.join([
                                    cm.name, k,
                                    'sim_'+str(job_number),
                                    'time_'+str(t_ind)])+'.'+save_type

                            self.save_gridded_observation(
                                os.path.join(cm.grid_obs[k].output_dir, filename),
                                val, file_extension=save_type)

                    # We don't want to provide gridded observation
                    # in the output of single_step_model so we mark key k for deletion
                    if k in cm.grid_obs_keys:
                        keys_to_be_removed.append(output_name)

                    # Process local observations
                    if k in cm.local_obs:
                        for nm in cm.local_obs[k]:
                            ind = cm.local_obs[k][nm]
                            total_out['.'.join([cm.name, nm])] = np.asarray(val)[ind]

                for k in keys_to_be_removed:
                    total_out.pop(k)  # remove arrays/matrices from output

        return total_out

    def system_model(self, pardict=None, job_number=None):
        """
        Execute system model simulation.

        :param pardict: dictionary of parameter values keyed by parameter names
        :type pardict: dict
        """
        if self.single_time_point_flag:  # if single time point is supplied
            sm_kwargs = {'time_point': self.time_point,
                         'time_step': self.time_step, 'time_index': 0}
            return self.single_step_model(
                pardict, to_reset=True, sm_kwargs=sm_kwargs, job_number=job_number)

        # Proceed below if several time points are provided
        num_time_points = len(self.time_array)  # number of time points

        # Set up dictionary of system model parameters potentially useful for
        # component models
        sm_kwargs = {'time_point': self.time_array[0],
                     'time_step': self.time_array[1] - self.time_array[0],
                     'time_index': 0}   # time_index is needed for dynamic kwargs

        # Get observations for the first time point
        total_out = self.single_step_model(
            pardict, to_reset=True, sm_kwargs=sm_kwargs, job_number=job_number)
        for obs_nm in self.obs_base_names:
            ts_obs_nm = obs_nm+'_0'
            # Check whether observation is already in the output
            if ts_obs_nm not in total_out:
                total_out[ts_obs_nm] = total_out[obs_nm]

        # Iterate over the remaining time points
        for n in range(1, num_time_points):
            # Define time point and time step used by several component models
            sm_kwargs = {'time_point': self.time_array[n],
                         'time_step': self.time_array[n] - self.time_array[n-1],
                         'time_index': n}   # time_index is needed for dynamic kwargs

            # Find all observations of the system model
            out = self.single_step_model(pardict, sm_kwargs=sm_kwargs, job_number=job_number)

            # Set observations relevant to the current time step
            # The next 4 lines deal with observations
            # whose names start with string in base_obs_names attribute
            # of system component and end with '_' + corresponding time index.
            # We don't use the "indexed" observations for connections
            # between components: to simplify connections we use outputs of
            # component models with base_obs_names.
            # Since the end user may not be necessary interested in all time indices,
            # the corresponding outputs are collected after each time step.
            for obs_nm in self.obs_base_names:
                ts_obs_nm = obs_nm+'_'+str(n)
                # Check whether the observation is needed and whether it is already in the output
                # TODO if we approve this approach we might need to
                # take into account the connections between components
                if (ts_obs_nm in self.obs) and (ts_obs_nm not in total_out):
                    out[ts_obs_nm] = out[obs_nm]

            # Update dictionary of outputs
            total_out.update(out)

        return total_out

    def register_ml_model_component(self, cm_object, cm_type):
        """
        Create a record for a component object that uses ml based model as a backend model.

        The method is recommended to be used within the constructor method of the component
        to be recorded for purpose of saving/reusing information about loaded models.

        :param cm_object: component model of the system model for which record
            is created
        :type name: ComponentModel object

        :param cm_type: type of component model for which record
            is created
        :type name: str
        """
        if cm_type not in self.ml_models_components:
            self.ml_models_components.__setitem__(cm_type, cm_object)

    @staticmethod
    def get_gridded_observation_file(file_path, file_extension=None):
        """
        Load file with data using method dependent on the file type.

        :param file_path: Path to the file with data to be loaded.
        :type file_path: str

        :param file_extension: Extension of file to be loaded. The default is 'npz'.
        :type file_extension: str

        :returns: data, numpy.ndarray
            Array containing data loaded from file.

        """
        if file_extension is None:
            # Find occurence of last period in file name and assume that what follows
            # is a file extension
            extension_index = file_path.rfind('.') + 1
            file_extension = file_path[extension_index:]

        if file_extension == 'npz':
            data_npz = np.load(file_path)
            data = data_npz['data']
            data_npz.close()
        elif file_extension == 'csv':
            data = np.loadtxt(file_path, delimiter=',')
        elif file_extension == 'npy':
            data = np.load(file_path)
        else:
            err_msg = 'File extension {} is not supported.'.format(file_extension)
            raise ValueError(err_msg)

        return data

    @staticmethod
    def save_gridded_observation(filename, grid_obs_data, file_extension='npz'):
        """
        Save gridded observation data in a selected format.

        :param filename: Path to the file for data to be saved
        :type filename: str

        :param grid_obs_data: Data to be saved
        :type grid_obs_data: Array or matrix like.

        :param file_extension: File format to be used to save data
        :type file_extension: str

        :returns: None

        """
        # Once filename is defined, save data in the file
        if file_extension == 'npz':
            # Save several arrays into a single file in compressed .npz format.
            np.savez_compressed(filename, data=np.asarray(grid_obs_data))

        elif file_extension == 'csv':
            # Save several arrays into a single file in CSV format.
            np.savetxt(filename, np.asarray(grid_obs_data), delimiter=',')

        elif file_extension == 'npy':
            # Save several arrays into a single file in Numpy ('npy') format.
            np.save(filename, np.asarray(grid_obs_data))


class ComponentModel():
    """
    OpenIAM ComponentModel class.

    Class represents basic framework to construct component models, handle different
    types of parameters and observations.

    Certain attributes are specified in the component model for connecting
    the component model to the system.  These attributes are:
    needs_locXY = False # locX and locY are kwargs
    system_inputs = [] # inputs from the system model (e.g., 'pressure', 'CO2saturation')
    system_params = [] # parameters from the system model (e.g., aquiferThickness)
    composite_inputs = OrderedDict() # Inputs that are composites of system
    parameters (e.g., 'reservoirDepth')
    system_collected_inputs = {} # Inputs collected into arrays
    adapters = [] # List of adapters needed to supply input (RateToMass)
    needsXY = False # If the component needs keyword arguments x and y as arrays of locations.
    """
    def __init__(self, name, parent, model='',
                 model_args=None, model_kwargs=None, workdir=None):
        """
        Constructor method of ComponentModel class.

        :param name: name of component model
        :type name: str

        :param parent: the SystemModel object that the component model belongs to
        :type parent: SystemModel object

        :param model: python function whose first argument is a dictionary
            of parameters. The function returns dictionary of model outputs.
        :type model: function or method that is called when the component is run

        :param model_args: additional optional parameters of the component
            model; by default model_args is empty list []
        :type model_args: [float]

        :param model_kwargs: additional optional keyword arguments of
            the component model; by default model_kwargs is empty dictionary {}.
        :type model_kwargs: dict

        :param workdir: name of directory to use for model runs (serial run case)
        :type workdir: str

        :returns: object -- ComponentModel object
        """
        self._parent = parent
        self.name = name
        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.workdir = workdir
        self.class_type = 'ComponentModel'  # name of class needed for CFI and GUI

        # Parameters
        self.default_pars = OrderedDict()
        self.deterministic_pars = OrderedDict()
        self.pars = OrderedDict()
        self.composite_pars = OrderedDict()
        self.gridded_pars = OrderedDict()  # placeholder for future work
        self.parlinked_pars = OrderedDict()
        self.obslinked_pars = OrderedDict()

        # Keyword arguments
        self.obs_linked_kwargs = OrderedDict()
        self.grid_obs_linked_kwargs = OrderedDict()
        self.dynamic_kwargs = OrderedDict()
        self.collection_linked_kwargs = OrderedDict()

        # Observations
        self.linkobs = OrderedDict()
        self.accumulators = OrderedDict()
        self.obs = OrderedDict()
        self.grid_obs = OrderedDict()
        self.local_obs = OrderedDict()

        # Record of gridded observations
        self.grid_obs_keys = list() # empty list means there are no gridded observations

        # Added attribute obs_base_names to keep track on what observations
        # will have a time index added to the end of name
        self.obs_base_names = list()

        # Parameters and temporal inputs boundaries if applicable
        self.pars_bounds = dict()
        self.temp_data_bounds = dict()

        # Set the working directory index for parallel runs
        self.workdir_index = 0

        # Setup how often the model method should be run
        # Possible values:
        # 0 - do not run but setup parameters, container type of component
        # 1 - run only once for the very first time step
        # 2 - run for all time steps (most common)
        self.run_frequency = 2

    def add_par(self, name, value=None, vary=True, min=None, max=None,
                expr=None, discrete_vals=[], **kwargs):
        """
        Add parameter to component model.

        :param name: name of parameter
        :type name: str

        :param value: Initial parameter value
        :type value: float

        :param vary: flag indicating whether parameter is deterministic or
            stochastic (varied); by default parameter is assumed to be varied
        :type vary: boolean

        :param min: minimum bound of parameter
        :type min: float

        :param max: maximum bound of parameter
        :type max: float

        :param expr: mathematical expression used to calculate parameter value
        :type expr: str

        :param discrete_vals: tuple of two array_like defining discrete values
            and associated probabilities (probabilities are normalized if they
            do not sum up to 1)
        :type discrete_vals: (lst,lst)

        :param kwargs: additional keyword arguments passed to __init__ method
            of Parameter class
        :type kwargs: any

        """
        if vary is False:  # if parameter to be added is deterministic
            if name in self.deterministic_pars:
                self.deterministic_pars[name] = Parameter(
                    '.'.join([self.name, name]), parent=self._parent,
                    value=value, vary=False)
            else:
                self.deterministic_pars.__setitem__(name, Parameter(
                    '.'.join([self.name, name]), parent=self._parent,
                    value=value, vary=False))
        else: # if parameter to be added is stochastic
            # Check if discrete values are provided in a different format
            if 'Values' in kwargs:
                kwargs['values'] = kwargs['Values']
                # Remove from dict due to inconsistency with the MATK Parameter class
                # argument list
                kwargs.pop('Values', None)
            if 'Weights' in kwargs:
                kwargs['weights'] = kwargs['Weights']
                # Remove from dict due to inconsistency with the MATK Parameter class
                # argument list
                kwargs.pop('Weights', None)
            # Check whether the user entered parameters as low case
            if 'values' in kwargs:
                if 'weights' in kwargs:
                    discrete_vals = (kwargs['values'], kwargs['weights'])
                    kwargs.pop('weights', None)
                else:
                    num_values = len(kwargs['values'])
                    discrete_vals = (kwargs['values'],
                                     num_values*[1./num_values])
                kwargs.pop('values', None)

            # Check whether parameter is already added
            if '.'.join([self.name, name]) in self._parent.pars:
                self._parent.pars['.'.join([self.name, name])] = Parameter(
                    '.'.join([self.name, name]), parent=self._parent,
                    value=value, vary=vary, min=min, max=max, expr=expr,
                    discrete_vals=discrete_vals, **kwargs)
            else:
                self._parent.pars.__setitem__(
                    '.'.join([self.name, name]),
                    Parameter('.'.join([self.name, name]), parent=self._parent,
                              value=value, vary=vary, min=min, max=max, expr=expr,
                              discrete_vals=discrete_vals, **kwargs))
            self.pars[name] = self._parent.pars['.'.join([self.name, name])]

    def add_default_par(self, name, value=None, **kwargs):
        """
        Add default parameter to component model.

        :param name: name of parameter
        :type name: str

        :param value: parameter value
        :type value: float

        :param kwargs: keyword arguments passed to constructor method of
            Parameter class
        :type kwargs: any
        """
        if name in self.default_pars:
            self.default_pars[name] = Parameter(
                '.'.join([self.name, name]), parent=self._parent,
                value=value, vary=False, **kwargs)
        else:
            self.default_pars.__setitem__(name, Parameter(
                '.'.join([self.name, name]), parent=self._parent,
                value=value, vary=False, **kwargs))

    def add_par_linked_to_par(self, name, parlink):
        """
        Add parameter linked to another parameter.

        Add parameter that obtains its value from parameter
        which comes from the same or another component models.
        The parameter from which the value is taken can be default,
        deterministic, stochastic or composite.

        :param name: name of parameter to be added
        :type name: str

        :param parlink: ComponentModel parameter
        :type parlink: Parameter class object
        """
        self.parlinked_pars[name] = parlink.name

    def add_par_linked_to_obs(self, name, obslink):
        """
        Add parameter linked to observation.

        Add parameter that obtains its value from observation returned
        by the same or another component model. The observation has to be
        observation to be linked.

        :param name: name of parameter
        :type name: str

        :param obslink: ComponentModel observation
        :type obslink: Observation class object
        """
        self.obslinked_pars[name] = obslink.name

    def add_gridded_par(self, name, interpolator):
        """
        Add parameter sampled from a gridded data.

        Add parameter sampled from a gridded data. The parameter also keeps the data
        about the interpolator
        """
        # TODO Replace interpolator with link to the parameter generator/sampler
        if name in self.gridded_pars:
            self.gridded_pars[name] = interpolator
        else:
            self.gridded_pars.__setitem__(name, interpolator)

    def add_composite_par(self, name, expr=None):
        """
        Add composite parameter.

        We assign composite parameter its value evaluating expression
        which can contain names of default, deterministic, stochastic
        or other composite parameters.

        :param name: name of composite parameter
        :type name: str

        :param expression: expression for calculating the value of parameter.
            It has a form: 'f(par_nm1,par_nm2,par_nm3,...)'
        :type expression: str

        """
        if name in self.composite_pars:
            self.composite_pars[name] = Parameter(
                '.'.join([self.name, name]), parent=self._parent, expr=expr)
        else:
            self.composite_pars.__setitem__(name, Parameter(
                '.'.join([self.name, name]), parent=self._parent, expr=expr))

    def add_dynamic_kwarg(self, name, time_series_data):
        """
        Add keyword argument which obtains its value from time series array.

        :param name: name of keyword argument
        :type name: str

        :param time_series_data: data to be assigned to the kwarg argument name of
            the component's model method; data length should be equal
            to the number of time points defined by the system model. It should
            be possible to obtain the value of kwarg argument through simple
            reference, e.g. value(s) at the first time point should be
            time_series_data[0]; value(s) at the second time point should be
            time_series_data[1]. time_series_data[ind] can have any type
            appropriate for a particular component
        :type time_series_data: list
        """
        # Check whether system model is to be run for several time points
        if self._parent.time_array is not None:
            # Check whether the number of data points provided coincide with
            # the length of time_series_data
            data_len = len(time_series_data)
            time_arr_len = len(self._parent.time_array)
            if data_len == time_arr_len:
                self.dynamic_kwargs[name] = time_series_data
            elif data_len < time_arr_len:
                raise IndexError(''.join([
                    'Dynamic parameter {name} cannot be created since there are ',
                    'not enough time series data points provided. Expected ',
                    '{lta} data points, received {ltsd} data points.']).format(
                        name=name, lta=time_arr_len, ltsd=data_len))
            else:
                raise IndexError(''.join([
                    'Dynamic parameter {name} cannot be created since there are ',
                    'more time series data points provided than are needed. ',
                    'Expected {lta} data points, received {ltsd} data points.']).format(
                        name=name, lta=time_arr_len,ltsd=data_len))

        else: # If system model will be run for a single time point
            self.dynamic_kwargs[name] = time_series_data

    def add_kwarg_linked_to_obs(self, name, obslink, obs_type='scalar', **kwargs):
        """
        Add keyword argument linked to observation.

        Add keyword argument to the component model that obtains its
        value from the same or another component model observation.
        The observation should be created as an observation to be linked.

        :param name: name of keyword parameter
        :type name: str

        :param obs_type: type of observation that the keyword argument will
            be linked to. Possible values: 'scalar' and 'grid'. By default,
            the parameter value is 'scalar'.
        :type obs_type: str

        :param kwargs: additional keyword arguments specifying constr_type and
            loc_ind for obs_type='grid'. Option is added to accomodate the need
            to request only several of grid points observations.
            kwargs['constr_type'] is string in ['array','matrix'];
            kwargs['loc_ind'] is list of scalars (for 'array' constr_type) or
            list of tuples (for 'matrix' constr_type). Default empty dictionary kwargs
            indicates that all data provided by the gridded observation is needed.
        :type kwargs: dict

        :param obslink: ComponentModel observation
        :type obslink: Observation class or GriddedObservation class object
        """
        if obs_type == 'scalar':
            self.obs_linked_kwargs[name] = obslink.name

        elif obs_type == 'grid':
            self.grid_obs_linked_kwargs[name] = {'name': obslink.name}
            if (('constr_type' in kwargs) and ('loc_ind' in kwargs)):
                # Primitive check for consistency between provided loc_ind and constr_type
                if (((kwargs['constr_type'] == 'array') and (
                        all(isinstance(item, int) for item in kwargs['loc_ind']))) or
                        ((kwargs['constr_type'] == 'matrix') and (
                            all(isinstance(item, tuple) for item in kwargs['loc_ind'])))):
                    self.grid_obs_linked_kwargs[name]['loc_ind'] = kwargs['loc_ind']
                else:
                    raise Exception(''.join([
                        'Incompatible combination of keyword parameters ',
                        'constr_type and loc_ind, or wrong types used.']))
            else:
                self.grid_obs_linked_kwargs[name]['loc_ind'] = []

    def add_kwarg_linked_to_collection(self, name, obslink_list):
        """
        Add keyword argument linked to list of observations.

        :param name: name of keyword parameter
        :type name: str

        :param obslink_list: list of ComponentModel observations
        :type obslink_list: list of Observation class objects

        """
        self.collection_linked_kwargs[name] = list()
        for obslink in obslink_list:
            self.collection_linked_kwargs[name].append(obslink.name)

    def add_obs_to_be_linked(self, name, obs_type='scalar', **kwargs):
        """
        Add observation to be used as input to some component model.

        The method is used when a particular observation of a given component
        is not necessary of interest for analysis but is needed as an input for one of
        the subsequent components.

        :param name: name of observation
        :type name: str

        :param obs_type: type of observation to be linked. Possible values:
            'scalar' and 'grid'. By default, the parameter value is 'scalar'.
        :type obs_type: str

        :param kwargs: optional additional keyword arguments of the Observation or
            GriddedObservation classes constructors.
        :type kwargs: dict
        """
        if obs_type == 'scalar':
            if name in self.linkobs:
                self.linkobs[name] = Observation('.'.join([self.name, name]), **kwargs)
            else:
                self.linkobs.__setitem__(name, Observation(
                    '.'.join([self.name, name]), **kwargs))

        elif obs_type == 'grid':
            if name in self.linkobs:
                self.linkobs[name] = GriddedObservation('.'.join([self.name, name]), **kwargs)
            else:
                self.linkobs.__setitem__(name, GriddedObservation(
                    '.'.join([self.name, name]), **kwargs))
        else:
            raise ValueError('Observation type argument is not recognized.')

    def add_accumulator(self, name, sim=None, weight=1.0, value=None):
        """
        Add an observation that will be accumulated over time steps.

        :param name: name of observation
        :type name: str

        :param sim: simulated value of observation
        :type sim: fl64

        :param weight: observation weight
        :type weight: fl64

        :param value: measured/initial value of observation
        :type value: fl64
        """
        if name in self.accumulators:
            self.accumulators[name] = Observation(
                '.'.join([self.name, name]), sim=sim, weight=weight, value=value)
        else:
            self.accumulators.__setitem__(name, Observation(
                '.'.join([self.name, name]), sim=sim, weight=weight, value=value))

    def add_obs(self, name, index='all', sim=None, weight=1.0, value=None):
        """
        Add observation to component model.

        :param name: base name of observation
        :type name: str

        :param index: indices of time points at which observations values are
            of interest; index is a str('all') if all points should be added, or
            is a list if only selected point(s) is (are) added. By default,
            all points are added.
        :type index: str or array-like

        :param sim: simulated value of observation
        :type sim: fl64

        :param weight: observation weight
        :type weight: fl64

        :param value: measured/initial value of observation; by default,
            the value is None. value should be a list of length of index,
            unless it's None
        :type value: None or array-like
        """
        base_nm = '.'.join([self.name, name])

        # Add self to the list of system model components providing given observation
        if name not in self._parent.observation2components:
            self._parent.observation2components[name] = [self.name]
        else:
            self._parent.observation2components[name].append(self.name)

        # If single point is requested, there is no need to add time point index
        if self._parent.time_array is None:
            if base_nm in self._parent.obs:
                self._parent.obs[base_nm] = Observation(
                    base_nm, sim=sim, weight=weight, value=value)
            else:
                self._parent.obs.__setitem__(base_nm, Observation(
                    base_nm, sim=sim, weight=weight, value=value))
                self.obs[name] = self._parent.obs[base_nm]
        else: # if time array is defined
            if index == 'all':
                ind_array = list(range(len(self._parent.time_array)))
            else:
                ind_array = index

            # Index and value are supposed to have the same length, unless val is None
            if value is None:
                val_list = [value]*len(ind_array)
            else:
                val_list = value
                if len(val_list) != len(ind_array):
                    raise TypeError(''.join([
                        'Index and value used as lists/arrays ',
                        'should have the same number of elements.']))

            # Add observation name to the list of base observations in the system model
            if base_nm not in self._parent.obs_base_names:
                self._parent.obs_base_names.append(base_nm)

            if base_nm not in self.obs_base_names:
                self.obs_base_names.append(name)

            for ind_counter, ind_val in enumerate(ind_array):
                obs_nm = base_nm+'_'+str(ind_val)

                if obs_nm in self._parent.obs:
                    self._parent.obs[obs_nm] = Observation(
                        obs_nm, sim=sim, weight=weight, value=val_list[ind_counter])
                else:
                    self._parent.obs.__setitem__(obs_nm, Observation(
                        obs_nm, sim=sim, weight=weight, value=val_list[ind_counter]))
                self.obs[name+'_'+str(ind_val)] = self._parent.obs[obs_nm]

    def add_grid_obs(self, name, constr_type, coordinates=None, output_dir='', save_type='npz',
                     index='all', sim=None, weight=1.0, value=None):
        """
        Add observation computed on a grid.

        :param name: observation name
        :type name: str

        :param constr_type: type of the gridded observation. Possible values:
            'array', 'matrix', and ''. The parameter indicates what type of gridded
            observation, a given component will provide. 'array' option means that
            the returned observation obs will be an array-like object and
            every element can be accessed as obs[ind]; 'matrix' means that the returned
            observation is a matrix-like object and every element can be accessed
            by providing 2 (or 3) indices of elements: either like
            obs[ind1][ind2] or obs[ind1,ind2] for 2d case, or
            obs[ind1][ind2][ind3] or obs[ind1,ind2,ind3] for 3d case;
            '' option means that constr_type is not relevant, suitable for
            defining gridded obs to be linked.
        :type constr_type: str

        :param coordinates: x, y (and z) coordinates of the grid associated
            with the observation; by default, it's an empty dictionary. In general,
            coordinates should be a dictionary with keys 'x', 'y' (and 'z').
            coordinates[key] should be an array-like object
            for any key in coordinates.
        :type coordinates: dict of array-like objects

        :param output_dir: directory where the observations will be stored
            as compressed binary files
        :type output_dir: str

        :param save_type: file format for save of gridded observation,
                          currently supports 'npz' (default), 'csv', and 'npy'
        :type save_type: str


        :param index: indices of time points at which observations values are
            of interest; index is a str('all') if all points should be saved, or
            is an array-like object if only selected point(s) is (are) to be
            saved. By default, all points are saved.
        :type index: str or array-like

        :param sim: simulated value
        :type sim: fl64

        :param weight: observation weight
        :type weight: fl64

        :param value: measured/initial value of observation
        :type value: fl64
        """
        # Check whether all time points are of interest
        if index == 'all':
            if self._parent.time_array is None:
                ind_array = [0]     # single time point
            else:
                ind_array = list(range(len(self._parent.time_array)))
        else:
            ind_array = index

        if name in self.grid_obs:
            self.grid_obs[name] = GriddedObservation(
                '.'.join([self.name, name]), constr_type=constr_type,
                coordinates=coordinates, output_dir=output_dir, save_type = save_type, index=ind_array, sim=sim, weight=weight, value=value)
        else:
            self.grid_obs.__setitem__(name, GriddedObservation(
                '.'.join([self.name, name]), constr_type=constr_type,
                coordinates=coordinates, output_dir=output_dir, save_type = save_type, index=ind_array, sim=sim, weight=weight, value=value))

    def add_local_obs(self, name, grid_obs_name, constr_type, loc_ind, index='all',
                      sim=None, weight=1.0, value=None):
        """
        Add local observation linked to observation defined on a grid.

        The local observation is considered to be an observation of the same
        component as the original gridded observation and system model.
        One thing that distinguishes local obs from other observations is that
        its name should never coincide with the name of any observation
        returned by a given component.

        :param name: base name of observation
        :type name: str

        :param grid_obs_name: name of gridded observation that belongs to
            the same component. It is not necessary for the
            gridded observation to be created separately. It is only important
            that the component produces an array-like output with the given name.
        :type grid_obs_name: str

        :param constr_type: type of the gridded observation. Possible values:
            'array' and 'matrix'. The parameter indicates what type of gridded
            observation, a given component will provide. 'array' option means that
            the returned observation obs will be an array-like object and
            every element can be accessed as obs[ind]; 'matrix' means that the returned
            observation is a matrix-like object and every element can be accessed
            by providing 2 (or 3) indices of elements: either like
            obs[ind1][ind2] or obs[ind1,ind2] for 2d case, or
            obs[ind1][ind2][ind3] or obs[ind1,ind2,ind3] for 3d case.
        :type constr_type: str

        :param loc_ind: index of row containing local observation of interest
            for 'array' type of gridded observation, or indices of row and column
            at which value is located for 'matrix' type of gridded observation.
        :type loc_ind: int for array-type observation or tuple of integers
            for matrix-type observation

        :param index: indices of time points at which observations values are
            of interest; index is a str('all') if all points should be added, or
            is a list if only selected point(s) is (are) added. By default,
            all points are added.
        :type index: str or array-like

        :param sim: simulated value of observation
        :type sim: fl64

        :param weight: observation weight
        :type weight: fl64

        :param value: measured/initial value of observation; by default,
            the value is None. value should be a list of length of index,
            unless it's None
        :type value: None or array-like
        """
        if name == grid_obs_name:   # if the name coincide
            raise ValueError('Name of local observation cannot coincide with '+
                             'the name of the gridded observation. '+
                             'Possible names (among others) are ' + name +'_locind_###.')
        else:
            # Primitive check for consistency between provided loc_ind and constr_type
            if (((constr_type == 'array') and (isinstance(loc_ind, int))) or
                    ((constr_type == 'matrix') and (isinstance(loc_ind, tuple)))):

                self.add_obs(name, index=index, sim=sim, weight=weight, value=value)

                # Check whether any local observations are already created
                if grid_obs_name not in self.local_obs:
                    # Set the corresponding entry of the ordered dictionary
                    self.local_obs.__setitem__(grid_obs_name, dict())

                # Since the local observations come from the same component as gridded_obs
                self.local_obs[grid_obs_name][name] = loc_ind
            else:
                raise Exception(''.join([
                    'Incompatible combination of input parameters ',
                    'constr_type and loc_ind, or wrong types used.']))

    def check_input_parameters(self, p):
        """
        Check whether input parameters fall within specified boundaries.

        :param p: input parameters of component model
        :type p: dict
        """
        import logging
        debug_msg = 'Input parameters of {name} component are {p}.'.format(
            name=self.name, p=p)
        logging.debug(debug_msg)

        if hasattr(self, 'pars_bounds') and len(self.pars_bounds) > 0:

            for key, val in p.items():
                if key in self.pars_bounds:
                    if (val < self.pars_bounds[key][0]) or (val > self.pars_bounds[key][1]):
                        warn_msg = ''.join([
                            'Parameter {} of component {} with value {} ',
                            'is out of boundaries.']).format(key, self.name, val)
                        logging.warning(warn_msg)
                else:
                    warn_msg = ''.join([
                        'Parameter {key} is not recognized as component ',
                        '{name} input parameter.']).format(key=key, name=self.name)
                    logging.warning(warn_msg)
        else:
            debug_msg = ''.join([
                'Component {name} does not define boundaries ',
                'of its model parameters.']).format(name=self.name)
            logging.debug(debug_msg)

    def reset(self):
        """
        Reset parameters, observations and accumulators.

        Parameters, observations and accumulators are reset to their
        initial/default values at the beginning of each new simulation.
        Right now, the method contains only 'pass' since we assume that
        all accumulators need to be set to zero. Method can be redefined
        in a particular construction of ComponentModel class.
        """
        pass

    def show_input_limits(self):
        """
        Show boundaries of the parameters and time varying inputs if applicable.
        """
        if hasattr(self, 'pars_bounds') and len(self.pars_bounds) > 0:
            print('Model parameters boundaries:')
            print(self.pars_bounds)
        else:
            print("Parameters boundaries are not defined. Please refer\n"+
                  "to the model documentation for more information.")

        if hasattr(self, 'temp_data_bounds') and len(self.temp_data_bounds) > 0:
            print('Model temporal inputs boundaries:')
            print(self.temp_data_bounds)
        else:
            print("Temporal input boundaries are not accessible. Please refer\n"+
                  "to the model documentation for more information.")

    # Attributes used to set up component model in system
    needs_locXY = False # locX and locY are kwargs
    system_inputs = [] # inputs from the system model
    system_params = [] # parameters from the system model
    composite_inputs = OrderedDict()
    system_collected_inputs = {}
    adapters = []


class SamplerModel(ComponentModel):
    """
    Constructor method of SamplerModel class.
    """
    def __init__(self, name, parent, reproducible=True,
                 default_seed_value=1):

        super().__init__(name=name, parent=parent, model=self.sample,
                         model_args=None, model_kwargs=None)

        # Add class_type attribute
        self.class_type = 'SamplerModel'

        # Whether results of the sample generation are reproducible
        # affects whether the parameter seed will be used
        self.reproducible = reproducible
        self.add_default_par('seed', value=default_seed_value)

        # Run only once for the very first time step
        self.run_frequency = 1

    def check_seed_parameter(self, seed):
        """
        Check whether seed is an integer greater than zero.
        """
        if not isinstance(seed, int):
            err_msg = 'Parameter seed of component {} is not an integer.'.format(
                self.name)
            logging.error(err_msg)
            raise TypeError(err_msg)
        else:
            if seed <= 0:
                err_msg = 'Parameter seed of component {} is less or equal to zero.'.format(
                    self.name)
                logging.error(err_msg)
                raise ValueError(err_msg)


    def sample(self, p, **kwargs):
        """
        :param p: input parameters of SamplerModel sample method
        :type p: dict

        :param kwargs: additional keyword arguments of SamplerModel sample method.
        """
        pass


if __name__ == "__main__":
    sm = SystemModel()
    cmpnt = ComponentModel('cmpnt', sm)

    print('Attributes of the ComponentModel class object: ', dir(cmpnt))
