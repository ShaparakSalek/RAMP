from __future__ import print_function
from re import split
import numpy as np
from matk import sampleset


class SampleSet(sampleset.SampleSet):
    """ NRAP-Open-IAM SampleSet class (derived from MATK SampleSet class -
        Stores information related to a sample including parameter samples,
        associated responses, and sample indices
    """
    def __init__(self, name, samples, parent, index_start=1, **kwargs):
        # Inherent matk sampleset class .__init__ method
        super().__init__(name, samples, parent, index_start=index_start, **kwargs)

    def collect_observations_as_time_series(
            self, cmpnt=None, obs_nm=None, indices=None):
        """
        Collect observations of the system model corresponding to all simulations.

        Parameters
        ----------
        cmpnt : object of ComponentModel class whose observations need to be
            collected. The argument is optional. The default is None. If None
            is provided observations of all components are collected.
        obs_nm : str, name of the observation to be collected. The argument is
            optional. The default is None. If None is provided all the
            observations of the given component (if provided) are collected.
        indices : list, indices of time points at which observations are to
            be collected. The argument is optional. The default is None. If None
            is provided observations at all the time points are collected.

        Returns
        -------
        out : dict. Dictionary of observations with keys of the form "cm_name.obs_nm".
        """
        ssr = self.responses.recarray
        t_series_names = [] # Time series observation basenames
        nms = [] # Non time series observation names
        times = [] # Times
        # Check what arguments
        if cmpnt is not None:
            if obs_nm is not None:
                # Check that observation exists
                if obs_nm in cmpnt.obs_base_names:
                    t_series_names = ['.'.join([cmpnt.name, obs_nm])]
                else:
                    raise KeyError(''.join([
                        'Name {} is not recognised as an observation name ',
                        'of the component {}. Check whether the observation ',
                        'was added with add_obs methid.']).format(
                            obs_nm, cmpnt.name))
            else:
                t_series_names = ['.'.join([cmpnt.name, nm]) for nm in cmpnt.obs_base_names]

            uniq_t_series_names = t_series_names

            # If indices are not specified then we want all indices in time array of system model
            if indices is None:
                uniq_times = list(range(len(self._parent.time_array)))
            else:
                uniq_times = indices
        else:
            # Split on underscore followed by integer at end of name to identify
            # time series observations and collect their basenames
            for nm in ssr.dtype.names:
                vs = split("_[0-9]+$", nm)
                if len(vs) == 2:
                    t_series_names.append(vs[0])
                    times.append(split("_", nm)[-1])
                elif len(vs) == 1:
                    nms.append(vs[0])
                else:
                    raise KeyError("".join([
                        "Error: Observation name '{}' is not recognized neither as",
                        "time series nor as a constant observation"]).format(nm))
            # Collect unique time series observation basenames
            uniq_t_series_names = np.unique(t_series_names)
            uniq_times = sorted(np.unique(times), key=lambda x: int(x))

        # Collect observation names associated with each basename
        out = {}
        for bn in uniq_t_series_names:
            onms = []
            for ut in uniq_times:
                onms.append(bn+"_"+str(ut))
            # Collect values associated with the basename into an array
            ovs = []
            for t in onms:
                ovs.append(ssr[t])
            # Transpose so that rows are samples (realizations) and columns are times
            ovs = np.transpose(ovs)
            out[bn] = ovs
        # Add non time series variables to output dictionary
        for nm in nms:
            out[nm] = ssr[nm]

        return out
