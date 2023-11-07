# -*- coding: utf-8 -*-
"""
Last modified:

@author: Alexander Hanna (Alexander.Hanna@pnnl.gov)
PNNL (Battelle)
"""
import numpy as np


def pareto(det, ttd):
    """
    Compute the Pareto rank of each pair of detection and time-to-detection values

    Parameters
    ----------
    det : list or numpy array
    ttd : list or numpy array
    Returns
    -------
    ranks : numpy array
        List of Pareto rank values for each det/ttd pair
    """
    thisRank = 1
    ranks = list(np.ones(len(ttd), dtype='int'))
    while True:
        nextRank = []
        for i, ttd_i in enumerate(ttd):
            for j, ttd_j in enumerate(ttd):
                if i == j:
                    pass
                elif ranks[i] < thisRank or ranks[j] < thisRank:
                    pass
                elif ttd_j < ttd_i and det[j] > det[i]:
                    nextRank += [i]
                    ranks[i] = thisRank + 1
                    break
        if len(nextRank) == 0:
            break
        else:
            thisRank += 1
        break
    return np.array(ranks)


def scatter2step(x, y):
    x_step = []
    y_step = []
    for i in range(len(x)):
      x_step += [x[i]]
      x_step += [x[i]]
      y_step += [y[i]]
      y_step += [y[i]]
    return x_step[1:], y_step[:-1]


def find_unique_pareto(plans):
    det = np.array([plan[1] for plan in plans])
    ttd = np.array([plan[2] for plan in plans])
    ranks = pareto(det, ttd)

    temp, ui = np.unique(np.concatenate([
        det.reshape([len(det), 1]),
        ttd.reshape([len(ttd), 1])],
        axis=1), axis=0, return_index=True)

    plansUniquePareto = []
    for i in np.where(ranks == 1)[0]:
        if i in ui:
            plansUniquePareto += [list(plans)[i]]
    return plansUniquePareto


def find_different_density_same_timestep(plans,arrays):
    selected = set()
    for plan in plans:
      if len(plan[0])>0:
        for iDeployment in range(len(plan[0])):
          for jDeployment in range(len(plan[0])):
            if iDeployment==jDeployment: continue
            if plan[0][iDeployment][1]!=plan[0][jDeployment][1]: continue
            dens1 = np.abs(arrays[plan[0][iDeployment][0]]['receivers'][0]-arrays[plan[0][iDeployment][0]]['receivers'][1])
            dens2 = np.abs(arrays[plan[0][jDeployment][0]]['receivers'][0]-arrays[plan[0][jDeployment][0]]['receivers'][1])
            if arrays[plan[0][iDeployment][0]]['receivers']==arrays[plan[0][jDeployment][0]]['receivers']: selected.add(plan)
            elif dens1!=dens2: selected.add(plan)
    return selected


def single_array_timings(nrmsBool):
    """
    Create a set of array/timestep index combinations

    Parameters
    ----------
    nrmsBool : numpy array [n, m, o]
        NRMS boolean detect/no-detect values as a function of array,
        leakage scenario and timestep indexes
    Returns
    -------
    set of tuples

    """
    plans = set()
    for iArray in range(nrmsBool.shape[0]):
        for iTimeStep in range(nrmsBool.shape[2]):
            det = int(np.sum(nrmsBool[iArray, :, iTimeStep]))
            ttd = iTimeStep
            plan = (((iArray, iTimeStep),), det, ttd, tuple([int(x) for x in np.where(nrmsBool[iArray, :, iTimeStep])[0]]) )
            if plan[1] > 0:
                plans.add(plan)
    return plans


def additional_array_timings(plans_input, nrmsBool):
    """
    Create a set of array/timestep index combinations added on to the
    inputted monitoring plans

    Parameters
    ----------
    plans_input : list of tuples representing monitoring plans
    nrmsBool : numpy array [n, m, o]
        NRMS boolean detect/no-detect values as a function of array,
        leakage scenario and timestep indexes
    Returns
    -------
    set of tuples

    """
    #iPlan=0
    plans = set()
    for plan in plans_input:
        #iPlan+=1
        #print(iPlan,len(plans_input))
        for iArray in range(nrmsBool.shape[0]):
            for iTimeStep in range(nrmsBool.shape[2]):
                thisTTD = []
                for iScenario in range(nrmsBool.shape[1]):
                    thisScenarioTTD = []
                    for deployment in plan[0]+((iArray, iTimeStep),):
                        if nrmsBool[iArray, iScenario, deployment[1]]:
                            thisScenarioTTD += [deployment[1]]
                    if len(thisScenarioTTD) > 0:
                        thisTTD += [np.min(thisScenarioTTD)]
                if len(thisTTD) > 0:
                    thisTTD = float(np.mean(thisTTD))
                    if np.any(nrmsBool[iArray, :, iTimeStep]) or thisTTD < plan[2]:
                        thisDet = int(np.sum(nrmsBool[iArray, :, iTimeStep]))
                        skip = False
                        for jPlan in plans_input:
                            if jPlan[1] >= thisDet and jPlan[2] <= thisTTD:
                                skip = True
                                break
                        if not skip:
                            newPlan = (plan[0]+((iArray, iTimeStep),), thisDet, thisTTD, tuple([int(x) for x in np.where(nrmsBool[iArray, :, iTimeStep])[0]]) )
                            plans.add(newPlan)
    return plans
