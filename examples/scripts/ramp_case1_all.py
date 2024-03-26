# -*- coding: utf-8 -*-
"""
Combined scripts into one:
1. download_folder_content_on_edx.py
Script analyzes folder content (seismic_data or vp_model) and downloads selected
data files. User needs to copy their API-key into line 18 obtained from EDX to use
this script.

2. ramp_sys_seismic_monitoring_optimization_data.py

3. array_construction_nrms_processing.ipynb

@author: Veronika Vasylkivska (Veronika.Vasylkivska@NETL.DOE.GOV)
LRST (Battelle) supporting NETL

4. Added optimization code as well as json/yaml input/output files
@author: Alexander Hanna (alexander.hanna@pnnl.gov)
"""

import os
import sys
import re
import requests
import zipfile
import shutil
import json
import yaml
import pickle

import scipy
import scipy.spatial
import scipy.interpolate
from scipy.spatial import distance_matrix

import numpy as np
import itertools
import time

import subprocess
import h5py
import copy
import re
from tqdm import tqdm
import glob

import sklearn
import sklearn.decomposition

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
import matplotlib.gridspec as gridspec
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import networkx as networkx



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.sep.join(['..', '..', 'source']))
sys.path.insert(0, os.sep.join(['..', '..', 'src']))

from openiam import SystemModel

sys.path.insert(0, os.sep.join(['..', '..', 'ramp']))
sys.path.insert(0, os.sep.join(['..', '..', 'ramp', 'components','base']))

from ramp.utilities.data_readers import default_bin_file_reader

from ramp import SeismicDataContainer
from ramp import SeismicSurveyConfiguration
from ramp import SeismicMonitoring
from ramp.components.seismic.seismic_configuration import SeismicSurveyConfiguration, five_n_receivers_array_creator
from ramp.optimize.ttd_det_optimization import *

#from ramp import five_n_receivers_array_creator

#from ramp.data_container import default_bin_file_reader
#from ramp.seismic_data_container import SeismicDataContainer
#from ramp.seismic_configuration import SeismicSurveyConfiguration
#from ramp.seismic_monitoring import SeismicMonitoring
#from ramp.seismic_configuration import five_n_receivers_array_creator

def generate_minimal_valid_subgrids(gravBool, min_nodes, min_mx=1, max_mx=None, min_my=1, max_my=None):
    """
    Generate all sub-grids within a 2D array shape that meet a specified minimum number of nodes,
    based on user-specified step sizes (mx, my).
    """
    nreals, nrows, ncols = gravBool.shape
    if max_mx is None:
        max_mx = nrows
    if max_my is None:
        max_my = ncols

    grids = []
    for i0 in range(nrows):
        for j0 in range(ncols):
            for i1 in range(i0 + 1, nrows):  # Ensure i1 > i0
                for j1 in range(j0 + 1, ncols):  # Ensure j1 > j0
                    for mx in range(min_mx, min(max_mx + 1, i1 - i0 + 1)):
                        #for my in range(min_my, min(max_my + 1, j1 - j0 + 1)):
                        for my in [max_my]:
                            # Calculate the number of nodes in the sub-grid
                            num_nodes = ((i1 - i0) // mx + 1) * ((j1 - j0) // my + 1)
                            if num_nodes == min_nodes:
                                if np.any(gravBool[:,i0:(i1+1):mx,j0:(j1+1):my]):
                                    grids += [[i0,j0,i1,j1,mx,my]]
    return grids


def one_step_modified_grids(gravBool, grid, min_mx=1, max_mx=None, min_my=1, max_my=None, max_nodes=None):
    """
    Generate all valid grids that are a one-step modification from a given grid,
    within specified constraints.

    Parameters:
    - array_shape: The shape of the 2D array (nrows, ncols).
    - grid: The original grid specified as (i0, j0, i1, j1, mx, my).
    - min_mx, max_mx: Minimum and maximum values for mx.
    - min_my, max_my: Minimum and maximum values for my.
    """
    nreals, nrows, ncols = gravBool.shape
    i0, j0, i1, j1, mx, my = grid
    if max_mx is None:
        max_mx = nrows
    if max_my is None:
        max_my = ncols

    # Adjustments: i0, j0, i1, j1, mx, and my
    adjustments = [
        (-1, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), # i0 adjustments
        (0, -1, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), # j0 adjustments
        (0, 0, -1, 0, 0, 0), (0, 0, 1, 0, 0, 0), # i1 adjustments
        (0, 0, 0, -1, 0, 0), (0, 0, 0, 1, 0, 0), # j1 adjustments
        (0, 0, 0, 0, -1, 0), (0, 0, 0, 0, 1, 0), # mx adjustments
        (0, 0, 0, 0, 0, -1), (0, 0, 0, 0, 0, 1)  # my adjustments
    ]

    modified_grids = []
    for adj in adjustments:
        new_i0, new_j0, new_i1, new_j1, delta_mx, delta_my = [a+b for a, b in zip(grid, adj)]
        new_mx, new_my = mx + delta_mx, my + delta_my

        # Check constraints for the new grid
        if 0 <= new_i0 < nrows and 0 <= new_j0 < ncols and new_i1 < nrows and new_j1 < ncols and new_i0 < new_i1 and new_j0 < new_j1:
            if min_mx <= new_mx <= max_mx and min_my <= new_my <= max_my:

                num_nodes = ((i1 - i0) // mx + 1) * ((j1 - j0) // my + 1)
                if num_nodes<max_nodes:
                    if np.any(gravBool[:,i0:(i1+1):mx,j0:(j1+1):my]):
                        modified_grids.append((new_i0, new_j0, new_i1, new_j1, new_mx, new_my))

    return modified_grids


def subsample_to_n_points(points, n):
    """
    Subsample a set of points to the most uniformly-distributed n points,
    by returning the indexes of the sparsified points.

    :param points: List of (x, y) coordinates
    :param n: Desired number of points
    :return: List of indexes of subsampled points
    """
    points_array = np.array(points)
    points_array[:, 0] -= np.min(points_array[:, 0])
    points_array[:, 0] /= np.max(points_array[:, 0])
    points_array[:, 1] -= np.min(points_array[:, 1])
    points_array[:, 1] /= np.max(points_array[:, 1])
    indexes = np.arange(len(points_array))  # Create an array of indexes

    while len(points_array) > n:
        dists = distance_matrix(points_array, points_array)
        np.fill_diagonal(dists, np.inf)
        min_dist_idx = np.argmin(dists)
        delete_idx = np.unravel_index(min_dist_idx, dists.shape)[0]

        # Delete the point with the minimum distance and its index
        points_array = np.delete(points_array, delete_idx, axis=0)
        indexes = np.delete(indexes, delete_idx)

    return list(indexes)

def check_multi_source(arrays,array1,array2):
    if set(arrays[array1.iArray]['receivers'])==set(arrays[array2.iArray]['receivers']):
        if not arrays[array1.iArray]['source']==arrays[array2.iArray]['source']:
            if array1.iTime==array2.iTime: return True
    return False

def is_dominated(p, q):
    """Check if point p is dominated by point q."""
    return all(p_i >= q_i for p_i, q_i in zip(p, q)) and any(p_i > q_i for p_i, q_i in zip(p, q))

def naive_pareto_front(points):
    """Compute the Pareto front using a naive approach for small datasets."""
    pareto_front = []
    for p in points:
        if not any(is_dominated(p, q) for q in points):
            if not any(np.array_equal(p, q) for q in pareto_front):
                pareto_front.append(p)
    return np.array(pareto_front)

def merge_pareto_fronts(front1, front2):
    """Merge two Pareto fronts."""
    combined = np.vstack((front1, front2))
    return naive_pareto_front(combined)

def pareto_front_divide_and_conquer(points):
    """Compute the Pareto front using a divide-and-conquer approach."""
    if len(points) <= 50:  # For small datasets, use the naive approach
        return naive_pareto_front(points)
    mid = len(points) // 2
    left_front = pareto_front_divide_and_conquer(points[:mid])
    right_front = pareto_front_divide_and_conquer(points[mid:])
    return merge_pareto_fronts(left_front, right_front)

def add_sensor_combs(input_plans,satuBool,nrmsBool,maxWells,maxArrays):
    """
    Generate a list of monitoring plans based on the provided list of input plans.

    Args:
    inputs_plans (list): List of previously generated monitoring plans.
    satuBool: Numpy array (shape [realization_index,i,j,k,time]) of boolean
              values representing whether the saturation exceeds the threshold

    Returns:
    list: A list of monitoring plan objects.
    """

    input_plans = set(input_plans)
    nx = satuBool.shape[1]
    ny = satuBool.shape[2]
    nz = satuBool.shape[3]
    nt = satuBool.shape[4]
    output_plans = set()

    for plan in input_plans:

        # if we're at the maximum numbers of sensors, skip adding any more
        if len(plan.point_sensors)<int(inputs['max_pressure_sensors']):
            for i,j,k in itertools.product(range(nx),range(ny),range(nz)):

                # if we already have a sensor of this kind at this i,j,k location, we don't place another one
                existing_sensors_ijkt = [[sensor.well.i,sensor.well.j,sensor.k,sensor.type] for sensor in plan.point_sensors]
                if [i,j,k,'Pressure'] in existing_sensors_ijkt:
                    continue

                # if we're at the maximum numbers of wells, ie unique i,j combinations,
                # and this i,j combination isn't one of them, we don't put any sensors here
                if len(plan.wells)>=int(inputs['max_wells']):
                    if [i,j] not in [[well.i,well.j] for well in plan.wells]:
                        continue

                # if this i,j,k never exceeds the threshold for any leak scenario, don't put any sensors there
                if not np.any(presBool[:,i,j,k,:]):
                    continue

                for it in range(nt):

                    if np.any(presBool[:,i,j,k,it:]):
                        planNew = copy.deepcopy(plan)

                        planNew.add_point_sensor(i,j,k,it,'Pressure')
                        planNew.update_detections(nrmsBool,presBool,satuBool,gravBool)

                        # if the new monitoring plan detects some leak scenarios that the old one didn't, its a keeper
                        # if it detects the same scenarios but detects some of them sooner, it's a keeper
                        # if it detects fewer leaks, something is wrong
                        keep = False
                        if len(planNew.detections.difference(plan.detections))>0:  keep = True
                        elif (planNew.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)<plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)): keep = True
                        elif len(plan.detections.difference(planNew.detections))>0:
                            raise Exception("Error, the improved plan somehow detects fewer leaks than the plan it's based on")

                        if keep: output_plans.add(planNew)

        if len(plan.point_sensors)<int(inputs['max_saturation_sensors']):
            #output_plans.add(copy.deepcopy(plan))
            #continue

            for i,j,k in itertools.product(range(nx),range(ny),range(nz)):

                # if we already have a sensor of this kind at this i,j,k location, we don't place another one
                existing_sensors_ijkt = [[sensor.well.i,sensor.well.j,sensor.k,sensor.type] for sensor in plan.point_sensors]
                if [i,j,k,'Saturation'] in existing_sensors_ijkt:
                    continue

                # if we're at the maximum numbers of wells, ie unique i,j combinations,
                # and this i,j combination isn't one of them, we don't put any sensors here
                if len(plan.wells)>=int(inputs['max_wells']):
                    if [i,j] not in [[well.i,well.j] for well in plan.wells]:
                        continue

                # if this i,j,k never exceeds the threshold for any leak scenario, don't put any sensors there
                if not np.any(satuBool[:,i,j,k,:]):
                    continue

                for it in range(nt):

                    if np.any(satuBool[:,i,j,k,it:]):
                        planNew = copy.deepcopy(plan)

                        planNew.add_point_sensor(i,j,k,it,'Saturation')
                        planNew.update_detections(nrmsBool,presBool,satuBool,gravBool)

                        # if the new monitoring plan detects some leak scenarios that the old one didn't, its a keeper
                        # if it detects the same scenarios but detects some of them sooner, it's a keeper
                        # if it detects fewer leaks, something is wrong
                        keep = False
                        if len(planNew.detections.difference(plan.detections))>0:  keep = True
                        elif (planNew.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)<plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)): keep = True
                        elif len(plan.detections.difference(planNew.detections))>0:
                            raise Exception("Error, the improved plan somehow detects fewer leaks than the plan it's based on")

                        if keep: output_plans.add(planNew)

        # if we already have the maximum number of seismic arrays, don't add another one
        if len(plan.seismic_arrays)<int(inputs['max_seismic_arrays']):
            for iArray in range(nrmsBool.shape[0]):

                # if this seismic array never detects any leaks, don't add it
                if not np.any(nrmsBool[iArray,:,:]): continue

                for it in range(nt):

                    # if this seismic array doesn't detect any leaks at this timestep, don't add it
                    if np.any(nrmsBool[iArray,:,it]):

                        planNew = copy.deepcopy(plan)

                        planNew.add_seismic_array(iArray,it)
                        planNew.update_detections(nrmsBool,presBool,satuBool,gravBool)

                        # if the new monitoring plan detects some leak scenarios that the old one didn't, its a keeper
                        # if it detects the same scenarios but detects some of them sooner, it's a keeper
                        # if it detects fewer leaks, something is wrong
                        keep = False
                        if len(planNew.detections.difference(plan.detections))>0:  keep = True
                        elif (planNew.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)<plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)): keep = True
                        elif len(plan.detections.difference(planNew.detections))>0:
                            raise Exception("Error, the improved plan somehow detects fewer leaks than the plan it's based on")

                        if keep: output_plans.add(planNew)

        if len(plan.gravity_arrays)<int(inputs['max_gravity_surveys']):

            min_st = inputs['min_gravity_stations']
            max_st = inputs['max_gravity_stations']
            min_mx = inputs['min_gravity_spacing_x']
            max_mx = inputs['max_gravity_spacing_x']
            min_my = inputs['min_gravity_spacing_y']
            max_my = inputs['max_gravity_spacing_y']
            t0 = time.time()
            for it in range(nt):
                valid_minimal_grids = generate_minimal_valid_subgrids(gravBool[:,:,:,it], min_st, min_mx, max_mx, min_my, max_my)
                #print('it,len grids',it,len(valid_minimal_grids))
                for grid in valid_minimal_grids:
                    i0,j0,i1,j1,mx,my = grid
                    detections = [int(detection) for detection in np.where(np.any(gravBool[ :, i0:(i1+1):mx, j0:(j1+1):my, it ],axis=(1,2)))[0].tolist()]
                    if len(detections)>0:
                        planNew = copy.deepcopy(plan)
                        planNew.add_gravity_array(i0,j0,i1,j1,mx,my,it)
                        planNew.update_detections(nrmsBool,presBool,satuBool,gravBool)

                        # if the new monitoring plan detects some leak scenarios that the old one didn't, its a keeper
                        # if it detects the same scenarios but detects some of them sooner, it's a keeper
                        # if it detects fewer leaks, something is wrong
                        keep = False
                        if len(planNew.detections.difference(plan.detections))>0:  keep = True
                        elif (planNew.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)<plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)): keep = True
                        elif len(plan.detections.difference(planNew.detections))>0:
                            raise Exception("Error, the improved plan somehow detects fewer leaks than the plan it's based on")
                        if keep: output_plans.add(planNew)

            #print('plan.gravity_arrays',plan.gravity_arrays)
            for gravArray in plan.gravity_arrays:
                grid = [gravArray.i0,gravArray.j0,gravArray.i1,gravArray.j1,gravArray.mx,gravArray.my]
                it = gravArray.it
                valid_augmented_grids = one_step_modified_grids(gravBool[:,:,:,it], grid, min_mx, max_mx, min_my, max_my, max_st)
                for grid in valid_augmented_grids:
                    i0,j0,i1,j1,mx,my = grid
                    detections = [int(detection) for detection in np.where(np.any(gravBool[ :, i0:(i1+1):mx, j0:(j1+1):my, it ],axis=(1,2)))[0].tolist()]
                    if len(detections)>0:
                        planNew = copy.deepcopy(plan)
                        planNew.add_gravity_array(i0,j0,i1,j1,mx,my,it)
                        planNew.update_detections(nrmsBool,presBool,satuBool,gravBool)

                        # if the new monitoring plan detects some leak scenarios that the old one didn't, its a keeper
                        # if it detects the same scenarios but detects some of them sooner, it's a keeper
                        # if it detects fewer leaks, something is wrong
                        keep = False
                        if len(planNew.detections.difference(plan.detections))>0:  keep = True
                        elif (planNew.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)<plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)): keep = True
                        elif len(plan.detections.difference(planNew.detections))>0:
                            raise Exception("Error, the improved plan somehow detects fewer leaks than the plan it's based on")
                        if keep: output_plans.add(planNew)

        if len(plan.ert_arrays)<int(inputs['max_ert_arrays']): pass

    return list(output_plans)

def remove_iReal_tt(presIn,satuIn,nrmsIn,gravIn,iReals,tt):
    """
    Return a modified copy of the inputted detectability values.

    Set a specified list of leakage indicces equal to false, meaning those scenarios are not
    to be included in the optimization.

    Also set all timesteps before the current timestep equal to false, meaning its too late
    to add sensors at those times and so they should be ignored by the optimization.

    Args:
    satuIn: Numpy array [realization_index,i,j,k,t] of Boolean values
            representing whether the saturation exceeds the threshold
    iReas:  List or 1d numpy array of integers, representing the indicces of leakage scenarios
            to be included in the analysis
    tt:     An integer value representing the current timestep

    Returns:
    satuOut: Numpy array (shape [realization_index,i,j,k,time]) of Boolean values representing
             both whether the given realizations should be included in the analysis, and whether
             leakage detections are possible at each time and position
    """

    # un-comment to disable this feature
    return presIn,satuIn,nrmsIn,gravIn

    # construct a list of all leakage scenarios indicces not in iReals
    jReals = list(set(range(satuIn.shape[0]))-set(iReals))

    # make a deep copy of the saturation array
    presOut = copy.deepcopy(presIn)
    satuOut = copy.deepcopy(satuIn)
    nrmsOut = copy.deepcopy(nrmsIn)
    gravOut = copy.deepcopy(gravIn)

    # set all jReals equal to false
    presOut[jReals,:,:,:,:] = False
    satuOut[jReals,:,:,:,:] = False
    nrmsOut[:,jReals,:] = False
    gravOut[jReals,:,:,:] = False

    # set all timesteps earlier than tt to false
    if tt>0: presOut[:,:,:,:,:tt] = False
    if tt>0: satuOut[:,:,:,:,:tt] = False
    if tt>0: nrmsOut[:,:,:tt] = False
    if tt>0: gravOut[:,:,:tt] = False

    return presOut,satuOut,nrmsOut,gravOut

def compute_pareto(plans,presBool,satuBool,nrmsBool,gravBool):
    det  = np.array([len(plan.detections) for plan in plans])
    ttfd = np.array([plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool) for plan in plans])
    #dtim = np.array([plan.get_avg_commitment_time() for plan in plans])
    #points = np.array(list(zip(-det,ttfd,-dtim)))
    points = np.array(list(zip(-det,ttfd)))
    rank  = pareto_front_divide_and_conquer( points )
    #print('points',points)
    #print('points.shape',points.shape)
    #print('rank',rank)
    #print('rank.shape',rank.shape)
    #print(rank)
    #exit()
    return [int(np.where((points == item).all(axis=1))[0][0]) for item in rank]

def scatter2step(x,y):
    x_step = []
    y_step = []
    for i in range(len(x)):
      x_step += [x[i]]
      x_step += [x[i]]
      y_step += [y[i]]
      y_step += [y[i]]
    return x_step[1:], y_step[:-1]

def remove_future_wells(oldPlans,tt):
    newPlans = []
    for oldPlan in oldPlans:
        newPlan = MonitoringPlan()
        for well in oldPlan.wells:
            if well.drill_timestep<=tt:
                newPlan.wells.add(well)
        if len(newPlan.wells)>0:
            newPlans += [newPlan]
    if len(newPlans)==0: newPlans = [MonitoringPlan()]
    return newPlans


class Optimization:

    def __init__(self,inputs,presBool,satuBool,gravBool,nrmsBool):
        self.edx_api_key = inputs['edx_api_key']
        self.download_data = inputs['download_data']
        self.run_optimization = inputs['run_optimization']
        self.plot_results = inputs['plot_results']
        self.data_case = inputs['data_case']
        self.scenarios = inputs['scenarios']
        self.directory_seismic_data = inputs['directory_seismic_data']
        self.directory_velocity_data = inputs['directory_velocity_data']
        self.directory_nrms_data = inputs['directory_nrms_data']
        self.directory_plots = inputs['directory_plots']
        self.directory_simulation_data = inputs['directory_simulation_data']
        self.directory_output_files = inputs['directory_output_files']
        self.sources = inputs['sources']
        self.sourceNum = inputs['sourceNum']
        self.sourceMin = inputs['sourceMin']
        self.sourceMax = inputs['sourceMax']
        self.receiversNum = inputs['receiversNum']
        self.receiversMin = inputs['receiversMin']
        self.receiversMax = inputs['receiversMax']
        self.number_receivers = inputs['number_receivers']
        self.seismic_total_duration = inputs['seismic_total_duration']
        self.seismic_sampling_interval = inputs['seismic_sampling_interval']
        self.threshold_nrms = inputs['threshold_nrms']
        self.threshold_co2 = inputs['threshold_co2']
        self.threshold_pressure = inputs['threshold_pressure']
        self.threshold_gravity = inputs['threshold_gravity']
        self.max_pressure_sensors = inputs['max_pressure_sensors']
        self.max_saturation_sensors = inputs['max_saturation_sensors']
        self.max_wells = inputs['max_wells']
        self.max_seismic_arrays = inputs['max_seismic_arrays']
        self.max_gravity_surveys = inputs['max_gravity_surveys']
        self.max_ert_arrays = inputs['max_ert_arrays']
        self.min_gravity_stations = inputs['min_gravity_stations']
        self.max_gravity_stations = inputs['max_gravity_stations']
        self.min_gravity_spacing_x = inputs['min_gravity_spacing_x']
        self.max_gravity_spacing_x = inputs['max_gravity_spacing_x']
        self.min_gravity_spacing_y = inputs['min_gravity_spacing_y']
        self.max_gravity_spacing_y = inputs['max_gravity_spacing_y']
        self.number_proposals = inputs['number_proposals']
        self.fixed_wells = inputs['fixed_wells']
        self.stages = inputs['stages']

        self.presBool = presBool
        self.satuBool = satuBool
        self.gravBool = gravBool
        self.nrmsBool = nrmsBool

class Stage:

    def __init__(self,*args):
        if len(args)==1 and isinstance(args[0],dict):
            self.plans  = [MonitoringPlan(planDict) for planDict in args[0]['plans']]
            self.pareto = args[0]['pareto']
            self.iReals = args[0]['iReals']
            self.selected = [MonitoringPlan(planDict) for planDict in args[0]['selected']]
        else:
            self.plans  = args[0]
            self.pareto = args[1]
            self.iReals = args[2]
            self.selected = args[3]

    def __eq__(self,other):
        if self.plans==other.plans and self.pareto==other.pareto and self.iReals==other.iReals and self.selected==other.selected: return True
        else: return False

    def __hash__(self):
        return hash(frozenset([frozenset(self.plans),frozenset(self.pareto),frozenset(self.iReals),frozenset(self.selected)]))

    def to_dict(self):
        return {'plans': [plan.to_dict() for plan in self.plans],
                'pareto': self.pareto,
                'iReals': self.iReals,
                'selected': [plan.to_dict() for plan in self.selected]}


class MonitoringPlan:

    def __init__(self, *args):
        if len(args)==0:

            wells = []
            if 'none' not in inputs['fixed_wells']:
                for wellStr in inputs['fixed_wells'].split(';'):
                    #print(wellStr)
                    wellStr = wellStr.split(',')
                    if wellStr[0].strip()=='ijk':
                        i  = int(wellStr[1])
                        j  = int(wellStr[2])
                        k  = int(wellStr[3])
                        tt = int(wellStr[4])
                    elif wellStr[0].strip()=='xyz':
                        xi = float(wellStr[1])
                        yi = float(wellStr[2])
                        zi = float(wellStr[3])
                        ti = float(wellStr[4])
                        file = h5py.File(download_directory+'/sim%04i.h5'%1,'r')
                        x = np.array(file['data']['x'])
                        y = np.array(file['data']['y'])
                        z = np.array(file['data']['z'])
                        times = np.array(file['data']['times'])
                        i  = int(np.argsort(np.abs(x-xi))[0])
                        j  = int(np.argsort(np.abs(y-yi))[0])
                        k  = int(np.argsort(np.abs(z-zi))[0])
                        tt = int(np.argsort(np.abs(times-ti))[0])
                    wells += [Well(i,j,k,tt,[])]

            self.wells          = set(wells)
            self.seismic_arrays = set()
            self.gravity_arrays = set()
            self.ert_arrays     = set()
            self.point_sensors  = set()
            self.detections  = set()
        if len(args)==1 and isinstance(args[0],dict):
            self.wells  = set([Well(wellDict) for wellDict in args[0]['wells']])
            self.seismic_arrays = set([SeismicArray(arrayDict) for arrayDict in args[0]['seismic_arrays']])
            self.gravity_arrays = set([GravityArray(arrayDict) for arrayDict in args[0]['gravity_arrays']])
            self.ert_arrays = set([ERTArray(arrayDict) for arrayDict in args[0]['ert_arrays']])
            self.point_sensors = set()
            for well in self.wells:
                for sensor in well.point_sensors:
                    self.point_sensors.add(sensor)
            self.detections = set(args[0]['detections'])
        elif len(args)==4:
            self.wells          = set(args[0])
            self.seismic_arrays = set(args[1])
            self.gravity_arrays = set(args[2])
            self.ert_arrays     = set(args[3])
            self.point_sensors  = set(args[4])
            self.detections     = set(args[5])
        self.n_detected         = None
        self.avg_ttfd           = None
        self.avg_commitment_time = None

    def __eq__(self,other):
        if self.wells==other.wells and \
           self.point_sensors==other.point_sensors and \
           self.seismic_arrays==other.seismic_arrays and \
           self.gravity_arrays==other.gravity_arrays and \
           self.ert_arrays==other.ert_arrays:
             return True
        else: return False

    def __hash__(self):
        return hash(
                 frozenset([
                     frozenset(self.wells),
                     frozenset(self.point_sensors),
                     frozenset(self.seismic_arrays),
                     frozenset(self.gravity_arrays),
                     frozenset(self.ert_arrays),
                 ]))

    def to_dict(self):
        return {'wells':[well.to_dict(self.point_sensors) for well in self.wells],
                'seismic_arrays':[array.to_dict() for array in self.seismic_arrays],
                'gravity_arrays':[array.to_dict() for array in self.gravity_arrays],
                'ert_arrays':[array.to_dict() for array in self.ert_arrays],
                'detections':list(self.detections)}

    def update_detections(self,nrmsBool,presBool,satuBool,gravBool):
        self.detections = set()
        for sensor in self.point_sensors:     self.detections.update( sensor.detections(presBool,satuBool) )
        for seisArray in self.seismic_arrays: self.detections.update( seisArray.detections(nrmsBool) )
        for gravArray in self.gravity_arrays: self.detections.update( gravArray.detections(gravBool) )

    def add_point_sensor(self,i,j,k,it,type):
        wellExists = False
        for well in self.wells:
            if well.i==i and well.j==j:
                thisWell = well
                wellExists = True
                break
        if not wellExists: thisWell = Well(i,j,k,it,[])
        else:
            if thisWell.depth<k: thisWell.depth = k
            if thisWell.drill_timestep>it: thisWell.drill_timestep = it
        thisSensor = PointSensor(thisWell,k,it,type)
        if not wellExists: self.wells.add(thisWell)
        self.point_sensors.add(thisSensor)

    def add_seismic_array(self,iArray,it):
        thisArray = SeismicArray(iArray,it)
        self.seismic_arrays.add(thisArray)

    def add_gravity_array(self,i0,j0,i1,j1,mx,my,it):
        thisArray = GravityArray(i0,j0,i1,j1,mx,my,it)
        self.gravity_arrays.add(thisArray)

    def get_num_leaks_detected(self): return self.n_detected

    def compute_num_leaks_detected(self):
        self.update_detections(nrmsBool,presBool,satuBool,gravBool)
        self.n_detected = float(len(self.detections))

    def get_avg_ttfd(self,presBool,satuBool,nrmsBool,gravBool):
        #print('self.avg_ttfd',self.avg_ttfd)
        if self.avg_ttfd is None: self.compute_avg_ttfd(presBool,satuBool,nrmsBool,gravBool)
        return self.avg_ttfd

    def get_avg_commitment_time(self):
        if self.avg_commitment_time is None: self.compute_avg_commitment_time()
        return self.avg_commitment_time

    def compute_avg_ttfd(self,presBool,satuBool,nrmsBool,gravBool):
        ttfd = []
        if len(self.detections)==0:
            print('something is wrong computing avg ttfd!')
            print('len(self.detections)==0')
            print('wells',self.wells)
            for well in self.wells:
                print(well.i,well.j,well.depth,well.drill_timestep)
            print('sensors',self.point_sensors)
            for sensor in self.point_sensors:
                print(sensor.well,sensor.k,sensor.install_timestep,sensor.type)
            print('seismic arrays',self.seismic_arrays)
            print('gravity arrays',self.gravity_arrays)
            print('detections',self.detections)
            print('ttfd',ttfd)
            exit()

        #print('detections',self.detections)
        for detection in self.detections:
            #print('detection',detection)

            ttd = []
            for sensor in self.point_sensors:
                ttd += [sensor.time_to_first_detection(detection,presBool,satuBool)]

            for array in self.gravity_arrays:
                ttd += [array.time_to_first_detection(detection,gravBool)]

            for array in self.seismic_arrays:
                ttd += [array.time_to_first_detection(detection,nrmsBool)]

            if len(ttd)>0:
                ttfd += [np.nanmin(ttd)]

        if len(ttfd)==0:
            print('something is wrong computing avg ttfd!')
            print('wells',self.wells)
            for well in self.wells:
                print(well.i,well.j,well.depth,well.drill_timestep)
            print('sensors',self.point_sensors)
            for sensor in self.point_sensors:
                print(sensor.well,sensor.k,sensor.install_timestep,sensor.type)
            print('seismic arrays',self.seismic_arrays)
            for array in self.gravity_arrays:
                print('gravity array',array.i0,array.i1,array.j0,array.j1,array.mx,array.my,array.it)
                for detection in self.detections:
                    print(detection)
                    print(gravBool[detection,array.i0:(array.i1+1):array.mx,array.j0:(array.j1+1):array.my,array.it])
                    print(np.any(gravBool[ :, array.i0:(array.i1+1):array.mx, array.j0:(array.j1+1):array.my, array.it], axis=(1,2)))

            print('detections',self.detections)
            print('ttfd',ttfd)
            exit()

        self.avg_ttfd = np.mean(ttfd)

    def compute_avg_commitment_time(self):
        commitments = []
        for array in self.seismic_arrays: commitments += [time_array[array.iTime]/365.25]
        for array in self.gravity_arrays: commitments += [time_array[array.it]/365.25]
        for well in self.wells:   commitments += [times[well.drill_timestep]]
        self.avg_commitment_time = np.mean(commitments)

    #def compute_avg_plume_identification_potential(self,satuBool):
    #    pips = []
    #    for detection in self.detections:
    #        # 
    #    return np.mean(pips)

    def compute_avg_plume_delineation_potential(self,satuBool):
        pdps = []
        for detection in self.detections:
            for tt in range(satuBool.shape[4]):
                # collect xyz locations of sensors
                # only the ones that are deployed this timestep or earlier, if any
                # if only 0 or 1 sensors detecting, pdp=0 for this leak and timestep
                #   the difference between detecting a leak or not detecting it
                #   is important, but already accounted for by other objectives
                # if 2 sensors detecting, compute the line between them
                # if 3 sensors detecting, compute the polygon between them
                # if 4+ sensors detecting, compute the convex hull between them

                # collect shape/dimensions of the detectable plume at this timestep
                satuBool[detection,:,:,:,tt]

                # of course its always better to detect the plume in multiple locations than not
                # compute how much the plume and line/polygon/hull overlap
                # for 4+, count detectable nodes within convex hull, divide by number of nodes detectable

                # there's also some value in having some sensors that are just outside the plume,
                # that gives us useful delineation information as well

                pdps+=[]
        return np.mean(pdps)


class Well:

    def __init__(self,*args):
        if len(args)==1 and isinstance(args[0],dict):
            self.i = args[0]['i']
            self.j = args[0]['j']
            self.depth = args[0]['depth']
            self.drill_timestep = args[0]['drill_timestep']
            self.point_sensors = set([PointSensor(sensorDict,self) for sensorDict in args[0]['point_sensors']])
        elif len(args)==5:
            self.i = args[0]
            self.j = args[1]
            self.depth = args[2]
            self.drill_timestep = args[3]
            self.point_sensors = args[4]

    def __eq__(self,other):
        if self.i==other.i and \
           self.j==other.j and \
           self.depth==other.depth and \
           self.drill_timestep==other.drill_timestep: return True
        else: return False

    def __hash__(self):
        return hash((self.i,self.j,self.depth,self.drill_timestep))

    def to_dict(self,sensors):
        sensorsThis = []
        for sensor in sensors:
            if sensor.well==self:
                sensorsThis += [sensor.to_dict()]
        return { 'i':self.i,'j':self.j,'depth':self.depth,'drill_timestep':self.drill_timestep,'point_sensors':sensorsThis }


class PointSensor:

    def __init__(self,*args):
        if len(args)==2 and isinstance(args[0],dict) and isinstance(args[1],Well):
            self.well = args[1]
            self.k    = args[0]['k']
            self.install_timestep = args[0]['install_timestep']
            self.type = args[0]['type']
        elif len(args)==4:
            self.well = args[0]
            self.k    = args[1]
            self.install_timestep = args[2]
            self.type = args[3]
    def __eq__(self,other):
        if self.well==other.well and \
           self.k==other.k and \
           self.install_timestep==other.install_timestep and \
           self.type==other.type: return True
        else: return False

    def __hash__(self):
        return hash((self.well,self.k,self.install_timestep,self.type))

    def to_dict(self):
        return { 'k':self.k,'install_timestep':self.install_timestep,'type':self.type }

    def time_to_first_detection(self,detection,presBool,satuBool):
        ii = self.well.i
        jj = self.well.j
        kk = self.k
        tt = self.install_timestep
        if   self.type=='Pressure':   tds = np.where(presBool[detection,ii,jj,kk,:])[0]
        elif self.type=='Saturation': tds = np.where(satuBool[detection,ii,jj,kk,:])[0]
        if np.any(tds>=tt):
            return np.min(times[tds[tds>=tt]])
        else: return np.nan

    def detections(self,presBool,satuBool):
        if self.type=='Pressure':
            return set([int(detection) for detection in np.where(np.any(presBool[:,self.well.i,self.well.j,self.k,self.install_timestep:],axis=1))[0].tolist()])
        elif self.type=='Saturation':
            return set([int(detection) for detection in np.where(np.any(satuBool[:,self.well.i,self.well.j,self.k,self.install_timestep:],axis=1))[0].tolist()])


class SeismicArray:

    def __init__(self,*args):
        if len(args)==1 and isinstance(args[0],dict):
            self.iArray = args[0]['iArray']
            self.iTime  = args[0]['survey_timestep']
        else:
            self.iArray = args[0]
            self.iTime  = args[1]
    def __eq__(self,other):
        if self.iArray==other.iArray and \
           self.iTime==other.iTime: return True
        else: return False

    def __hash__(self):
        return hash((self.iArray,self.iTime))

    def to_dict(self):
        return { 'iArray':self.iArray,'survey_timestep':self.iTime }

    def time_to_first_detection(self,detection,nrmsBool):
        iArray = self.iArray
        iTime  = self.iTime
        if nrmsBool[iArray,detection,iTime]: return time_array[iTime]/365.25
        else: return np.nan

    def detections(self,nrmsBool):
        return set([int(detection) for detection in np.where(nrmsBool[self.iArray,:,self.iTime])[0].tolist()])


class GravityArray:

    def __init__(self,*args):
        if len(args)==1 and isinstance(args[0],dict):
            self.i0 = args[0]['i0']
            self.j0 = args[0]['j0']
            self.i1 = args[0]['i1']
            self.j1 = args[0]['j1']
            self.mx = args[0]['mx']
            self.my = args[0]['my']
            self.it = args[0]['it']
        else:
            self.i0 = args[0]
            self.j0 = args[1]
            self.i1 = args[2]
            self.j1 = args[3]
            self.mx = args[4]
            self.my = args[5]
            self.it = args[6]
    def __eq__(self,other):
        if self.i0==other.i0 and \
           self.j0==other.j0 and \
           self.i1==other.i1 and \
           self.j1==other.j1 and \
           self.mx==other.mx and \
           self.my==other.my and \
           self.it==other.it: return True
        else: return False

    def __hash__(self):
        return hash((self.i0,self.j0,self.i1,self.j1,self.mx,self.my,self.it))

    def to_dict(self):
        return { 'i0':int(self.i0),'j0':int(self.j0),'i1':int(self.i1),'j1':int(self.j1),'mx':int(self.mx),'my':int(self.my),'it':int(self.it) }

    def time_to_first_detection(self,detection,gravBool):
        i0 = self.i0
        j0 = self.j0
        i1 = self.i1
        j1 = self.j1
        mx = self.mx
        my = self.my
        it = self.it
        if np.any(gravBool[detection,i0:(i1+1):mx,j0:(j1+1):my,it]): return times[it]
        else: return np.nan

    def detections(self,gravBool):
        return set([int(detection) for detection in np.where(np.any(gravBool[ :, self.i0:(self.i1+1):self.mx, self.j0:(self.j1+1):self.my, self.it ],axis=(1,2)))[0].tolist()])





if len(sys.argv) == 1:
  raise Exception('''Please include an input argument specifying the YAML or JSON filename.
Example:
>> python3 %s inputs.json
'''%sys.argv[0])

try:
    inputs = json.load(open(sys.argv[1], 'r'))
except:
    try:
        inputs = yaml.safe_load(open(sys.argv[1], 'r'))
    except: ValueError


# ====================================
# =========== Setup Step =============
# ====================================
# Duplicate options commented out in detailed setup below
# Use your API-Key
api_key = inputs['edx_api_key']

# Choose what data needs to be downloaded
data_case = inputs['data_case']  # 1 is seismic data, 2 is velocity data

# Choose scenarios to download
#scenario_indices = list(range(1, 992))  # e.g., list(range(51, 201)) requests files from 51 to 200
#scenario_indices = list(range(11, 14))  # e.g., list(range(51, 201)) requests files from 51 to 200
#print(inputs['scenarios'])
#print(type(inputs['scenarios']))
#if '-' in inputs['scenarios']: print(inputs['scenarios'])
if isinstance(inputs['scenarios'], int):
    scenario_indices = list(range(1, inputs['scenarios']+1))
elif isinstance(inputs['scenarios'], list):
    scenario_indices = inputs['scenarios']
elif isinstance(inputs['scenarios'], str):
    if '-' in inputs['scenarios']:
        scenario_indices = list(range(int(inputs['scenarios'].split('-')[0]),
                                      1+int(inputs['scenarios'].split('-')[1])))
    else:
        raise Exception('Error, scenarios list is not formatted correctly')
else:
    raise Exception('Error, scenarios list is not formatted correctly')

#scenario_indices = inputs['scenarios']  # e.g., list(range(51, 201)) requests files from 51 to 200

# Setup whether downloaded files should be unzipped
to_unzip = True  # False means do not unzip, True means unzip after download

# Setup whether archives will be deleted after unzipping
to_delete = True  # False: do not delete archive, True: delete archives
# ====================================
# =========== End Setup ==============
# ====================================

if __name__ == "__main__":
    # Start of script from download_folder_content_on_edx.py
    # Use your API-Key
    # api_key = ""

    headers = {"EDX-API-Key": api_key, "User-Agent": "achanna"}

    workspace_id = 'nrap-task-4-monitoring'
    # Kimb1.2 for RAMP folder (https://edx.netl.doe.gov/workspace/resources/nrap-task-4-monitoring?folder_id=b1ecb785-e1f9-46bc-9d04-7b943d3fe379)
    # has 2 subfolders: seismic_data and vp_model
    # Both subfolders have 986 archives with data files
    # Switching between the subfolders allows to download the needed files
    # using scripts rather than manually

    # ====================================
    # =========== Setup Step 1 ===========
    # ====================================
    # Choose what data needs to be downloaded
    # data_case = 1  # 1 is seismic data, 2 is velocity data
    if data_case == 1:
        folder_id = '9a032568-5977-494c-a5b1-a903704104e4'  # seismic_data folder id
    elif data_case == 2:
        folder_id = '2e5d0a00-281f-45e2-bc97-c6fef29d9e9b'  # vp_model folder id
    else:
        err_msg = 'Script is not setup for a data_case {}'.format(data_case)
        raise ValueError(err_msg)

    # ====================================
    # =========== Setup Step 2 ===========
    # ====================================
    # Setup indices of data files to be downloaded
    # scenario_indices = [12, 109, 302, 622, 141, 318, 881, 986, 22, 76, 269]
    # scenarios with incomplete data [449, 518, 136, 970, 397, 590, 150, 598, 863, 37, 935, 937, 749, 302, 686, 500, 245, 182, 118, 312, 313, 315, 316]
    # missing_scenario_indices = [312, 313, 315, 316, 500]
    # scenario_indices = [12, 109, 622, 141, 318, 881, 986, 22, 76, 269] #[37, 118, 136, 150, 182, 245, 302, 303] #
    # scenario_indices = list(range(555,556))  # e.g., list(range(51, 201)) requests files from 51 to 200
    # scenario_indices = [37, 118, 136, 150, 182, 245, 397, 449, 452, 454, 455, 458, 503, 505, 508, 509, 516, 526, 539, 542, 545, 548, 550, 555, 559, 560, 562, 564, 566, 567, 576, 580, 583, 588, 589, 598, 590, 605, 607, 609, 612, 614, 615, 617, 621, 625, 640, 646, 648, 649, 650, 653, 686, 743, 744, 746, 747, 748, 749, 831, 832, 833, 834, 835, 836, 837, 839, 840, 842, 843, 863, 935, 937, 970, 991]
    # Print names of the requested files
    # Define file name format
    if data_case == 1:
        base_name = 'data_sim{:04}'
    elif data_case == 2:
        base_name = 'vp_sim{:04}'
    file_name = base_name + '.zip'
    print('The names of the requested files:')
    for ind in scenario_indices:
        print(file_name.format(ind))
    print('')

    # ====================================
    # =========== Setup Step 3 ===========
    # ====================================
    # Setup whether downloaded files should be unzipped
    # to_unzip = True  # False means do not unzip, True means unzip after download

    # ====================================
    # =========== Setup Step 4 ===========
    # ====================================
    # Setup output directory where the downloaded and unzipped files will be stored
    #output_directory = os.sep.join(['..', '..', 'data', 'user'])
    if data_case == 1:
        output_directory = os.sep.join([inputs['directory_seismic_data']])
    elif data_case == 2:
        output_directory = os.sep.join([inputs['directory_velocity_data']])

    download_directory = os.sep.join([inputs['directory_simulation_data']])
    if not os.path.exists(download_directory):
        os.mkdir(download_directory)


    step_ind = 1
    if inputs['download_data']:
        print('Step {}: Downloading data from EDX...'.format(step_ind))
        step_ind = step_ind + 1
        # ====================================
        # =========== Setup Step 5 ===========
        # ====================================
        # Setup whether archives will be deleted after unzipping
        # to_delete = True  # False: do not delete archive, True: delete archives

        data = {
            "workspace_id": workspace_id,
            "folder_id": folder_id,
            # "folder_id": ['7c3e598c-3486-468d-9d5b-9eee6da7637a', '4d932b54-57be-404a-b38e-6e9caa6e3b23']  list of folders format
            # "only_show_type": 'folders' # Uncomment this line if you wish to only return folders
            # "only_show_type": 'resources' # Uncomment this line if you wish to only return resources
        }

        # The following link stays the same even for different workspaces
        # This is an URL to API endpoint
        url = 'https://edx.netl.doe.gov/api/3/action/folder_resources'

        # Get data associated with folder
        r = requests.post(
            url, # URL to API endpoint
            headers=headers, # Headers dictionary
            data=data, # Dictionary of data params
            )

        # Convert data into dictionary format
        json_data = r.json()
        #print(json_data)
        #print(json_data.keys())

        # Get folder resources: files names, their urls, etc.
        resources = json_data['result']['resources']

        # Print number of resources to see expected number of 986 resources
        print('Total number of resources', len(resources), '\n')
        # for res in resources:
        #     print(res['name'])

        # Get URL of files to be downloaded
        urls = {}
        # Go over all resources in the folder
        for res in resources:
            if data_case == 1:
                scen_ind = int(res['name'][8:12])  # valid for seismic_data
            elif data_case == 2:
                scen_ind = int(res['name'][6:10])  # valid for vp_model
            if scen_ind in scenario_indices:
                urls[scen_ind] = res['url']

        # Download files
        for scen_ind, url_link in urls.items():
            print(f'scen_ind: {scen_ind}')
            if os.path.exists(os.path.join(output_directory, base_name.format(scen_ind)+'.zip')) or \
                    os.path.exists(os.path.join(output_directory, base_name.format(scen_ind))):
                print('Skipping file', base_name.format(scen_ind))
                continue

            print('Downloading file:', file_name.format(scen_ind))
            print('---')

            # print("Getting resource...")
            r = requests.get(url_link, headers=headers)

            fname = ''

            if "Content-Disposition" in r.headers.keys():
                fname = re.findall("filename=(.+)", r.headers["Content-Disposition"])[0]
            else:
                fname = url_link.split("/")[-1]

            if fname.startswith('"'):
                fname = fname[1:]

            if fname.endswith('"'):
                fname = fname[:-1]

            with open(os.sep.join([output_directory, fname]), 'wb') as file:
                file.write(r.content)

        # Unzip all archives if user has requested it
        if to_unzip:
            for scen_ind in urls:
                print('Unzipping {} file ...'.format(file_name.format(scen_ind)))
                path_name = os.sep.join([output_directory, file_name.format(scen_ind)])
                try:
                    with zipfile.ZipFile(path_name, 'r') as zip_ref:
                        folder_to_extract = os.sep.join([output_directory,
                                                         base_name.format(scen_ind)])
                        try:
                            os.mkdir(folder_to_extract)
                            zip_ref.extractall(folder_to_extract)
                        except FileExistsError:
                            pass

                    if to_delete:
                        print('Removing {} file ...'.format(file_name.format(scen_ind)))
                        print('---')
                        os.remove(path_name)
                except:
                    pass

        # Check downloads for complete data
        # Attempt second download for missing/incomplete files
        incomplete = []
        for scen_ind in scenario_indices:
            files = os.listdir(os.sep.join([output_directory,
                                            base_name.format(scen_ind), 'data']))
            if len(files) != 20: # could check for specific files if necessary
                incomplete.append(scen_ind)
        print(f'incomplete scenarios (1st round): {incomplete}')

        # Delete incomplete data folders
        for scen_ind in incomplete:
            files = os.sep.join([output_directory, base_name.format(scen_ind)])
            if os.path.isdir(files):
                shutil.rmtree(files)

        # Run download script again on deleted files
        # Get URL of files to be downloaded
        urls = {}
        # Go over all resources in the folder
        for res in resources:
            if data_case == 1:
                scen_ind = int(res['name'][8:12])  # valid for seismic_data
            elif data_case == 2:
                scen_ind = int(res['name'][6:10])  # valid for vp_model
            if scen_ind in incomplete: ### Only change to download script is list of files ###
                urls[scen_ind] = res['url']

        # Download files
        for scen_ind, url_link in urls.items():
            print(f'scen_ind: {scen_ind}')
            if os.path.exists(os.path.join(output_directory, base_name.format(scen_ind)+'.zip')) or \
                    os.path.exists(os.path.join(output_directory, base_name.format(scen_ind))):
                print('Skipping file', base_name.format(scen_ind))
                continue

            print('Re-downloading file:', file_name.format(scen_ind))
            print('---')

            # print("Getting resource...")
            r = requests.get(url_link, headers=headers)

            fname = ''

            if "Content-Disposition" in r.headers.keys():
                fname = re.findall("filename=(.+)", r.headers["Content-Disposition"])[0]
            else:
                fname = url_link.split("/")[-1]

            if fname.startswith('"'):
                fname = fname[1:]

            if fname.endswith('"'):
                fname = fname[:-1]

            with open(os.sep.join([output_directory, fname]), 'wb') as file:
                file.write(r.content)

        # Unzip all archives if user has requested it
        if to_unzip:
            for scen_ind in urls:
                print('Unzipping {} file ...'.format(file_name.format(scen_ind)))
                path_name = os.sep.join([output_directory, file_name.format(scen_ind)])
                try:
                    with zipfile.ZipFile(path_name, 'r') as zip_ref:
                        folder_to_extract = os.sep.join([output_directory,
                                                         base_name.format(scen_ind)])
                        try:
                            os.mkdir(folder_to_extract)
                            zip_ref.extractall(folder_to_extract)
                        except FileExistsError:
                            pass

                    if to_delete:
                        print('Removing {} file ...'.format(file_name.format(scen_ind)))
                        print('---')
                        os.remove(path_name)
                except:
                    pass

        headers = {"EDX-API-Key": inputs['edx_api_key']}

        data = {
            'workspace_id': 'nrap-task-4-monitoring',
            'folder_id': 'b1ecb785-e1f9-46bc-9d04-7b943d3fe379',
        }

        url = 'https://edx.netl.doe.gov/api/3/action/folder_resources'

        r = requests.post(
            url, # URL to API endpoint
            headers=headers, # Headers dictionary
            data=data, # Dictionary of data params
        )

        json_data = r.json()
        resources = json_data['result']['resources']

        fname = 'kimb12_h5s.zip'
        for resource in resources:
            if resource['name']==fname:
                r = requests.get( resource['url'], headers=headers, stream=True )
                total_size_in_bytes= int(r.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(os.sep.join([download_directory,fname]), 'wb') as file:
                    for data in r.iter_content(block_size):
                        progress_bar.update(len(data))
                        file.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    print("ERROR, something went wrong")
        p = subprocess.Popen(['unzip','-o', fname], cwd=download_directory)

    if inputs['download_data'] or inputs['run_optimization'] or inputs['plot_results']:
        print('Step {}: Performing check of the downloaded data...'.format(step_ind))
        step_ind = step_ind + 1
        # Check 2nd round of downloads for incomplete data to exclude from NRMS calculations
        excluded = []
        for scen_ind in scenario_indices:
            files = os.listdir(os.sep.join([output_directory, base_name.format(scen_ind), 'data']))
            if len(files) != 20: # could check for specific files if necessary
                excluded.append(scen_ind)
        print(excluded)

        if data_case == 1:
            #Start of code from ramp_sys_seismic_monitoring_optimization_data.py
            # Define keyword arguments of the system model
            time_points = 10*np.arange(1, 21)
            time_array = 365.25*time_points
            sm_model_kwargs = {'time_array': time_array}   # time is given in days

            # Setup required information for data container before creating one
            obs_name = 'seismic'
            #data_directory = os.path.join('..', '..', 'data', 'user', 'seismic')
            #output_directory = os.path.join('..', '..', 'examples', 'user', 'output',
            #                                'ramp_sys_seismic_monitoring_optimization_data')
            data_directory = inputs['directory_seismic_data']
            output_directory = inputs['directory_nrms_data']
            if not os.path.exists(output_directory):
                os.mkdir(output_directory)
            data_reader = default_bin_file_reader
            data_reader_kwargs = {'data_shape': (1251, 101, 9),
                                'move_axis_destination': [-1, -2, -3]}

            num_time_points = len(time_points)
            # excluded = [37, 118, 136, 150, 182, 245]  # 6 scenarios
            # excluded = [37, 118, 136, 150, 182, 245, 397, 449, 456, 457, 468, 469, 498, 499, 500, 590, 598, 686, 749, 831, 832, 833, 834, 835, 836, 837, 839, 840, 842, 843, 839, 863, 935, 937, 970, 991]  # 17 scenarios
            #job 477 = 468, maybe 467 with exclusions # 469?
            #job 508 = 499, maybe 498, 500?

            scenarios = set(scenario_indices).difference(excluded)
            scenarios = list(scenarios)

            num_scenarios = len(scenarios)
            family = 'seismic'
            data_setup = {}
            for scen in scenarios:
                data_setup[scen] = {'folder': os.path.join('data_sim{:04}'.format(scen), 'data')}
                for t_ind, tp in enumerate(time_points):
                    data_setup[scen]['t{}'.format(t_ind+1)] = 'data_sim{:04}_t{}.bin'.format(scen, tp)
            baseline = True

            '''# Define coordinates of sources
            num_sources = 9
            sources = np.c_[4000 + np.array([240, 680, 1120, 1600, 2040, 2480, 2920, 3400, 3840]),
                            np.zeros(num_sources),
                            np.zeros(num_sources)]

            # Define coordinates of receivers
            num_receivers = 101
            receivers = np.c_[4000 + np.linspace(0, 4000, num=num_receivers),
                            np.zeros(num_receivers),
                            np.zeros(num_receivers)]'''

            if 'sources' in inputs.keys():
                num_sources = len(inputs['sources'])
                sources = np.c_[inputs['sources'],
                                np.zeros(num_sources),
                                np.zeros(num_sources)]
            else:
                num_sources = inputs['sourcesNum']
                min_sources = inputs['sourcesMin']
                max_sources = inputs['sourcesMax']
                sources = np.c_[np.linspace(min_sources, max_sources, num=num_sources),
                                np.zeros(num_sources),
                                np.zeros(num_sources)]

            if 'receivers' in inputs.keys():
                num_receivers = len(inputs['receivers'])
                receivers = np.c_[inputs['receivers'],
                                  np.zeros(num_receivers),
                                  np.zeros(num_receivers)]
            else:
                num_receivers = inputs['receiversNum']
                min_receivers = inputs['receiversMin']
                max_receivers = inputs['receiversMax']
                receivers = np.c_[np.linspace(min_receivers, max_receivers, num=num_receivers),
                                  np.zeros(num_receivers),
                                  np.zeros(num_receivers)]

        # load one HDF5 file into memory to get the shape of the dataset, min/max values etc
        file = h5py.File(download_directory+'/sim%04i.h5'%1,'r')
        nx,ny,nz=np.array(file['plot0']['pressure']).shape
        nt = 12
        vx = np.array(file['data']['vertex-x'])
        vy = np.array(file['data']['vertex-y'])
        vz = np.array(file['data']['vertex-z'])
        x = np.array(file['data']['x'])
        y = np.array(file['data']['y'])
        z = np.array(file['data']['z'])
        times = np.array(file['data']['times'])
        #print(time_array/365.25)
        #print(times)
        #exit()

        xmin = np.min(vx)
        xmax = np.max(vx)
        ymin = np.min(vy)
        ymax = np.max(vy)
        zmin = np.min(vz)
        zmax = np.max(vz)

        # use that info to construct the numpy arrays the leak scenarios will be loaded into
        # then read in all HDF5 files
        nn=len(scenarios)
        pres=np.zeros([nn,nx,ny,nz,nt])
        satu=np.zeros([nn,nx,ny,nz,nt])
        grav=np.zeros([nn,nx,ny,nt])
        for n in range(nn):
            print('Loading Scenario %i'%scenarios[n])
            file = h5py.File(download_directory+'/sim%04i.h5'%scenarios[n],'r')
            for it in range(12):
                pres[n,:,:,:,it] = np.array(file['plot%i'%it]['pressure'])/1e3
                satu[n,:,:,:,it] = np.array(file['plot%i'%it]['saturation'])*100
                grav[n,:,:,it]   = np.abs(np.array(file['plot%i'%it]['gravity'])[1:,1:]).T

        #print( np.min(satu),np.mean(satu),np.max(satu) )
        presBool = pres>inputs['threshold_pressure']
        satuBool = satu>inputs['threshold_co2']
        gravBool = grav>inputs['threshold_gravity']

        # remove any simulations that never exceed the detection threshold
        #ii = np.where(np.any(satuBool,axis=(1,2,3,4)))[0]
        #satuBool = satuBool[ii,:,:,:,:]
        nn = satuBool.shape[0]

        print('Step {}: Processing seismic data into NRMS values...'.format(step_ind))
        step_ind = step_ind + 1
        # Create survey with defined coordinates
        survey_config = SeismicSurveyConfiguration(sources, receivers, name='Test Survey')

        # ------------- Create system model -------------
        sm = SystemModel(model_kwargs=sm_model_kwargs)

        # ------------- Add data container -------------
        dc = sm.add_component_model_object(
            SeismicDataContainer(name='dc', parent=sm, survey_config=survey_config,
                                total_duration=inputs['seismic_total_duration'],
                                sampling_interval=inputs['seismic_sampling_interval'],
                                family=family, obs_name=obs_name,
                                data_directory=data_directory, data_setup=data_setup,
                                time_points=time_points, baseline=baseline,
                                data_reader=data_reader,
                                data_reader_kwargs=data_reader_kwargs,
                                presetup=True))
        # Add parameters of the container
        dc.add_par('index', value=scenarios[0],
                discrete_vals=[scenarios, num_scenarios*[1/num_scenarios]])
        # Add gridded observation
        # dc.add_grid_obs(obs_name, constr_type='matrix', output_dir=output_directory)
        # dc.add_grid_obs('delta_{}'.format(obs_name), constr_type='matrix',
        #                 output_dir=output_directory)

        # ------------- Add seismic monitoring technology -------------
        smt = sm.add_component_model_object(
            SeismicMonitoring(name='smt', parent=sm, survey_config=survey_config, time_points=time_points))
        # Add keyword arguments linked to the seismic data container outputs
        smt.add_kwarg_linked_to_obs('data', dc.linkobs['seismic'], obs_type='grid')
        smt.add_kwarg_linked_to_obs('baseline', dc.linkobs['baseline_seismic'], obs_type='grid')
        # Add gridded observation
        smt.add_grid_obs('NRMS', constr_type='matrix', output_dir=output_directory)
        # Add scalar observations
        for nm in ['ave_NRMS', 'max_NRMS', 'min_NRMS']:
            smt.add_obs(nm)
            smt.add_obs_to_be_linked(nm)

        print('--------------------------------')
        print('Stochastic simulation started...')
        print('--------------------------------')
        print('Number of scenarios: {}'.format(num_scenarios))
        # Create sampleset varying over scenarios: this is not a typical setup
        # We want to make sure scenarios are not repeated.
        samples = np.array(scenarios).reshape(num_scenarios, 1)
        s = sm.create_sampleset(samples)

        results = s.run(cpus=5, verbose=False)
        print('--------------------------------')
        print('Stochastic simulation finished.')
        print('--------------------------------')

        print('--------------------------------')
        print('Collecting results...')
        print('--------------------------------')
        # Get saved gridded observations from files
        nrms = np.zeros((num_scenarios, num_time_points, num_sources, num_receivers))
        time_indices = list(range(num_time_points))

        for rlzn_number in range(1, num_scenarios+1):
            print('Realization {}'.format(rlzn_number))
            data = sm.collect_gridded_observations_as_time_series(
                smt, 'NRMS', output_directory, indices=time_indices,
                rlzn_number=rlzn_number) # data shape (num_time_points, num_sources, num_receivers)
            nrms[rlzn_number-1] = data

        file_to_save = os.path.join(
            output_directory,
            'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=nrms)

        data_check = 0
        if data_check:
            # Read file
            file_to_read = os.path.join(
                output_directory,
                'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
            d = np.load(file_to_read)
            # Determine shape of the data
            data_shape = d['data'].shape
            print('data_shape',data_shape)  # (36, 20, 9, 101)
            nrms = d['data']

            # Check that the data makes sense
            fig = plt.figure(figsize=(12, 5))
            ax = fig.add_subplot(111)
            ax_im = ax.imshow(nrms[0, 8, :, :], aspect='auto')

            # Set title
            title = ax.set_title('NRMS at {} years'.format(90))

            # Set x-labels
            x = np.linspace(0, 100, num=11)
            xlabels = np.linspace(1, 101, num=11, dtype=int)
            ax.set_xticks(x, labels=xlabels)
            ax.set_xlabel('Receivers')

            # Set y-labels
            y = np.linspace(0, 8, num=9)
            ylabels = np.linspace(1, 9, num=9, dtype=int)
            ax.set_yticks(y, labels=ylabels)
            ax.set_ylabel('Sources')

            # Add colorbar
            cbar = plt.colorbar(ax_im, label='Percentage, %')
            fig.tight_layout()

        step_ind = step_ind + 1
        # Start of array_construction_nrms_processing.ipynb
        # Setup directories
        #data_directory = os.path.join('..', 'user', 'output', 'ramp_sys_seismic_monitoring_optimization_data')
        #output_directory = os.path.join('..', 'user', 'output', 'ramp_sys_seismic_monitoring_optimization_data')
        data_directory = inputs['directory_nrms_data']
        output_directory = inputs['directory_nrms_data']

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # Create survey configuration with defined coordinates
        array_creator_kwargs = {'source_coords': sources,
                                'receiver_coords': receivers}
        configuration = SeismicSurveyConfiguration(
            sources, receivers, name='Test Survey', create_arrays=True,
            array_creator=five_n_receivers_array_creator,
            array_creator_kwargs=array_creator_kwargs)
        print('Number of created arrays:', configuration.num_arrays)

        # Load NRMS data
        nrms_data_file = os.path.join(data_directory, 'nrms_optimization_data_{}_scenarios.npz'.format(num_scenarios))
        d = np.load(nrms_data_file)
        # Determine shape of the data
        data_shape = d['data'].shape
        print('data_shape (opt)',data_shape)  # (300, 20, 9, 101)  scenarios, time points, sources, receivers
        nrms = d['data']

        # Setup data that will hold results of processing NRMS data for all created arrays
        arrays_nrms = np.zeros((configuration.num_arrays,
                                num_scenarios,
                                num_time_points,
                                3)) # 3 is for number of largest NRMS values

        # Process NRMS data for each array in the set
        for array_ind in configuration.arrays:
            sind = configuration.arrays[array_ind]['source']
            rind = configuration.arrays[array_ind]['receivers']
            # Get subset of NRMS data for a given array
            subset_nrms = nrms[:, :, sind, rind]
            # Sort numbers in increasing order
            sorted_subset_nrms = np.sort(subset_nrms)
            # Keep the largest three nrms associated with a given array
            arrays_nrms[array_ind, :, :, :] = sorted_subset_nrms[:, :, -3:]

        # Save arrays_nrms data
        file_to_save = os.path.join(
            output_directory,
            'arrays_nrms_data_3max_values_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=arrays_nrms)

        sub_arrays_nrms = arrays_nrms[:, :, :, 0]
        file_to_save = os.path.join(
            output_directory,
            'arrays_nrms_data_3rd_max_value_{}_scenarios.npz'.format(num_scenarios))
        np.savez_compressed(file_to_save, data=sub_arrays_nrms)
        #print(arrays_nrms.shape)
        #print(sub_arrays_nrms.shape)

        num_arrays = configuration.num_arrays
        produced_arrays = configuration._arrays
        threshold = inputs['threshold_nrms']
        nrms = sub_arrays_nrms
        nrmsBool = np.array(sub_arrays_nrms>threshold, dtype='bool')

        detectableDepths = []
        detectableDepths_shallow = []
        detectableDepths_deep = []
        countAll = []
        countShallow = []
        countMid = []
        countDeep = []
        for iReal in range(satuBool.shape[0]):
            newDets = z[np.where(np.any(satuBool[iReal,:,:,:,:],axis=(0,1,3)))[0]].tolist()
            detectableDepths += newDets
            if len(newDets)>0:
                if np.max(newDets)<700:
                    detectableDepths_shallow += newDets
                    countShallow += [iReal]
                elif np.min(newDets)>700:
                    detectableDepths_deep += newDets
                    countDeep += [iReal]
                else:
                    countMid += [iReal]
                countAll += [iReal]

    if inputs['run_optimization']:

        nRec = np.array([len(array['receivers']) for array in configuration.arrays.values()])

        if isinstance(inputs['number_receivers'],list):
            rmin = inputs['number_receivers'][0]
            rmax = inputs['number_receivers'][1]
            ii = np.where(nRec<rmin)
            nrmsBool[ii,:,:] = 0
            ii = np.where(nRec>rmax)
            nrmsBool[ii,:,:] = 0
        elif '<=' in inputs['number_receivers']:
            rbound = float(inputs['number_receivers'].split('<=')[1])
            ii = np.where(nRec>rbound)
            nrmsBool[ii,:,:] = 0
        elif '>=' in inputs['number_receivers']:
            rbound = float(inputs['number_receivers'].split('>=')[1])
            ii = np.where(nRec<rbound)
            nrmsBool[ii,:,:] = 0
        elif '<' in inputs['number_receivers']:
            rbound = float(inputs['number_receivers'].split('<')[1])
            ii = np.where(nRec>=rbound)
            nrmsBool[ii,:,:] = 0
        elif '>' in inputs['number_receivers']:
            rbound = float(inputs['number_receivers'].split('>')[1])
            ii = np.where(nRec<=rbound)
            nrmsBool[ii,:,:] = 0

        totalPlansBuilt = 0
        totalMultiSourcePlansBuilt = 0
        stages = []
        for tt in np.array(inputs['stages'].split(','),dtype='int'):

            if tt==0:
                iReals = np.sort(np.random.choice(list(range(nn)),nn//2,replace=False)).tolist()
                input_plans = [MonitoringPlan()]
            else:
                iReals += np.sort(np.random.choice(list(set(range(nn))-set(iReals)),nn//2//nt,replace=False)).tolist()
                input_plans = np.array(stages[-1].plans)[stages[-1].pareto]
                input_plans = remove_future_wells(input_plans,tt)

            print(' ')
            print('Building monitoring plans based on data avaiable in year %i of %i'%(tt*10,(nt*10)))
            plans  = []
            pareto = []
            iteration = 0
            while True:

                print('Running iteration %i'%iteration)
                presTemp,satuTemp,nrmsTemp,gravTemp = remove_iReal_tt(presBool,satuBool,nrmsBool,gravBool,iReals,tt)
                if iteration==0: newPlans = add_sensor_combs(input_plans,satuTemp,nrmsTemp,int(inputs['max_wells']),int(inputs['max_seismic_arrays']))
                else:            newPlans = add_sensor_combs(np.array(plans[-1])[pareto[-1]],satuTemp,nrmsTemp,int(inputs['max_wells']),int(inputs['max_seismic_arrays']))
                if len(newPlans)>0:
                    plans += [newPlans]
                    print('...built %i monitoring plans'%len(plans[-1]))
                    print('...identifying Pareto optimal plans')
                    pareto += [compute_pareto(plans[-1],presTemp,satuTemp,nrmsTemp,gravTemp)]
                    print('...identified %i Pareto-optimal plans'%len(pareto[-1]))
                    iteration += 1
                else: break

            final = []
            for iPlan in range(len(plans)): final += np.array(plans[iPlan])[pareto[iPlan]].tolist()
            #for iPlan in range(len(plans)): final += np.array(plans[iPlan]).tolist()
            par = compute_pareto(final,presTemp,satuTemp,nrmsTemp,gravTemp)
            stages += [Stage(final,par,iReals,[])]

        for stage in stages:
            det  = np.array([len(plan.detections) for plan in np.array(stage.plans)[stage.pareto]])
            ii = np.where(det==np.max(det))[0]
            selected = np.array(stage.plans)[stage.pareto][ii]
            ttfd = np.array([10.0*plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool) for plan in selected])
            ii = np.argsort(ttfd)
            selected = selected[ii]
            selected = selected[0:inputs['number_proposals']].tolist()
            stage.selected = selected


        time_dict1 = {index: value for index, value in enumerate(time_array.tolist())}
        time_dict2 = {index: value for index, value in enumerate(np.array(365.25*times,dtype='float64').tolist())}
        sources_dict = {index: value.tolist() for index, value in enumerate(sources)}
        receivers_dict = {index: value.tolist() for index, value in enumerate(receivers)}

        outputDict = {'seismic_time_days':time_dict1,
                      'seismic_sourceLocation_meters':sources_dict,
                      'seismic_receiverLocation_meters':receivers_dict,
                      'seismic_arrays':configuration.arrays,
                      'point_time_days':time_dict2,
                      'stages':[stage.to_dict() for stage in stages]}
        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results.pkl'])
        pickle.dump(outputDict, open(filename,'wb'))
        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results.json'])
        json.dump( outputDict, open(filename,'w'), indent=4 )
        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results.yaml'])
        yaml.dump( outputDict, open(filename,'w') )
        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results.h5'])
        #hf = h5py.File( filename, 'w')
        #hf.create_dataset( 'data', data=outputDict)

        outputDict = {'seismic_time_days':time_dict1,
                      'seismic_sourceLocation_meters':sources_dict,
                      'seismic_receiverLocation_meters':receivers_dict,
                      'seismic_arrays':configuration.arrays,
                      'point_time_days':time_dict2,
                      'selected':[[plan.to_dict() for plan in stage.selected] for stage in stages]}
        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results_selected.pkl'])
        pickle.dump( outputDict, open(filename,'wb'))
        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results_selected.json'])
        json.dump( outputDict, open(filename,'w'), indent=4 )
        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results_selected.yaml'])
        yaml.dump( outputDict, open(filename,'w') )


    if inputs['plot_results']:

        filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results.yaml'])
        temp = yaml.safe_load( open(filename,'r') )
        stages = [Stage(stageDict) for stageDict in temp['stages']]

        print(stages)

        az = 30
        th = 45
        th = 45+90+90+110

        xx,yy,zz=np.meshgrid(vx,vy,vz,indexing='ij')

        for iStage in range(len(stages)):

            presTemp,satuTemp,nrmsTemp,gravTemp = remove_iReal_tt(presBool,satuBool,nrmsBool,gravBool,stages[iStage].iReals,0)

            #print('Stage %i...'%iStage)
            plans = stages[iStage].plans
            pareto = stages[iStage].pareto

            det  = np.array([len(plan.detections) for plan in plans])
            #ttfd = np.array([plan.get_avg_ttfd(satuTemp,nrmsTemp) for plan in plans])
            ttfd = np.array([plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool) for plan in plans])
            dtim = np.array([plan.get_avg_commitment_time() for plan in plans])

            points = np.array(list(zip(-det,ttfd)))
            rank  = pareto_front_divide_and_conquer( points )

            ii = [int(np.where((points == item).all(axis=1))[0][0]) for item in rank]
            ii = np.array(ii)[np.argsort(det[ii])].tolist()

            '''filename = os.sep.join([inputs['directory_output_files'],'well_seismic_results.yaml'])
            temp = yaml.safe_load( open(filename,'r') )
            stages = [Stage(stageDict) for stageDict in temp['stages']]
            plt.figure(figsize=(12,8))
            det  = np.array([len(plan.detections) for plan in stages[0].plans])
            ttfd = np.array([plan.get_avg_ttfd(presBool,satuBool,nrmsBool,gravBool) for plan in stages[0].plans])
            #points = np.array(list(zip(-det,ttfd)))
            #rank  = pareto_front_divide_and_conquer( points )
            plt.scatter(det,ttfd)
            #plt.scatter(det[pareto[iteration]],ttfd[pareto[iteration]],zorder=-iteration*2,label='Iteration %i (pareto)'%iteration)
            plt.legend(bbox_to_anchor=(1.2, 0.95))
            plt.xlabel('Number of Detections',fontsize=14)
            plt.ylabel('Average Time to First Detection [years]',fontsize=14)
            plt.savefig('det_ttfd_%02i.png'%iStage,format='png',bbox_inches='tight',dpi=300)
            plt.close()'''

            print(ii)
            kk=0
            #for iPlan in range(len(plans)):
            #for iPlan in ii:
            #for iPlan in [ii[-1]]:
            for iPlan in ii[::-1]:
              ##if len(plans[iPlan].gravity_arrays)==0: continue
              #if len(plans[iPlan].seismic_arrays)==0: continue
              #if len(plans[iPlan].wells)==0: continue
              #if not 'Pressure' in [sensor.type for sensor in plans[iPlan].point_sensors]: continue
              #if not 'Saturation' in [sensor.type for sensor in plans[iPlan].point_sensors]: continue

              print([sensor.type for sensor in plans[iPlan].point_sensors])
              #exit()
              print(nrmsBool.shape)

              detectableLeaksSatu = set(np.where(np.any(satuTemp,axis=(1,2,3,4)))[0])
              print('detectableLeaksSatu',detectableLeaksSatu)
              detectableLeaksSeis = set(np.where(np.any(nrmsTemp,axis=(0,2)))[0])
              print('detectableLeaksSeis',detectableLeaksSeis)
              detectableLeaks = detectableLeaksSatu.union(detectableLeaksSeis)
              print('detectableLeaks',detectableLeaks,len(detectableLeaks))
              #exit()
              detectedLeaks = set(np.array(list( np.array(plans)[iPlan].detections )))
              print('detectedLeaks',detectedLeaks,len(detectedLeaks))
              detectableUndetected = detectableLeaks-detectedLeaks
              print('detectableUndetected',detectableUndetected)

              '''detectedNew = set()
              for sensor in plans[iPlan].point_sensors:
                  i = sensor.well.i
                  j = sensor.well.j
                  k = sensor.k
                  #print( i,j,k,sensor.install_timestep )
                  print( sensor, np.where(np.any(satuTemp[:,i,j,k,sensor.install_timestep:],axis=1))[0] )
                  detectedNew.update(set( np.where(np.any(satuTemp[:,i,j,k,sensor.install_timestep:],axis=1))[0] ))
              for array in plans[iPlan].arrays:
                  print( array, np.where(nrmsTemp[array.iArray,:,array.iTime])[0] )
                  detectedNew.update(set( np.where(nrmsTemp[array.iArray,:,array.iTime])[0] ))
              print('detectedNew',detectedNew,len(detectedNew))
              exit()'''

              iDetect = 0
              #for iReal in np.sort(list(detectableUndetected)):
              for iReal in np.random.choice(list(detectableUndetected),1):
                iDetect += 1
                for it in range(satuTemp.shape[-1]):

                    print('...plotting monitoring plan %i'%iPlan)
                    fig = plt.figure(figsize=(14,8))

                    plt.suptitle('Monitoring plan %i'%(iPlan))

                    ax1 = fig.add_subplot(121,projection='3d',proj_type='ortho')
                    ax1.set_title('Leak %i of %i detectable undetected, time=%i [years]'%(iDetect,len(detectableUndetected),times[it]),fontsize=14)

                    print([well.drill_timestep for well in plans[iPlan].wells])
                    print([sensor.install_timestep for sensor in plans[iPlan].point_sensors])
                    print([array.iTime for array in plans[iPlan].seismic_arrays])
                    print([array.it for array in plans[iPlan].gravity_arrays])

                    norm   = mpl.colors.Normalize(vmin=np.min(satu),vmax=np.max(satu))
                    sm     = mpl.cm.ScalarMappable(norm=norm,cmap=mpl.cm.jet)

                    if len(np.array(plans)[iPlan].detections)==0: continue

                    detected = np.any(satuTemp[np.array(list(np.array(plans)[iPlan].detections)),:,:,:],axis=(0,4))
                    ax1.voxels(xx,yy,zz,detected,facecolors='gray',alpha=0.05)

                    detectable = satuTemp[iReal,:,:,:,it]
                    ax1.voxels(xx,yy,zz,detectable,facecolors='blue',alpha=0.15)

                    #ax1.text(3500,1750,-800,'Sensors:',fontsize=14)
                    if 'none' not in inputs['fixed_wells']:
                        for wellStr in inputs['fixed_wells'].split(';'):
                            #print(wellStr)
                            wellStr = wellStr.split(',')
                            if wellStr[0].strip()=='ijk':
                                i  = int(wellStr[1])
                                j  = int(wellStr[2])
                                k  = int(wellStr[3])
                                ax1.plot([x[i],x[i]],[y[j],y[j]],[-100,z[k]],lw=0.75,c='red')
                            elif wellStr[0].strip()=='xyz':
                                xi = float(wellStr[1])
                                yi = float(wellStr[2])
                                zi = float(wellStr[3])
                                ax1.plot([xi,xi],[yi,yi],[-100,zi],lw=0.75,c='red')

                    for well in plans[iPlan].wells:
                        i = well.i
                        j = well.j
                        k = well.depth
                        print(i,j,k)
                        ax1.plot([x[i],x[i]],[y[j],y[j]],[-100,z[k]],lw=0.75,c='red')

                    colors = [mpl.cm.jet(value) for value in np.linspace(0, 1, len(plans[iPlan].point_sensors)+len(plans[iPlan].seismic_arrays)+len(plans[iPlan].gravity_arrays) )]

                    nSensor=0
                    for sensor in plans[iPlan].point_sensors:
                        i = sensor.well.i
                        j = sensor.well.j
                        k = sensor.k
                        ax1.scatter(x[i],y[j],z[k],s=50,c=colors[nSensor])
                        #ax1.text(3600,1750,-700+100*nSensor,'Sensor %i: %.2f [years]'%(nSensor+1,sensor.install_timestep*10.0),fontsize=12)
                        nSensor+=1
                        #print(iReal,i,j,k,sensor.install_timestep)
                        #print(np.any(satuTemp[iReal,i,j,k,sensor.install_timestep:]))

                    for array in plans[iPlan].seismic_arrays:
                        for iReceiver in configuration.arrays[array.iArray]['receivers']:
                            xyzR = receivers[iReceiver]
                            ax1.scatter( xyzR[0], xyzR[1]+2250, xyzR[2], s=50,c=colors[nSensor],marker='.')
                        xyzS = sources[configuration.arrays[array.iArray]['source']]
                        ax1.scatter( xyzS[0], xyzS[1]+2250, xyzS[2], s=80,c=colors[nSensor],marker='^')
                        nSensor+=1

                    for array in plans[iPlan].gravity_arrays:
                        print(array.i0,array.i1,array.mx,array.my)
                        for i in range(array.i0,array.i1+1,array.mx):
                            for j in range(array.j0,array.j1+1,array.my):
                                ax1.scatter(x[i],y[j],0, s=50,c=colors[nSensor],marker='s')
                        nSensor+=1
                        #xyzS = sources[configuration.arrays[array.iArray]['source']]
                        #ax1.scatter( xyzS[0], xyzS[1]+2250, xyzS[2], s=80,c=colors[nSensor],marker='^')
                        #nSensor+=1

                    ax1.set_xlim([xmin,xmax])
                    ax1.set_ylim([ymin,ymax])
                    ax1.set_zlim([zmin,zmax])
                    ax1.set_xlabel('Easting [m]',fontsize=14)
                    ax1.set_ylabel('Northing [m]',fontsize=14)
                    ax1.set_zlabel('Depth [m]',fontsize=14)
                    ax1.invert_zaxis()
                    ax1.view_init(az,th)

                    proxy1 = mpl.patches.Rectangle((0, 0), 1, 1, fc="gray",alpha=0.1)
                    proxy2 = mpl.patches.Rectangle((0, 0), 1, 1, fc="blue",alpha=0.1)

                    ax1.legend([proxy1,proxy2], ['Detected Leakage','Undetected Leakage'])

                    ax2 = fig.add_subplot(122)
                    ax2.scatter(det,ttfd,s=20,c='darkgray',label='Pareto-Optimal Plans')
                    ax2.scatter(det[iPlan],ttfd[iPlan],s=20,c='black',label='Selected Plan')
                    ax2.set_xlabel('Number of Leaks Detected',fontsize=14)
                    ax2.set_ylabel('Average Time to First Detection [years]',fontsize=14)
                    ax2.legend()


                    line = mpl.lines.Line2D([0.20,0.45], [0.08,0.08], transform=fig.transFigure, color='black', linewidth=1.5, zorder=0)
                    fig.lines.append(line)
                    fig.text( 0.20+0.25*0.5, 0.08, 'Timeline [years]', ha='center', va='bottom', fontsize=12, color='black')

                    nSensor = 0

                    for array in plans[iPlan].seismic_arrays:
                        arrayTime = mpl.patches.Ellipse(( 0.20+0.25*time_array[array.iTime]/np.max(time_array), 0.08), 0.005,0.01, transform=fig.transFigure, color=colors[nSensor], zorder=2)
                        fig.add_artist(arrayTime)
                        fig.text( 0.20+0.25*time_array[array.iTime]/np.max(time_array), 0.06, '%i'%(time_array[array.iTime]/365.25), ha='center', va='top', fontsize=12, color='black')
                        nSensor+=1

                    for array in plans[iPlan].gravity_arrays:
                        arrayTime = mpl.patches.Ellipse(( 0.20+0.25*times[array.it]/np.max(times), 0.08), 0.005,0.01, transform=fig.transFigure, color=colors[nSensor], zorder=2)
                        fig.add_artist(arrayTime)
                        fig.text( 0.20+0.25*times[array.it]/np.max(times), 0.06, '%i'%(times[array.it]), ha='center', va='top', fontsize=12, color='black')
                        nSensor+=1

                    for sensor in plans[iPlan].point_sensors:
                        arrayTime = mpl.patches.Ellipse(( 0.20+0.25*times[sensor.install_timestep]/np.max(times), 0.08), 0.005,0.01, transform=fig.transFigure, color=colors[nSensor], zorder=2)
                        fig.add_artist(arrayTime)
                        fig.text( 0.20+0.25*times[sensor.install_timestep]/np.max(times), 0.06, '%i'%(times[sensor.install_timestep]), ha='center', va='top', fontsize=12, color='black')
                        nSensor+=1

                    plt.savefig('monitoringPlan_%02i_%05i_%05i_%05i.png'%(iStage,kk,iReal,it),format='png')
                    #for thi in range(720):
                    #    ax1.view_init(az,th)
                    #    th += 0.5
                    #    plt.savefig('monitoringPlan_%02i_%05i_%05i_%05i_%05i.png'%(iStage,kk,iReal,it,thi),format='png')
                    plt.close()
                    #exit()
                    #th += 0.75
              kk += 1
              #break
