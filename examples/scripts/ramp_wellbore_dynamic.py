import itertools
import pickle
import yaml
import json
import zipfile
import h5py
import copy
from tqdm import tqdm
import requests
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


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
    except:
        raise ValueError()

#stages = np.array(inputs['stages'].split(','), dtype='int').tolist()

download_directory = os.sep.join([inputs['directory_simulation_data']])
if not os.path.exists(download_directory):
    os.mkdir(download_directory)

if isinstance(inputs['scenarios'], int):
    scenario_indices = list(range(inputs['scenarios']))
elif isinstance(inputs['scenarios'], list):
    scenario_indices = inputs['scenarios']
elif isinstance(inputs['scenarios'], str):
    if '-' in inputs['scenarios']:
        scenario_indices = list(range(int(inputs['scenarios'].split('-')[0]),
                                      int(inputs['scenarios'].split('-')[1])))
    else:
        raise Exception('Error: scenarios list is not formatted correctly')
else:
    raise Exception('Error: scenarios list is not formatted correctly')

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

def add_sensor_combs(input_plans,satuBool,maxWells):
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

        # identify the list of i,j locations of each sensor,
        # then remove any duplicates in case two sensors are already
        # at the same point at different depths
        existing_wells = [[deployment.wellboreSensor.i,deployment.wellboreSensor.j] \
                          for deployment in plan.deployments]
        existing_wells = np.array(existing_wells)
        existing_wells = np.unique(existing_wells, axis=0)
        existing_wells = existing_wells.tolist()

        # if we're at the maximum numbers of sensors, skip adding any more
        if len(plan.deployments) >= int(inputs['max_sensors']):
            output_plans.add(copy.deepcopy(plan))
            continue

        for i, j, k in itertools.product(range(nx), range(ny), range(nz)):

            # if we already have a sensor at this i,j,k location, we don't place another one
            existing_sensors = [[deployment.wellboreSensor.i,
                                 deployment.wellboreSensor.j,
                                 deployment.wellboreSensor.k] for deployment in plan.deployments]
            if [i ,j, k] in existing_sensors:
                continue

            # if we're at the maximum numbers of wells, ie unique i,j combinations,
            # and this i,j combination isn't one of them, we don't put any sensors here
            if len(existing_wells) >= int(inputs['max_wells']):
                if [i, j] not in existing_wells:
                    continue

            # if this i,j,k never exceeds the threshold for any leak scenario,
            # don't put any sensors there
            if not np.any(satuBool[:, i, j, k, :]):
                continue

            for it in range(nt):
                if np.any(satuBool[:, i, j, k, it:]):
                    planNew = copy.deepcopy(plan)
                    sensor = WellboreSensor(i, j, k, 'Saturation')
                    deployment = Deployment(timestep=it, wellboreSensor=sensor)
                    detections = np.where(np.any(satuBool[:, i, j, k, it:], axis=1))[0].tolist()
                    planNew.deployments.add(deployment)
                    planNew.detections.update(detections)

                    # if the new monitoring plan detects some leak scenarios
                    # that the old one didn't, its a keeper
                    # if it detects the same scenarios but detects some of them
                    # sooner, it's a keeper
                    # if it detects fewer leaks, something is wrong
                    keep = False
                    if len(planNew.detections.difference(plan.detections)) > 0:
                        keep = True
                    elif planNew.avg_ttfd(satuBool) < plan.avg_ttfd(satuBool):
                        keep = True
                    elif len(plan.detections.difference(planNew.detections)) > 0:
                        raise Exception("".join([
                            "Error: the improved plan somehow detects fewer ",
                            "leaks than the plan it's based on"]))

                    if keep:
                        output_plans.add(planNew)

    return list(output_plans)

def remove_iReal_tt(satuIn, iReals, tt):
    jReals = list(set(range(satuIn.shape[0]))-set(iReals))
    satuOut = np.copy(satuIn)
    satuOut[jReals, :, :, :, :] = False
    if tt > 0:
        satuOut[:, :, :, :, :tt] = False
    return satuOut

def compute_pareto(plans, satuBool):
    det  = np.array([len(plan.detections) for plan in plans])
    ttfd = np.array([plan.avg_ttfd(satuBool) for plan in plans])
    dtim = np.array([plan.drill_time() for plan in plans])
    points = np.array(list(zip(-det, ttfd, -dtim)))
    rank  = pareto_front_divide_and_conquer(points)
    return [int(np.where((points == item).all(axis=1))[0][0]) for item in rank]

def scatter2step(x, y):
    x_step = []
    y_step = []
    for (xval, yval) in zip(x, y):
        x_step += 2*[xval] # take 2 copies of x_val and append to x_step
        y_step += 2*[yval] # take 2 copies of y_val and append to y_step
    return x_step[1:], y_step[:-1]


class Stage:
    def __init__(self, plans=[], pareto=[], iReals=[], selected=[]):
        self.plans  = plans
        self.pareto = pareto
        self.iReals = iReals
        self.selected = selected

    def to_dict(self):
        return {'plans': [plan.to_dict() for plan in self.plans],
                'pareto': self.pareto,
                'iReals': self.iReals,
                'selected': [plan.to_dict() for plan in self.selected]}


class MonitoringPlan:
    def __init__(self, deployments=[], detections=[]):
        self.deployments = set(deployments)
        self.detections = set(detections)

    def __eq__(self,other):
        return self.deployments==other.deployments

    def __hash__(self):
        return hash(frozenset(self.deployments))

    def to_dict(self):
        return {'deployments': [deployment.to_dict() for deployment in self.deployments],
                'detections': list(self.detections)}

    def avg_ttfd(self, satuBool):
        ttfd = []
        if len(self.detections) == 0:
            return np.inf

        for detection in self.detections:
            ttd = []
            for deployment in self.deployments:
                ii = deployment.wellboreSensor.i
                jj = deployment.wellboreSensor.j
                kk = deployment.wellboreSensor.k
                tt = deployment.timestep
                tds = np.where(satuBool[detection, ii, jj, kk, :])[0]
                tds = tds[tds>=tt]
                if len(tds) > 0:
                    ttd += [np.min(tds)]
            if len(ttd) > 0:
                ttfd += [np.min(ttd)]
        return np.mean(ttfd)

    def drill_time(self):
        wells = [[deployment.wellboreSensor.i,
                  deployment.wellboreSensor.j,
                  deployment.timestep] for deployment in self.deployments]
        wells = np.array(wells)
        unique_wells = np.unique(wells[:, 0:2], axis=0)
        drill_times = []
        for well in unique_wells:
            ii = np.where(np.all(well==wells[:, 0:2], axis=1))[0]
            drill_times += [np.min(wells[ii, 2])]
        return np.mean(drill_times)

    #def avg_plume_identification_potential(self, satuBool):
    #    pips = []
    #    for detection in self.detections:
    #        #
    #    return np.mean(pips)

    def avg_plume_delineation_potential(self, satuBool):
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
                satuBool[detection, :, :, :, tt]

                # of course it's always better to detect the plume in multiple
                # locations than not compute how much the plume and
                # line/polygon/hull overlap for 4+, count detectable nodes
                # within convex hull, divide by number of nodes detectable

                # there's also some value in having some sensors that are
                # just outside the plume, that gives us useful delineation
                # information as well

                pdps+=[]

        return np.mean(pdps)


class Deployment:
    def __init__(self, timestep=None, array=None, wellboreSensor=None):
        self.timestep = timestep
        self.array = array
        self.wellboreSensor = wellboreSensor

    def __eq__(self, other):
        return self.array==other.array and self.timestep==other.timestep \
            and self.wellboreSensor==other.wellboreSensor

    def __hash__(self):
        return hash((self.array, self.timestep, self.wellboreSensor))

    def to_dict(self):
        return {'timestep': self.timestep,
                'array': self.array,
                'wellboreSensor': self.wellboreSensor.to_dict()}


class WellboreSensor:
    def __init__(self, i, j, k, stype):
        self.i = i
        self.j = j
        self.k = k
        self.stype = stype

    def __eq__(self, other):
        return self.i==other.i and self.j==other.j and self.k==other.k \
            and self.stype==other.stype

    def __hash__(self):
        return hash((self.i, self.j, self.k, self.stype))

    def to_dict(self):
        return {'i': self.i, 'j': self.j, 'k': self.k, 'stype': self.stype}


def dict2obj(dict):
    stage = Stage()
    stage.pareto = dict['pareto']
    for planDict in dict['plans']:
        plan = MonitoringPlan()
        for deployment in planDict['deployments']:
            i = deployment['wellboreSensor']['i']
            j = deployment['wellboreSensor']['j']
            k = deployment['wellboreSensor']['k']
            stype = deployment['wellboreSensor']['type']
            tt = deployment['timestep']
            wellboreSensor = WellboreSensor(i, j, k, stype)
            plan.deployments.add(
                Deployment(timestep=tt, wellboreSensor=wellboreSensor))
        plan.detections.update(planDict['detections'])
        stage.plans += [plan]

    for planDict in dict['selected']:
        plan = MonitoringPlan()
        for deployment in planDict['deployments']:
            i = deployment['wellboreSensor']['i']
            j = deployment['wellboreSensor']['j']
            k = deployment['wellboreSensor']['k']
            stype = deployment['wellboreSensor']['stype']
            tt = deployment['timestep']
            wellboreSensor = WellboreSensor(i, j, k, stype)
            plan.deployments.add(
                Deployment(timestep=tt, wellboreSensor=wellboreSensor))
        plan.detections.update(planDict['detections'])
        stage.selected += [plan]
    return stage


def remove_future_wells(oldPlans, tt):
    newPlans = []
    for oldPlan in oldPlans:
        newPlan = MonitoringPlan()
        for deployment in oldPlan.deployments:
            if deployment.timestep <= tt:
                newPlan.deployments.add(deployment)
        if len(newPlan.deployments) > 0:
            newPlans += [newPlan]
    if len(newPlans) == 0:
        newPlans = [MonitoringPlan()]
    return newPlans


if inputs['download_data']:
    print('Downloading data...')
    print(20*'-')
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
        if resource['name'] == fname:
            r = requests.get(resource['url'], headers=headers, stream=True)
            total_size_in_bytes = int(r.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(os.sep.join([download_directory, fname]), 'wb') as file:
                for data in r.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes not in [0, progress_bar.n]:
                print("ERROR: something went wrong during data files download")

    print('Unzipping downloaded data file...')
    print(20*'-')
    with zipfile.ZipFile(os.sep.join([download_directory, fname]), 'r') as zip_ref:
        folder_to_extract = download_directory
        try:
            zip_ref.extractall(folder_to_extract)
        except FileExistsError:
            pass

if inputs['run_optimization'] or inputs['plot_results']:
    print('Setting up optimization...')
    print(20*'-')
    # load one HDF5 file into memory to get the shape of the dataset, min/max values etc
    file = h5py.File(download_directory+'/sim%04i.h5'%1, 'r')
    nx, ny, nz = np.array(file['plot0']['pressure']).shape
    nt = 12
    vx = np.array(file['data']['vertex-x'])
    vy = np.array(file['data']['vertex-y'])
    vz = np.array(file['data']['vertex-z'])
    x = np.array(file['data']['x'])
    y = np.array(file['data']['y'])
    z = np.array(file['data']['z'])
    times = np.array(file['data']['times'])

    xmin = np.min(vx)
    xmax = np.max(vx)
    ymin = np.min(vy)
    ymax = np.max(vy)
    zmin = np.min(vz)
    zmax = np.max(vz)

    # use that info to construct the numpy arrays the leak scenarios will be loaded into
    # then read in all HDF5 files
    nn = len(scenario_indices)
    #pres = np.zeros([nn, nx, ny, nz, nt])
    satu = np.zeros([nn, nx, ny, nz, nt])
    #grav = np.zeros([nn, nx, ny, nt])
    for n in range(nn):
        print('Loading Scenario %i'%scenario_indices[n])
        file = h5py.File(download_directory+'/sim%04i.h5'%scenario_indices[n], 'r')
        for it in range(12):
            # pres[n, :, :, :, it] = np.array(file['plot%i'%it]['pressure'])
            satu[n, :, :, :, it] = np.array(file['plot%i'%it]['saturation'])*100
            # grav[n, :, :, it]   = np.abs(np.array(file['plot%i'%it]['gravity'])[1:, 1:]).T

    satuBool = satu>inputs['threshold_co2']

    # remove any simulations that never exceed the detection threshold
    ii = np.where(np.any(satuBool, axis=(1, 2, 3, 4)))[0]
    satuBool = satuBool[ii, :, :, :, :]
    nn = satuBool.shape[0]

if inputs['run_optimization']:
    print('\nStarting optimization process...')
    print(20*'-')
    stages = []
    for tt in range(nt):

        if tt==0:
            iReals = np.sort(np.random.choice(list(range(nn)), nn//2, replace=False)).tolist()
            input_plans = [MonitoringPlan()]
        else:
            iReals += np.sort(np.random.choice(list(set(range(nn))-set(iReals)),
                                               nn//2//nt, replace=False)).tolist()
            input_plans = np.array(stages[-1].plans)[stages[-1].pareto]
            input_plans = remove_future_wells(input_plans, tt)

        print('Building monitoring plans for years %i-%i of %i'%(tt*10, (tt+1)*10, (nt*10)))
        plans  = []
        pareto = []
        for iSensor in range(int(inputs['max_sensors'])):

            print('Considering all viable %i-sensor monitoring plans'%(iSensor+1))
            if iSensor==0:
                plans += [add_sensor_combs(input_plans,
                                           remove_iReal_tt(satuBool, iReals, tt),
                                           inputs['max_wells'])]
            else:
                plans += [add_sensor_combs(np.array(plans[-1])[pareto[-1]],
                                           remove_iReal_tt(satuBool, iReals, tt),
                                           inputs['max_wells'])]
            print('...built %i monitoring plans'%len(plans[-1]))
            print('...identifying Pareto optimal plans')
            pareto += [compute_pareto(plans[-1], remove_iReal_tt(satuBool, iReals, tt))]
            print('...identified %i Pareto-optimal plans'%len(pareto[-1]))

        final = []
        for iPlan in range(len(plans)):
            final += np.array(plans[iPlan])[pareto[iPlan]].tolist()
        par = compute_pareto(final, remove_iReal_tt(satuBool, iReals, tt))
        stages += [Stage(plans=final, pareto=par, iReals=iReals)]

    for stage in stages:
        det  = np.array([len(plan.detections) for plan in np.array(stage.plans)[stage.pareto]])
        ii = np.where(det==np.max(det))[0]
        selected = np.array(stage.plans)[stage.pareto][ii]
        ttfd = np.array([10.0*plan.avg_ttfd(satuBool) for plan in selected])
        ii = np.argsort(ttfd)
        selected = selected[ii]
        selected = selected[0:inputs['number_proposals']].tolist()
        stage.selected = selected

    filename = os.sep.join([inputs['directory_output_files'],
                            'wellbore_results.pkl'])
    with open(filename, 'wb') as next_file:
        pickle.dump({'stages': stages}, next_file)

    filename = os.sep.join([inputs['directory_output_files'],
                            'wellbore_results.json'])
    with open(filename, 'w') as next_file:
        json.dump({'stages': [stage.to_dict() for stage in stages]}, next_file)

    filename = os.sep.join([inputs['directory_output_files'],
                            'wellbore_results.yaml'])
    with open(filename, 'w') as next_file:
        yaml.dump({'stages': [stage.to_dict() for stage in stages]}, next_file)

    filename = os.sep.join([inputs['directory_output_files'],
                            'wellbore_results_selected.pkl'])
    with open(filename, 'wb') as next_file:
        pickle.dump({'selected': [stage.selected for stage in stages]}, next_file)

    filename = os.sep.join([inputs['directory_output_files'],
                            'wellbore_results_selected.json'])
    with open(filename, 'w') as next_file:
        json.dump(
            {'selected': [[plan.to_dict() for plan in stage.selected] for stage in stages]},
            next_file)

    filename = os.sep.join([inputs['directory_output_files'],
                            'wellbore_results_selected.yaml'])
    with open(filename, 'w') as next_file:
        yaml.dump(
            {'selected': [[plan.to_dict() for plan in stage.selected] for stage in stages]},
            next_file)

if inputs['plot_results']:
    print('Creating output figures...')
    print(20*'-')
    filename = os.sep.join([inputs['directory_output_files'], 'wellbore_results.pkl'])
    with open(filename, 'rb') as next_file:
        stages = pickle.load(next_file)['stages']

    #filename = os.sep.join([inputs['directory_output_files'], 'wellbore_results.json'])
    #temp = json.load(open(filename, 'r'))
    #stages = [dict2obj(stage) for stage in temp['stages']]

    #filename = os.sep.join([inputs['directory_output_files'], 'wellbore_results.yaml'])
    #temp = yaml.safe_load(open(filename, 'r'))
    #stages = [dict2obj(stage) for stage in temp['stages']]

    for iStage in range(len(stages)):
        print(iStage, len(stages[iStage].plans), len(stages[iStage].pareto))
        #for iPlan in range(len(stages[iStage].plans)):
        #    print(iStage, iPlan, len(stages[iStage].plans[iPlan].deployments),
        #          len(stages[iStage].plans[iPlan].detections))

    for iStage in range(len(stages)):
        plt.figure(figsize=(10, 8))
        for iPlan in range(len(stages[iStage].plans)):
            timeline = []
            for deployment in stages[iStage].plans[iPlan].deployments:
                timeline += [deployment.timestep*10.0]
            if iPlan in stages[iStage].pareto:
                plt.scatter(timeline, iPlan*np.ones(len(timeline)), s=30, c='k', zorder=1)
                plt.plot(timeline, iPlan*np.ones(len(timeline)), lw=0.75, c='k', zorder=1)
            #else:
            #    plt.scatter(timeline, iPlan*np.ones(len(timeline)), s=30, c='gray', zorder=0)
            #    plt.plot(timeline, iPlan*np.ones(len(timeline)), lw=0.75, c='gray', zorder=0)
        plt.savefig('timeline_%02i.png'%iStage, format='png')
        plt.close()

    for iStage in range(len(stages)):

        plt.figure(figsize=(10, 8))

        for plan in np.array(stages[iStage].plans)[stages[iStage].pareto]:
            # print([[deployment.wellboreSensor.i,
            #         deployment.wellboreSensor.j,
            #         deployment.wellboreSensor.k] for deployment in plan.deployments])
            #print('iStage',iStage)
            #print('plan',plan)
            #print('plan.detections',plan.detections)
            t0 = []
            d0 = []
            for ti in range(nt):
                t0 += [times[ti]]
                detections = set()
                for deployment in plan.deployments:
                    satuThis = remove_iReal_tt(satuBool, stages[iStage].iReals, 0)
                    i = deployment.wellboreSensor.i
                    j = deployment.wellboreSensor.j
                    k = deployment.wellboreSensor.k
                    #print('i, j, k', i, j, k)
                    #print(satuThis[:, i, j, k, :ti])
                    #print(np.any(satuThis[:, i, j, k, :ti], axis=(1)))
                    #print(np.where(np.any(satuThis[:, i, j, k, :ti], axis=(1)))[0])
                    detections.update(np.where(np.any(satuThis[:, i, j, k, :ti], axis=(1)))[0])
                d0 += [len(detections)]
            t0,d0 = scatter2step(t0, d0)
            if plan in stages[iStage].selected:
                print(len(plan.detections), plan.avg_ttfd(satuBool), plan.drill_time())
                plt.plot(t0, d0, c='blue', lw=2.5, zorder=1)
            else:
                plt.plot(t0, d0, lw=0.75, zorder=0)

        satuThis = remove_iReal_tt(satuBool, stages[iStage].iReals, 0)
        tn = []
        dn = []
        for ti in range(nt):
            tn += [times[ti]]
            dn += [np.sum(np.any(satuThis[:, :, :, :, :ti], axis=(1, 2, 3, 4)))]
        tn, dn = scatter2step(tn, dn)

        plt.plot(tn, dn, label='Detectable', c='k', lw=1.5, zorder=2)
        plt.xlabel('Time [years]', fontsize=14)
        plt.ylabel('Number of Leaks', fontsize=14)
        plt.legend()

        plt.savefig('detection_breakthrough_%02i.png'%iStage, format='png')
        plt.close()

    xx, yy, zz=np.meshgrid(vx, vy, vz, indexing='ij')

    for iStage in range(len(stages)):

        print('Stage %i...'%iStage)
        plans = stages[iStage].plans
        pareto = stages[iStage].pareto

        det  = np.array([len(plan.detections) for plan in plans])
        ttfd = np.array([10.0*plan.avg_ttfd(satuBool) for plan in plans])
        dtim = np.array([10.0*plan.drill_time() for plan in plans])

        for iPlan in pareto:
            print('...plotting monitoring plan %i'%iPlan)
            fig = plt.figure(figsize=(22, 8))
            ax = fig.add_subplot(121, projection='3d', proj_type='ortho')

            norm = mpl.colors.Normalize(vmin=np.min(satu), vmax=np.max(satu))
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.jet)

            if len(np.array(plans)[iPlan].detections)==0: continue

            detected = np.any(satuBool[np.array(list(np.array(plans)[iPlan].detections)), :, :, :],
                              axis=(0, 4))
            ax.voxels(xx, yy, zz, detected, facecolors='gray', alpha=0.1)

            detectable = np.any(satuBool, axis=(0, 4))^detected
            ax.voxels(xx, yy, zz, detectable, facecolors='blue', alpha=0.1)

            ax.text(3500, 1750, -800, 'Deployments:', fontsize=14)
            nSensor = 0
            for deployment in plans[iPlan].deployments:
                i = deployment.wellboreSensor.i
                j = deployment.wellboreSensor.j
                k = deployment.wellboreSensor.k
                ax.plot([x[i], x[i]], [y[j], y[j]], [-100, z[k]], lw=0.75, c='red')
                ax.scatter(x[i], y[j], z[k], s=50, c='red')
                ax.text(3600, 1750, -700+100*nSensor,
                        'Sensor %i: %.2f [years]'%(nSensor+1, deployment.timestep*10.0),
                        fontsize=12)
                nSensor += 1

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
            ax.set_xlabel('Easting [m]', fontsize=14)
            ax.set_ylabel('Northing [m]', fontsize=14)
            ax.set_zlabel('Depth', fontsize=14)
            ax.invert_zaxis()

            proxy1 = mpl.patches.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.1)
            proxy2 = mpl.patches.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.1)

            ax.legend([proxy1,proxy2], ['Detected Leakage','Undetected Leakage'])

            ax = fig.add_subplot(122)
            ax.scatter(det, ttfd, s=20, c='lightgray')
            ax.scatter(det[iPlan], ttfd[iPlan], s=20, c='black')
            ax.set_xlabel('Number of Leaks Detected', fontsize=14)
            ax.set_ylabel('Average Time to First Detection', fontsize=14)

            plt.savefig('monitoringPlan_%02i_%05i.png'%(iStage, iPlan),
                        format='png', bbox_inches='tight')
            plt.close()

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111,projection='3d', proj_type='ortho')

        ax.scatter(det, ttfd, dtim, s=20, ec='darkgray', fc='lightgray',
                   zorder=0, alpha=0.50)
        ax.scatter(det[pareto], ttfd[pareto], dtim[pareto], s=20, c='black',
                   zorder=1, alpha=1.0)

        ax.set_xlabel('Number Leaks Detected', fontsize=14)
        ax.set_ylabel('Time to First Detection [years]', fontsize=14)
        ax.set_zlabel('Average Drilling Time [years]', fontsize=14, color='white')

        zticks = ax.get_zticks()
        ax.set_zticks([])

        az = 90-0.0001
        th = -90.0001
        for i in range(120):
            print(i)

            if i == 25:
                ax.set_zlabel('Average Drilling Time [years]', fontsize=14, color='black')
                ax.set_zticks(zticks)

            if i < 60:
                ax.view_init(az, th)
                filename = os.sep.join([
                    inputs['directory_plots'],
                    'wellbore_results_prelim_%02i_%03ia.png'%(iStage, i)])
                plt.savefig(filename, format='png', bbox_inches='tight')
                az -= 1
                th -= 0.15
            else:
                ax.view_init(az, th)
                filename = os.sep.join([
                    inputs['directory_plots'],
                    'wellbore_results_prelim_%02i_%03ib.png'%(iStage, i-60)])
                plt.savefig(filename, format='png', bbox_inches='tight')
                th += 0.75
        plt.close()

        plt.figure(figsize=(20, 8))

        det  = np.array([len(plan.detections) for plan in plans])
        ttfd = np.array([10.0*plan.avg_ttfd(satuBool) for plan in plans])
        dtim = np.array([10.0*plan.drill_time() for plan in plans])

        ii = np.argsort(dtim)[::-1]
        sc = plt.scatter(det[ii], ttfd[ii], s=20, c=dtim[ii], zorder=0)
        cb = plt.colorbar(sc)
        cb.set_label('Average well-drilling time [years]', fontsize=14)

        plt.scatter(det[pareto], ttfd[pareto], s=50, fc='none', ec='red', zorder=1)

        plt.xlabel('Average time to first detection [years]', fontsize=14)
        plt.ylabel('Number of leaks detected', fontsize=14)

        filename = os.sep.join([inputs['directory_plots'],
                                'wellbore_results_prelim.png'])
        plt.savefig(filename, format='png', bbox_inches='tight')
        plt.close()

        filename = os.sep.join([inputs['directory_output_files'],
                                'wellbore_results_selected.yaml'])
        with open(filename, 'r') as next_file:
            temp = yaml.safe_load(next_file)
        selected = []
        for planDict in temp['selected'][iStage]:
            plan = MonitoringPlan()
            for deployment in planDict['deployments']:
                i = deployment['wellboreSensor']['i']
                j = deployment['wellboreSensor']['j']
                k = deployment['wellboreSensor']['k']
                stype = deployment['wellboreSensor']['stype']
                tt = deployment['timestep']
                wellboreSensor = WellboreSensor(i, j, k, stype)
                plan.deployments.add(
                    Deployment(timestep=tt, wellboreSensor=wellboreSensor))
            plan.detections.update(planDict['detections'])
            selected += [plan]

        det  = np.array([len(plan.detections) for plan in selected])
        ttfd = np.array([10.0*plan.avg_ttfd(satuBool) for plan in selected])
        dtim = np.array([10.0*plan.drill_time() for plan in selected])

        plt.figure(figsize=(20, 8))

        ii = np.argsort(dtim)[::-1]
        sc = plt.scatter(det[ii], ttfd[ii], s=20, c=dtim[ii], zorder=0)
        cb = plt.colorbar(sc)
        cb.set_label('Average well-drilling time [years]', fontsize=14)

        plt.xlabel('Average time to first detection [years]', fontsize=14)
        plt.ylabel('Number of leaks detected', fontsize=14)

        filename = os.sep.join([
            inputs['directory_plots'], 'wellbore_results_prelim_selected_%02i.png'%iStage])
        plt.savefig(filename, format='png', bbox_inches='tight')
        plt.close()

        xx,yy,zz=np.meshgrid(vx, vy, vz, indexing='ij')

        for iPlan in range(len(selected)):
            print('Plotting monitoring plan %i'%iPlan)
            fig = plt.figure(figsize=(22, 8))
            ax = fig.add_subplot(121, projection='3d', proj_type='ortho')

            norm   = mpl.colors.Normalize(vmin=np.min(satu),vmax=np.max(satu))
            sm     = mpl.cm.ScalarMappable(norm=norm,cmap=mpl.cm.jet)

            if len(np.array(plans)[iPlan].detections)==0:
                continue

            detected = np.any(
                satuBool[np.array(list(np.array(selected)[iPlan].detections)), :, :, :],
                axis=(0, 4))
            ax.voxels(xx, yy, zz, detected, facecolors='gray', alpha=0.1)

            detectable = np.any(satuBool, axis=(0, 4))^detected
            ax.voxels(xx, yy, zz, detectable, facecolors='blue', alpha=0.1)

            ax.text(3500, 1750, -800, 'Deployments:', fontsize=14)
            nSensor = 0
            for deployment in selected[iPlan].deployments:
                i = deployment.wellboreSensor.i
                j = deployment.wellboreSensor.j
                k = deployment.wellboreSensor.k
                ax.plot([x[i], x[i]], [y[j], y[j]], [-100, z[k]], lw=0.75, c='red')
                ax.scatter(x[i], y[j], z[k], s=50, c='red')
                ax.text(3600, 1750, -700+100*nSensor,
                        'Sensor %i: %.2f [years]'%(nSensor+1, deployment.timestep*10.0),
                        fontsize=12)
                nSensor += 1

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_zlim([zmin, zmax])
            ax.set_xlabel('Easting [m]', fontsize=14)
            ax.set_ylabel('Northing [m]', fontsize=14)
            ax.set_zlabel('Depth', fontsize=14)
            ax.invert_zaxis()

            proxy1 = mpl.patches.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.1)
            proxy2 = mpl.patches.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.1)

            ax.legend([proxy1,proxy2], ['Detected Leakage','Undetected Leakage'])

            ax = fig.add_subplot(122)
            ax.scatter(det, ttfd, s=20, c='lightgray')
            ax.scatter(det[iPlan], ttfd[iPlan], s=20, c='black')
            ax.set_xlabel('Number of Leaks Detected', fontsize=14)
            ax.set_ylabel('Average Time to First Detection', fontsize=14)

            plt.savefig('monitoringPlan_selected_%02i_%05i.png'%(iStage, iPlan),
                        format='png', bbox_inches='tight')
            plt.close()

if inputs['download_data']:
    print('Check downloaded data files in the folder:',
          os.path.abspath(os.path.join(os.getcwd(), inputs['directory_simulation_data'])))

if inputs['run_optimization']:
    print('Check produced output files in the folder:',
          os.path.abspath(os.path.join(os.getcwd(), inputs['directory_output_files'])))

if inputs['plot_results']:
    print('Check produced output figures in the folder:',
          os.path.abspath(os.path.join(os.getcwd(), inputs['directory_plots'])))
