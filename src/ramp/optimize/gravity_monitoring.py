# -*- coding: utf-8 -*-
# Gravity monitoring code
# Yuan Tian, LLNL tian7@llnl.gov
# Xianjin Yang, LLNL  yang25@llnl.gov

import numpy as np
import h5py, sys, os, glob,shutil
import multiprocessing
import matplotlib.pyplot as plt
from natsort import natsorted # pip install natsort
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_readers.read_input_from_files import get_all_h5_filenames, read_yaml, download_data_from_edx
from utilities.read_write_hdf5 import *


input_yaml_path = './control_file_interface.yaml'
yaml_paras = read_yaml(input_yaml_path)
params = yaml_paras['GravityMonitoringOptimization']

# Extracting and setting up parameters
incomplete_simulations = params['incomplete_simulations']
nSimulations = params['nSimulations']
years = params['years']
nt = len(years)

rootdir = params['rootdir']
workspace_id = params['workspace_id']
data_folder_id = params['data_folder_id']
api_key = params['api_key']
if not glob.glob(rootdir + 'sim*'):
    download_data_from_edx(workspace_id, data_folder_id, api_key, rootdir)
    print('downloaded data from edx')
    for file in glob.glob(rootdir + 'sim*'):
        print(file)
        os.makedirs(file[:-4])
        shutil.unpack_archive(file, file[:-4])
        os.remove(file)

outdir = params['outdir']
ths = params['ths']
all_gra_sims_fn = get_all_h5_filenames(rootdir)
# Grid dimensions
nx, ny, nz = params['nx'], params['ny'], params['nz']
no_co2_P_tds = np.zeros((nx, ny, nz))

# Subregion boundary
xmin, xmax = params['xmin'], params['xmax']
ymin, ymax = params['ymin'], params['ymax']
zmin, zmax = params['zmin'], params['zmax']

# Vertex and voxel center
vx = np.linspace(xmin, xmax, nx + 1)
vy = np.linspace(ymin, ymax, ny + 1)
vz = np.linspace(zmin, zmax, nz + 1)
x = (vx[:-1] + vx[1:]) / 2
y = (vy[:-1] + vy[1:]) / 2
z = (vz[:-1] + vz[1:]) / 2

# Gravity measurement stations on the ground surface
ngx, ngy = params['ngx'], params['ngy']
no_grav = np.zeros((ngy, ngx))
gx = np.linspace(xmin, xmax, ngx)
gy = np.linspace(ymin, ymax, ngy)

gra_data_all_t_all_sims = []
for gra_file_name in all_gra_sims_fn:
    if gra_file_name[-7:-3] in incomplete_simulations:
        continue
    #print(gra_file_name)
    dum_hdf = h5py.File(gra_file_name, 'r')
    groups = get_groups('/', dum_hdf)
    print(groups)
    groups = natsorted(groups)
    gra_data_all_t_step = []
    for j, group in enumerate(groups):
        datasets = get_datasets(group+'/', dum_hdf)
        if j > 1:
            gra_data_all_t_step.append(dum_hdf[datasets[1]])
    gra_data_all_t_all_sims.append(gra_data_all_t_step)
gra_data_all_t_all_sims = np.abs(np.array(gra_data_all_t_all_sims))


datasets = get_datasets(groups[0]+'/', dum_hdf)

ert_abmn = np.array(dum_hdf[datasets[0]])
grav_x = np.array(dum_hdf[datasets[1]])
grav_y = np.array(dum_hdf[datasets[2]])
porosity = np.array(dum_hdf[datasets[3]])
steps = np.array(dum_hdf[datasets[4]])
times = np.array(dum_hdf[datasets[5]])
x = np.array(dum_hdf[datasets[10]])
y = np.array(dum_hdf[datasets[12]])
z = np.array(dum_hdf[datasets[14]])

if not os.path.exists(outdir):
    os.makedirs(outdir)
plt.figure(figsize=(32, 12))
k = 0
# Initialize a dictionary to store the data for YAML export
yaml_data = {}
for j, th_gra in enumerate(ths):
    for i, time in enumerate(times):
        k = k+1
        plt.subplot(len(ths), 12, k)
        gra_all_sims = gra_data_all_t_all_sims[:, i, :, :]
        detectability = np.sum(gra_all_sims>th_gra, axis=0)/len(gra_data_all_t_all_sims)

        plt.pcolormesh(grav_x, grav_y, detectability, cmap='jet', shading='gouraud')
        plt.clim(0, 0.4)
        if i == 11:
            cbar = plt.colorbar()
            cbar.set_label('detectability')
        if i < 6 and i > 2:
            n_con = 1
        elif i <= 2:
            n_con = 0
        else:
            n_con = 3
        # make contour plots, number of contuors are 3
        c = plt.contour(grav_x, grav_y, detectability, n_con, cmap='jet')
        # add label for each contour lines , font color black
        plt.clabel(c, inline=True, fontsize=8, colors='black')
        plt.title('time='+str(int(time))+' th='+str(th_gra))
        # remove ticks
        largest_indices = np.unravel_index(
            np.argsort(detectability.ravel())[-3:], detectability.shape)

        # Dictionary to store the data for this subplot
        subplot_data = {'threshold': th_gra, 'time': time, 'points': []}
    # List to store the locations for this subplot
        locations = []
        for x, y in zip(largest_indices[1], largest_indices[0]):
            plt.scatter(grav_x[x], grav_y[y], color='white', marker='x')  # Mark the point
            locations.append({'x': float(grav_x[x]), 'y': float(grav_y[y])})
            # Record the data for this subplot
            subplot_data['points'].append({'x': grav_x[x], 'y': grav_y[y],
                                           'detectability': detectability[y, x]})

        # Save the locations to a YAML file for this subplot
        yaml_filename = f'{outdir}/optimized_location_th{th_gra}_time{int(time)}_locations.yaml'
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(locations, yaml_file, default_flow_style=False)
        if i == 0:
            plt.ylabel('Northing(m)')
        else:
            plt.gca().set_yticks([])
        if j == 4:
            plt.xlabel('Easting(m)')
        else:
            plt.gca().set_xticks([])

plt.tight_layout()
plt.savefig(outdir+'/detectability_time_th.png')

th_gra = 10
plt.figure()
for th_gra in ths:
    det_array = []

    for i, time in enumerate(times):
        gra_all_sims = gra_data_all_t_all_sims[:,i,:,:]
        detectability = np.sum(gra_all_sims>th_gra, axis=0)/len(gra_data_all_t_all_sims)
        max_det = np.max(detectability)
        det_array.append(max_det)
    plt.plot(times,det_array,'o-',label='th='+str(th_gra))
plt.legend()
plt.xlabel('time (yr)')
plt.ylabel('max detectability')
plt.title('max detectability vs time')
plt.savefig(outdir+'/max_detectability_vs_time.png')
