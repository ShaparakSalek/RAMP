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
from utilities.data_readers.read_input_from_files import get_all_h5_filenames,read_yaml,download_data_from_edx
from utilities.read_write_hdf5 import *
import argparse

 
class GravityMonitoringOptimization:
    '''
    This class is designed for optimizing gravity monitoring. It initializes parameters for gravity monitoring and performs calculations and visualizations based on the provided configurations.

    Methods:
    - __init__(self, input_yaml_path): Initializes the class with parameters from a YAML file.
    - Other methods are described below.
    '''
    def __init__(self,input_yaml_path):
        '''
        Initialize the GravityMonitoringOptimization class with parameters from a YAML file.

        Parameters:
        - input_yaml_path (str): Path to the YAML file containing necessary parameters.

        The method sets up various parameters for gravity monitoring, including simulation details, grid dimensions, subregion boundaries, and measurement stations.

        Attributes Set:
        - Various attributes are set up based on the YAML file, including directories, thresholds, grid dimensions, and measurement locations.
       
        '''
        yaml_paras=read_yaml(input_yaml_path)
        params=yaml_paras['GravityMonitoringOptimization']

        # Extracting and setting up parameters
        incomplete_simulations = params['incomplete_simulations']
        nSimulations = params['nSimulations']
        years = params['years']
        nt = len(years)

        rootdir = params['rootdir']
        workspace_id=params['workspace_id']
        data_folder_id=params['data_folder_id']
        api_key=params['api_key']
        if not glob.glob(rootdir + 'sim*'):
            download_data_from_edx(workspace_id,data_folder_id,api_key,rootdir) 
            print('downloaded data from edx')
            for file in glob.glob(rootdir + 'sim*'):
                print(file)
                os.makedirs(file[:-4])
                shutil.unpack_archive(file, file[:-4])
                os.remove(file)

        self.outdir = params['outdir']
        self.ths=params['ths']
        all_gra_sims_fn=get_all_h5_filenames(rootdir)
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

        gra_data_all_t_all_sims=[]
        for gra_file_name in all_gra_sims_fn:
            if gra_file_name[-7:-3] in incomplete_simulations:
                continue
            #print(gra_file_name)
            dum_hdf=h5py.File(gra_file_name, 'r')
            groups = get_groups('/', dum_hdf)
            groups=natsorted(groups)
            gra_data_all_t_step=[]
            for j,group in enumerate(groups):
                datasets = get_datasets(group+'/', dum_hdf)
                if j>1:
                    gra_data_all_t_step.append(dum_hdf[datasets[1]])
            gra_data_all_t_all_sims.append(gra_data_all_t_step)
            
        self.gra_data_all_t_all_sims=np.abs(np.array(gra_data_all_t_all_sims))


        datasets = get_datasets(groups[0]+'/', dum_hdf)

        ert_abmn=np.array(dum_hdf[datasets[0]])
        self.grav_x=np.array(dum_hdf[datasets[1]])
        self.grav_y=np.array(dum_hdf[datasets[2]])
        porosity=np.array(dum_hdf[datasets[3]])
        steps=np.array(dum_hdf[datasets[4]])
        self.times=np.array(dum_hdf[datasets[5]])
        x=np.array(dum_hdf[datasets[10]])
        y=np.array(dum_hdf[datasets[12]])
        z=np.array(dum_hdf[datasets[14]])
        dum_hdf.close()
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)


    def calculate_detectability(self):
        '''
        Calculate the detectability of gravity measurements.

        This method computes the detectability of gravity measurements at each time step for each threshold. Detectability is defined as the fraction of gravity measurements that exceed a given threshold. It is calculated for every simulation and at each time step.

        The detectability is stored in a 4D array with dimensions (n_ths, n_times, n_grav_y, n_grav_x), where:
        - n_ths: Number of thresholds
        - n_times: Number of time steps
        - n_grav_y: Number of gravity measurements along the y-axis
        - n_grav_x: Number of gravity measurements along the x-axis

        The first two dimensions of the array represent the thresholds and the time steps, respectively. The third and fourth dimensions correspond to the gravity measurements on the ground surface.

        The method iterates over all thresholds and time steps, calculating the fraction of gravity measurements that are above the current threshold. This fraction is computed for each point on the ground surface and stored in the detectability array.

        Attributes Updated:
        - self.detect_array: A 4D numpy array containing the detectability values.

        Returns:
        None
        '''
        detect_array=np.zeros((len(self.ths), len(self.times),len(self.grav_y),len(self.grav_x)))
        for j,th_gra in enumerate(self.ths):
            for i,time in enumerate(self.times):
                gra_all_sims=self.gra_data_all_t_all_sims[:,i,:,:]
                detect_array[j,i,:,:]=np.sum(gra_all_sims>th_gra,axis=0)/len(self.gra_data_all_t_all_sims)
        self.detect_array=detect_array


    def plot_detectability(self):
        '''
        Visualize the detectability of gravity measurements over different thresholds and time steps.

        This method creates a series of subplot visualizations, each representing the detectability for a specific threshold and time step. The detectability data is plotted as a color mesh and contoured for better visualization. The method also marks the locations with the highest detectability values and saves these locations in a YAML file.

        A color bar and appropriate labels are added for clarity. The final plot layout is adjusted for tightness and then saved as an image file.

        Outputs:
        - An image file visualizing the detectability over different thresholds and time steps.
        - YAML files containing the locations of highest detectability for each subplot.

        '''
        k=0
        plt.figure(figsize=(32,12))
        for j,th_gra in enumerate(self.ths):
            for i,time in enumerate(self.times):
                k=k+1
                plt.subplot(len(self.ths),12,k)

                detectability=self.detect_array[j,i,:,:]
                
                plt.pcolormesh(self.grav_x,self.grav_y,detectability,cmap='jet',shading='gouraud')
                plt.clim(0,0.4)
                if i==11:
                    cbar = plt.colorbar()
                    cbar.set_label('detectability')
                if i<6 and i>2:
                    n_con=1
                elif i<=2:
                    n_con=0
                else:
                    n_con=3
                # make contour plots, number of contuors are 3
                c=plt.contour(self.grav_x,self.grav_y,detectability,n_con,cmap='jet')
                # add label for each contour lines , font color black
                plt.clabel(c, inline=True, fontsize=8, colors='black')
                plt.title('time='+str(int(time))+' th='+str(th_gra))
                # remove ticks
                largest_indices = np.unravel_index(np.argsort(detectability.ravel())[-3:], detectability.shape)
                
                # Dictionary to store the data for this subplot
                subplot_data = {'threshold': th_gra, 'time': time, 'points': []}
            # List to store the locations for this subplot
                locations = []
                for x, y in zip(largest_indices[1], largest_indices[0]):
                    plt.scatter(self.grav_x[x], self.grav_y[y], color='white', marker='x')  # Mark the point
                    locations.append({'x': float(self.grav_x[x]), 'y': float(self.grav_y[y])})
                    # Record the data for this subplot
                    subplot_data['points'].append({'x': self.grav_x[x], 'y': self.grav_y[y], 'detectability': detectability[y, x]})

                # Save the locations to a YAML file for this subplot
                yaml_filename = f'{self.outdir}/optimized_location_th{th_gra}_time{int(time)}_locations.yaml'
                with open(yaml_filename, 'w') as yaml_file:
                    yaml.dump(locations, yaml_file, default_flow_style=False)
                if i==0:
                    plt.ylabel('Northing(m)')
                else:
                    plt.gca().set_yticks([])
                if j==4:
                    plt.xlabel('Easting(m)')
                else:
                    plt.gca().set_xticks([])
                
                
        plt.tight_layout()
        plt.savefig(self.outdir+'/detectability_time_th.png')

    def plot_max_detectability_vs_time(self):
        """
        Plots the maximum detectability versus time for different thresholds.

        This method iterates over the set thresholds and calculates the maximum detectability at each time step. It then plots these maximum detectability values over time for each threshold, providing a visual representation of how detectability changes over time under different threshold settings.
        """
        plt.figure()
        for th_gra in self.ths:
            det_array = []

            for i, time in enumerate(self.times):
                gra_all_sims = self.gra_data_all_t_all_sims[:, i, :, :]
                detectability = np.sum(gra_all_sims > th_gra, axis=0) / len(self.gra_data_all_t_all_sims)
                max_det = np.max(detectability)
                det_array.append(max_det)

            plt.plot(self.times, det_array, 'o-', label='th=' + str(th_gra))

        plt.legend()
        plt.xlabel('time (yr)')
        plt.ylabel('max detectability')
        plt.title('max detectability vs time')
        plt.savefig(self.outdir + '/max_detectability_vs_time.png')

if __name__ == "__main__":
    #input_yaml_path='./control_file_interface.yaml'
    # Command-line argument parsing to get the YAML configuration file
    parser = argparse.ArgumentParser(description='Seismic Design Optimization Script')
    parser.add_argument('--config', type=str, default='control_file_interface.yaml',
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()
    yaml_path = args.config
    grav_data=GravityMonitoringOptimization(yaml_path)
    grav_data.calculate_detectability()
    grav_data.plot_detectability()
    grav_data.plot_max_detectability_vs_time()