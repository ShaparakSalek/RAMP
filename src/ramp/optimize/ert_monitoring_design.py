# -*- coding: utf-8 -*-
"""
Created on 2024-01-25
@author: Xianjin Yang, LLNL (yang25@llnl.gov), Yuan Tian, LLNL (tian7@llnl.gov)
ERT electrode index is 0-based in abmn
"""
import time, yaml
import os, sys, glob, h5py, multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import multiprocessing as mp

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from utilities.data_readers.read_input_from_files import read_yaml_parameters,read_sens_from_segy,download_data_from_edx
# from utilities.write_output_to_files import convert_matrix_to_segy,write_optimal_design_to_yaml
mpl.rcParams.update({'font.size': 18})
DEBUG = False

# -------------------------------
def create_cmap():
    ''' Create a custom linear colormap of two colors

    Returns
    -------
    cmap: LinearSegmentedColormap
    '''
    colors = ['yellow', 'red']
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=64)
    return cmap

# -------------------------------
def read_yaml_parameters(file_path):
    """
    Read and flatten parameters from a given YAML file.
    Ignore the section name or nested dictionaries

    Parameters:
        file_path (str): Path to the input YAML file.

    Returns:
        flattened_params (dict): Flattened dictionary of parameters read from the YAML file.
    """
    with open(file_path, 'r') as file:
        x = yaml.safe_load(file)
        params = x['ERTMonitoringOptimization']
    # Flattening the restructured YAML content
    flattened_params = {}
    for section, values in params.items():
        if isinstance(values, dict):  # Adding a check for dictionary items
            for key, value in values.items():
                flattened_params[key] = value
        else:
            flattened_params[section] = values
    return flattened_params

# ---------------------------------
def get_groups(key, archive):
    ''' return a list of groups with or without datasets
    '''
    if key[-1] != '/': key += '/'
    out = []
    for name in archive[key]:
        path = key + name
        if isinstance(archive[path], h5py.Group):
            out += [path]
            out += get_groups(path, archive)

    return out

# ---------------------------------
def get_datasets(key, archive):
    ''' return a list of paths to all dataset
    https://stackoverflow.com/questions/49851046/merge-all-h5-files-using-h5py
    '''
    if key[-1] != '/': key += '/'
    out = []
    for name in archive[key]:
        path = key + name
        if isinstance(archive[path], h5py.Dataset):
            out += [path]
        else:
            out += get_datasets(path, archive)
    return out


# ========================
class MonitoringDesignERT:
    '''Evaluate ERT arrays and select optimal ERT arrays for CO2 detection'''
    def __init__(self, yaml_path):
        """
        Initializes seismic parameters from a YAML file.

        Parameters:
            yaml_path (str): Path to the input YAML file.

        """
        self.params = read_yaml_parameters(yaml_path)
        self.ert = dict()
        self.co2 = dict()
        self.tds = dict()

        outdir = self.params['outdir']
        self.outdir = os.path.join(outdir, '')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        for f in self.params['output_format']:
            d = os.path.join(outdir, f)
            if not os.path.exists(d):
                os.makedirs(d)

        d = os.path.join(outdir, 'monitoring_design')
        if not os.path.exists(d):
            os.makedirs(d)

        d = os.path.join(outdir, 'detection')
        if not os.path.exists(d):
            os.makedirs(d)

        # simulation index lookup table. A h5 file name: sim0023.h5
        self.simulation_index = []
        for i in range(self.params['nSimulations']):
            sSim = f'{i+1:04d}'
            if sSim in self.params['incomplete_simulations']: continue
            self.simulation_index.append('sim' + sSim)
        self.mass_centroid_file = self.params['mass_centroid_file']
        self.dfmass = pd.read_csv(self.mass_centroid_file)

    # ---------------------------------
    def read_co2_tds_from_h5(self, h5file):
        # print('CO2 TDS', h5file)
        hdf5_data = h5py.File(h5file, 'r')

        tds = []
        co2 = []
        tds0 = np.array(hdf5_data['/t0/tds'])
        nt = len(self.times)
        for t in range(nt):  # nt is defined at the top
            d = np.array(hdf5_data['/t' + str(t) + '/tds'])
            d = d - tds0
            tds.append(d)
            # print('/t'+str(t), d.shape, np.min(d), np.max(d))
            d = np.array(hdf5_data['/t' + str(t) + '/saturation'])
            co2.append(d)
            # print('/t'+str(t), d.shape, np.min(d), np.max(d))

        hdf5_data.close()
        sim = h5file[-10:-3]
        # self.tds[sim] = np.array(tds)
        # self.co2[sim] = np.array(co2)
        return sim, np.array(co2), np.array(tds)

    # ---------------------------------
    def load_co2_tds_from_h5(self, h5dir):
        '''Read co2 sat and tds data from a h5 file
        '''
        print('--- read_co2_tds_from_h5 ')
        h5files0 = glob.glob(h5dir + '*.h5')
        h5files0.sort()
        h5files = []
        for h5file in h5files0:
            sim = h5file[-7:-3]
            if sim in self.params['incomplete_simulations']: continue
            h5files.append(h5file)

        hdf5_data = h5py.File(h5files[0], 'r')
        self.times = np.array(hdf5_data[f'/data/times'])

        # NUFT simulation mesh - voxel center
        self.xc = np.array(hdf5_data['/data/x'])
        self.yc = np.array(hdf5_data['/data/y'])
        self.zc = np.array(hdf5_data['/data/z'])

        # NUFT simulation mesh - voxel vertex
        self.xv = np.array(hdf5_data['/data/vertex-x'])
        self.yv = np.array(hdf5_data['/data/vertex-y'])
        self.zv = np.array(hdf5_data['/data/vertex-z'])

        nCores = mp.cpu_count() - 1
        pool = mp.Pool(nCores//2)
        results = pool.map(self.read_co2_tds_from_h5, h5files)
        # pool.close()
        # pool.join()
        # pool.terminate()

        for (sim, co2, tds) in results:
            self.co2[sim] = co2
            self.tds[sim] = tds

        print(len(self.co2))
        for sim in self.simulation_index[:10]:
            print(sim, np.min(self.co2[sim]), np.max(self.co2[sim]))

    # ---------------------------------
    def read_ert_from_h5(self, h5file):
        print('--- read_ert_from_h5', h5file)
        hdf5_data = h5py.File(h5file, 'r')
        nt = len(self.times_ert)

        data = []
        for t in range(nt):  # nt is defined at the top
            d = np.array(hdf5_data['/t' + str(t) + '/ert_appres'])
            data.append(d)

        hdf5_data.close()
        data = np.array(data)
        data = 100*(data[1:,:] - data[0,:]) / data[0,:]

        sim = h5file[-10:-3]
        # self.ert[sim] = np.abs(data)
        return sim, np.abs(data)
    # ---------------------------------
    def load_ert_data(self, h5dir):
        '''Read ERT data (apparent resistivity) from a h5 file
        '''
        print('--- load_ert_data')
        h5files0 = glob.glob(h5dir + '*.h5')
        h5files0.sort()
        h5files0 = h5files0[:self.params['nSimulations']+1]

        h5files = []
        for h5file in h5files0:
            sim = h5file[-7:-3]
            if sim in self.params['incomplete_simulations']: continue
            h5files.append(h5file)

        hdf5_data = h5py.File(h5files[0], 'r')
        self.abmn = np.array(hdf5_data['/data/ert_abmn'])
        self.xe = np.array(hdf5_data['/data/x_ert'])
        self.ye = np.array(hdf5_data['/data/y_ert'])
        self.ze = np.array(hdf5_data['/data/z_ert'])
        self.nElectrodes = len(self.xe)
        self.times_ert = np.array(hdf5_data[f'/data/times_ert'])

        args = [h5file for h5file in h5files]
        nCores = mp.cpu_count() - 1
        pool = mp.Pool(nCores//2)
        results = pool.map(self.read_ert_from_h5, args)
        pool.close()
        pool.join()

        for (sim, ert) in results:
            self.ert[sim] = ert

        print(len(self.ert))
        for sim in self.simulation_index[:10]:
            print(sim, np.min(self.ert[sim]), np.max(self.ert[sim]))
    # ---------------------------------
    def compute_pseudoxyz(self):
        '''calculate x and z coordinates of ERT pseudosection
        Five 2D ERT lines at different y coordinates
        assume ert arrays along x axis
        y = 1500, 2000, 2500, 30000, 3500m
        x is located at the middle of four electrodes
        z is 0.25 times the largest separation of 4 electrodes
        '''
        print('--- compute_pseudoxyz')
        print('abmn:', self.abmn.shape, np.min(self.abmn), np.max(self.abmn))
        print('xe:', self.xe.shape, np.min(self.xe), np.max(self.xe))
        print('ye:', self.ye.shape, np.min(self.ye), np.max(self.ye))
        print('ze:', self.ze.shape, np.min(self.ze), np.max(self.ze))

        pseudoxyz = []
        for array in self.abmn:
            # print(array)
            a,b,m,n = array
            x = (self.xe[a] + self.xe[b] + self.xe[m] + self.xe[n]) / 4
            xmin = min([self.xe[a],self.xe[b],self.xe[m],self.xe[n]])
            xmax = max([self.xe[a],self.xe[b],self.xe[m],self.xe[n]])
            z = 0.25 * (xmax - xmin)
            y = self.ye[a]
            pseudoxyz.append([x,y,z])

        self.pseudoxyz = np.array(pseudoxyz)
        print('pseudo-z min max', np.min(self.pseudoxyz[:,2]),
              np.max(self.pseudoxyz[:,2]))

    # ---------------------------------
    def build_ert_arrays(self):
        '''find and contruct ert arrays from simulated data
        arrays is a dict(), its key is a tuple (a,b).
        arrays' values are a list of lists or 2D arrays of data indices
        '''
        print('--- build_ert_array')
        ert_arrays = dict()
        for i, abmn in enumerate(self.abmn):
            a,b,m,n = abmn
            if (a,b) in ert_arrays.keys():
                ert_arrays[(a,b)].append(i)
            else:
                ert_arrays[(a,b)] = [i]

        # 2D list with variable list lengths, not fit for np array
        self.ert_arrays = list(ert_arrays.values())
        # for a in self.ert_arrays: print(a)

    # ---------------------------------
    def plot_detection_scatter(self, detections):
        '''
        Parameters
        ----------
        detections: ndarray(dtype=int, ndim=2)
            2D numpy array of num-detections of [nTimes, nData]

        Returns
        -------

        '''
        outdir = os.path.join(self.outdir, 'detection/')
        nTimes = len(self.times_ert) - 1  # delete baseline
        yu = np.unique(self.ye)               # unique y coordinates
        ny = len(yu)                          # number of ERT survey lines
        y0 = np.min(self.ye)
        dy = (np.max(self.ye) - y0) / (ny-1)  # Line spacing
        nData = (len(self.abmn) + 1) // ny    # number of data per line
        print('Electrode yu,ny,dy', yu,ny,dy)
        print('Number of obs per line:', nData)

        print('plot n_detections per time point along the middle line')
        for j in range(nTimes):
            year = f'{int(self.times_ert[j+1]):03d}'
            iy = ny // 2   # middle ERT profile
            y = y0 + dy * iy
            x = self.pseudoxyz[:,0][iy*nData:(iy+1)*nData]
            z = self.pseudoxyz[:,2][iy*nData:(iy+1)*nData]
            d = detections[j,iy*nData:(iy+1)*nData]
            print(year + ' years, min max detections', np.min(d), np.max(d))

            fig = plt.figure(figsize=(16, 6))
            sp = plt.scatter(x, z, c=d, s=30, marker='s', cmap='YlOrRd',
                             vmin=0, vmax=120)
            plt.plot([5000, 5000], [0, 1500], 'k-', lw=2)
            cbar = plt.colorbar(sp, orientation='horizontal',
                         label='Number of leaks detected',
                         fraction=0.04, shrink=0.35, pad=0.17, aspect=30)
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(label='Number of detected leaks',size=16)

            plt.ylim(plt.ylim()[::-1])   # plot depth
            plt.gca().set_ylim([1500, 0])
            plt.xlim(3000, 9000)
            plt.title(year + ' years', fontsize=20)
            plt.grid(True, lw=0.2, linestyle='--')
            plt.xlabel('Horizontal distance (m)')
            plt.ylabel('Depth (m)')
            fig.tight_layout()
            plt.savefig(outdir + 'detections_by_obs_' + year + 'years.png',
                        bbox_inches='tight')
            plt.cla()
            plt.clf()
            plt.close()

    # ---------------------------------
    def plot_nleaks_detected(self, nleaks):
        '''
        plot number of leaks detected with increasing number of ERT arrays
        Parameters
        ----------
        nleaks: list(list)
            2D list, number of leaks detected with increasing number of ert arrays per time point
            shape = (nTimes, n_arrays_needed)

        Returns
        -------

        '''
        outdir = os.path.join(self.outdir, 'monitoring_design/')
        nTimes = len(self.times_ert) - 1  # excluding baseline
        fig = plt.figure(figsize=(14, 8))
        nflat = 3
        maxy = 0
        for j in range(nTimes):
            year = f'{int(self.times_ert[j+1]):03d}'
            y = nleaks[j].copy()
            y += nflat*[y[-1]]
            maxy = max(maxy, max(y))
            x = np.arange(1, len(y)+1)

            if j%2==0:
                p = plt.plot(x, y, ls='--', lw=1, label=year+' years')
            else:
                p = plt.plot(x, y, ls='-', lw=1, label=year+' years')
            c = p[0].get_color()
            plt.scatter(x[:-nflat], y[:-nflat], c=c, s=15)

        # maxy = (maxy//5+1) * 5
        # plt.ylim(0, maxy)
        plt.legend(loc='lower right')
        plt.grid(True, lw=0.2, linestyle='--')
        plt.xlabel('Number of ERT arrays (sources)')
        plt.ylabel('Number of potential leaks detected')
        fig.tight_layout()
        plt.savefig(outdir + 'number_of_scenarios_detected_ert.png',
                    bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

    # ---------------------------------
    def find_optimal_design(self, detections):
        ''' ERT optimal survey design - one 2D line
        Parameters
        ----------
        detections: ndarray(dtype=int, ndim=3)
            3D numpy array of leak detections (0 or 1) [nTimes, nSim, nArrays]

        Returns
        -------
        design_arrays: list(list)
            2D list, shape = (nTimes, n_array_needed)
            ERT array indices per time point. Number of arrays varies per time point
            e.g., [[96], [105, 104, 108], [105, 104, 109], [103, 104, 105, 109],
            [103, 105, 47, 107, 109, 104], [105, 102, 104, 108, 110, 11, 106],
            [105, 102, 108, 104, 113, 100, 11, 106], [105, 103, 109, 102, 107, 46, 104, 111, 11],
            [103, 110, 104, 108, 113, 106, 45, 11], [104, 108, 103, 115, 106, 48, 11, 112, 107, 105],
            [105, 109, 104, 49, 118, 100, 107, 108, 325, 11]]
        design_leaks: list(list)
            2D list, shape = (nTimes, nSim)
            Detected leaks (sim) (1=detected, 0=nondetect) per time point corresponding
            to design_arrays (combination of arrays per time point)
        design_nleaks: list(list)
            2D list, shape = (nTimes, n_arrays_needed)
            number of leaks detected with increasing number of ert arrays per time point
            e.g., [[2], [25, 27, 28], [67, 70, 71], [92, 96, 98, 99], [115, 126, 129, 131, 132, 133],
            [122, 132, 136, 139, 141, 142, 143], [136, 148, 152, 156, 158, 159, 160, 161],
            [137, 152, 159, 164, 167, 169, 171, 172, 173], [126, 142, 151, 156, 159, 161, 162, 163],
            [128, 145, 152, 156, 160, 162, 163, 164, 165, 166],
            [124, 142, 150, 155, 159, 162, 164, 165, 166, 167]]

        '''
        print('--- ERT optimal survey design')
        nTimes, nSim, nArrays = detections.shape
        idx = self.params['ert_line_id']
        nlines = self.params['num_ert_lines']
        nArray1 = nArrays // nlines

        detections_line = np.copy(detections[:,:,(idx-1)*nArray1:idx*nArray1])
        # ERT design of array indices and corresponding detected leaks (sim idx) per time point
        design_arrays = []   # ERT array idx
        design_leaks = []    # detected leak idx
        design_nleaks = []   # number of leaks detected with increasing num of ERT arrays
        for i in range(nTimes):
            year = str(int(self.times_ert[i + 1]))
            detect_t = np.copy(detections_line[i,:,:])  # [nSim, nArrays]
            # number of leaks detected per array
            detect_sum = np.sum(detect_t, axis=0)
            sorted = np.argsort(detect_sum)
            # ert array idx with the highest leak detection
            idx = sorted[-1]
            design_array_ti = [idx]
            design_leak_ti = np.copy(detect_t[:, idx])
            design_nleak_ti = [np.sum(design_leak_ti)]

            while True:
                # find the array that can detect most new leaks
                nNewLeaks = []
                for j in range(nArray1):
                    leak2 = design_leak_ti | detect_t[:, j]
                    nNewLeaks.append(np.sum(leak2) - np.sum(design_leak_ti))

                iMaxLeak = np.argsort(nNewLeaks)[-1]
                # unable to detect new leaks
                if nNewLeaks[iMaxLeak] <= 0: break

                leak2 = design_leak_ti | detect_t[:, iMaxLeak]
                design_leak_ti = np.copy(leak2)
                design_array_ti.append(iMaxLeak)
                design_nleak_ti.append(np.sum(design_leak_ti))

            design_arrays.append(design_array_ti)
            design_leaks.append(design_leak_ti)
            design_nleaks.append(design_nleak_ti)

            print('-', year, 'years')
            print('array_idx', design_array_ti)
            print('nLeaks detected', design_nleak_ti)

        return design_arrays, design_leaks, design_nleaks
            

    # ---------------------------------
    def evaluate_ert_arrays(self):
        '''

        Returns
        -------
        detections_array: ndarray(dtype=int, ndim=3)
            3D int numpy array of leak detections, shape=(nTimes, nSim, nArrays)
        detections_sum: ndarray(dtype=int, ndim=2)
            2D int numpy array of leaks detected per time & data point, shape=(nTimes, nData)
        '''

        print('--- evaluate_detection')
        nTimes = len(self.times_ert) - 1  # delete baseline
        nData = len(self.abmn)
        nSim = len(self.simulation_index)
        nArrays = len(self.ert_arrays)
        print('nTimes,nSim,nData,nArrays', nTimes,nSim,nData,nArrays)  # 11,10000,1000,1765

        # Detect(1) or nondetect(0) per ERT data indices per time point per sim (leak)
        detections_full = np.zeros((nTimes, nSim, nData), dtype=np.int32)
        # Number of Detections (sum) per ERT data indices per time point
        detections_sum = np.zeros((nTimes, nData), dtype=np.int32)

        for i in range(nTimes):
            for j, sim in enumerate(self.simulation_index):
                detect = np.int32(self.ert[sim][i,:] > self.params['ert_threshold'])
                detections_sum[i,:] += detect
                detections_full[i,j,:] = detect[:]

        # Detect(1) or nondetect(0) per array per time point per sim (leak)
        # Does ert-array_k detect leak_j at time_i
        detections_array = np.zeros((nTimes, nSim, nArrays), dtype=np.int32)
        for i in range(nTimes):
            year = str(int(self.times_ert[i+1]))
            print(year, 'years')
            for j, sim in enumerate(self.simulation_index):
                for k in range(nArrays):
                    detect = False
                    for a in range(len(self.ert_arrays[k])):
                        m = self.ert_arrays[k][a]
                        detect = detect or detections_full[i, j, m]
                    detections_array[i,j,k] = int(detect)

        return detections_array, detections_sum

    # ---------------------------------
    def save_output_csv(self, arrays, leaks, nleaks):
        '''
        Save optimal arrays one file per time point in CSV format
        Save detected leaks (sims) one file per time point

        csv format (xyz of ABMN)
        xa ya za xb yb zb xm ym zm xn yn zn
        Parameters
        ----------
        arrays: list(list)
            2D list, shape = (nTimes, n_array_needed)
            ERT array indices per time point. Number of arrays varies per time point
        leaks: list(list)
            2D list, shape = (nTimes, nSim)
            Detected leaks (sim) (1=detected, 0=nondetect) per time point corresponding
            to design_arrays (combination of arrays per time point)
        nleaks: list(list)
            2D list, shape = (nTimes, n_arrays_needed)
            number of leaks detected with increasing number of ert arrays per time point

        Returns
        -------

        '''
        print('Save ERT monitoring design to csv and txt files')
        outdir = os.path.join(self.outdir, 'csv/')
        nTimes = len(arrays)
        for j in range(nTimes):
            year = f'{int(self.times_ert[j+1]):03d}'
            fname = 'optimal_design_' + str(year) + 'years.csv'
            fout = open(outdir + fname, 'w')
            fout.write('ArrayNo,Ax,Ay,Az,Bx,By,Bz,Mx,My,Mz,Nx,Ny,Nz,leaks_detected\n')
            nr = len(arrays[j])
            for i in range(nr):
                idx = arrays[j][i]
                for k in self.ert_arrays[idx]:
                    s = str(i + 1) + ','
                    a,b,m,n = self.abmn[k]
                    s += f'{self.xe[a]:.1f},{self.ye[a]:.1f},{self.ze[a]:.1f},'
                    s += f'{self.xe[b]:.1f},{self.ye[b]:.1f},{self.ze[b]:.1f},'
                    s += f'{self.xe[m]:.1f},{self.ye[m]:.1f},{self.ze[m]:.1f},'
                    s += f'{self.xe[n]:.1f},{self.ye[n]:.1f},{self.ze[n]:.1f},'
                    s += f'{nleaks[j][i]}\n'
                    fout.write(s)
            fout.close()

            fname = 'detected_scenarios_ctr_mass_volume_' + str(year) + 'years.csv'
            fout = open(outdir + fname, 'w')
            leak = leaks[j]
            sims = []
            for i in range(len(leak)):
                if leak[i] == 0: continue
                sims.append(int(self.simulation_index[i][3:]))

            ctr, mass, vol = self.find_mass_centroid(sims, int(year))
            fout.write('sim,cx,cy,cz,mass,volume\n')
            for i in range(len(sims)):
                x = f'{sims[i]:d},{ctr[i][0]:.1f},{ctr[i][1]:.1f},{ctr[i][2]:.1f},{mass[i]:.2e},{vol[i]:.3e}\n'
                fout.write(x)
            fout.close()

    # ---------------------------------
    def save_output_yaml(self, arrays, leaks):
        '''
        Save optimal arrays one file per time point in CSV format
        Save detected leaks (sims) one file per time point

        csv format (xyz of ABMN)
        xa ya za xb yb zb xm ym zm xn yn zn
        Parameters
        ----------
        arrays: list(list)
            2D list, shape = (nTimes, n_array_needed)
            ERT array indices per time point. Number of arrays varies per time point
        leaks: list(list)
            2D list, shape = (nTimes, nSim)
            Detected leaks (sim) (1=detected, 0=nondetect) per time point corresponding
            to design_arrays (combination of arrays per time point)

        Returns
        -------

        '''
        print('Save ERT monitoring design to yaml files')
        outdir = os.path.join(self.outdir, 'yaml/')
        nTimes = len(arrays)
        for j in range(nTimes):
            year = f'{int(self.times_ert[j+1]):03d}'
            out_dict = dict()
            nr = len(arrays[j])
            for i in range(nr):
                key = 'ERT_array_' + str(i+1)
                value = dict()
                idx = arrays[j][i]
                M = []; N = []
                for k, obs_idx in enumerate(self.ert_arrays[idx]):
                    # s = str(i + 1) + ','
                    a,b,m,n = self.abmn[obs_idx]
                    if k == 0:
                        value['A'] = f'{self.xe[a]:.1f},{self.ye[a]:.1f},{self.ze[a]:.1f}'
                        value['B'] = f'{self.xe[b]:.1f},{self.ye[b]:.1f},{self.ze[b]:.1f}'
                    M.append(f'{self.xe[m]:.1f},{self.ye[m]:.1f},{self.ze[m]:.1f}')
                    N.append(f'{self.xe[n]:.1f},{self.ye[n]:.1f},{self.ze[n]:.1f}')
                value['M'] = M
                value['N'] = N
                out_dict[key] = value

            detected = []
            leak = leaks[j]
            for i in range(len(leak)):
                if leak[i] != 0:
                    detected.append(int(self.simulation_index[i][3:]))

            out_dict['number_of_detected'] = len(detected)
            ctr, mass, vol = self.find_mass_centroid(detected, int(year))
            values = []
            for i in range(len(detected)):
                values.append(f'{detected[i]:d},{ctr[i][0]:.1f},{ctr[i][1]:.1f},{ctr[i][2]:.1f},{mass[i]:.2e},{vol[i]:.3e}')
            out_dict['sim_cx_cy_cz_mass_volume'] = values

            outfile = outdir + 'optimal_design_' + str(year) + 'years.yaml'
            with open(outfile,"w") as fout:
                yaml.dump(out_dict, fout)

    # ---------------------------------
    def find_mass_centroid(self, detected, yr):
        '''
        Find centroid xyz (m), mass (kg) and volume (m^3) from a lookup table
        Parameters
        ----------
        detected: list(int)
            a list of detected scenarios (simulation number)
        yr: int
            NUFT simulation time point

        Returns
        -------
        centroid: list(list(float))
        mass: list(float)
        volume: list(float)

        '''

        centroid = []; volume = []; mass = []
        for sim in detected:
            cmv = self.dfmass[(self.dfmass['sim']==sim) & (self.dfmass['year']==yr)]
            cmv = cmv.reset_index(drop=True)
            c = [cmv.loc[0,'cx'], cmv.loc[0,'cy'], cmv.loc[0,'cz']]
            centroid.append(c)
            volume.append(cmv.loc[0,'volume'])
            mass.append(cmv.loc[0,'mass'])
        return centroid, mass, volume

# ---------------------------------
    def save_output_h5(self, arrays, leaks):
        print('Save ERT monitoring design to an H5 file')
        outdir = os.path.join(self.outdir, 'h5/')
        hdf5 = h5py.File(outdir + 'ert_optimal_design.h5', 'w')

        nTimes = len(arrays)
        years = []
        for j in range(nTimes):
            year = self.times_ert[j+1]
            years.append(year)
            # one group per time point: 10 - 200 years
            group1 = hdf5.create_group('t' + str(j))

            nr = len(arrays[j])
            for i in range(nr):
                key = 'array_' + str(i+1)
                group2 = group1.create_group(key)

                # ERT source
                idx = arrays[j][i]
                obs_idx = self.ert_arrays[idx][0]
                a, b, m, n = self.abmn[obs_idx]
                A = [self.xe[a],self.ye[a],self.ze[a]]
                B = [self.xe[b],self.ye[b],self.ze[b]]
                group2.create_dataset('A', data=A, dtype='float32')
                group2.create_dataset('B', data=B, dtype='float32')

                M = []; N = []
                for k, obs_idx in enumerate(self.ert_arrays[idx]):
                    # s = str(i + 1) + ','
                    a,b,m,n = self.abmn[obs_idx]
                    M.append([self.xe[m],self.ye[m],self.ze[m]])
                    N.append([self.xe[n],self.ye[n],self.ze[n]])

                group2.create_dataset('M', data=M, dtype='float32')
                group2.create_dataset('N', data=N, dtype='float32')

            detected = []
            leak = leaks[j]
            for i in range(len(leak)):
                if leak[i] != 0:
                    detected.append(int(self.simulation_index[i][3:]))

            group1.create_dataset('number_of_scenarios detected', data=len(detected), dtype='int32')
            # str_type = h5py.string_dtype(encoding='utf-8')
            group1.create_dataset('scenarios_detected', data=detected, dtype='int32')
            centroid, mass, volume = self.find_mass_centroid(detected, int(year))
            group1.create_dataset('centroid', data=centroid, dtype='float64')
            group1.create_dataset('volume', data=volume, dtype='float64')
            group1.create_dataset('mass', data=mass, dtype='float64')

        hdf5.attrs['dataset'] = 'kimberlina 1.2'
        hdf5.attrs['years'] = years
        hdf5.attrs['data_type'] = 'ERT'
        hdf5.attrs['length_unit'] = 'm'
        hdf5.attrs['mass_unit'] = 'kg'
        hdf5.attrs['task'] = 'monitoring design'

        hdf5.close()

# ---------------------------------
    def plot_monitoring_design(self, arrays, nleaks):
        print('Plot and save ERT monitoring design figures')
        outdir = os.path.join(self.outdir, 'monitoring_design/')
        ne = len(self.xe) / 5
        x1 = min(self.xe); x2 = max(self.xe)
        dx = (x2 - x1) / (ne-1)
        nTimes = len(arrays)
        for j in range(nTimes):
            year = f'{int(self.times_ert[j+1]):03d}'
            yr = f'{int(self.times_ert[j+1])}'
            nr = len(arrays[j])
            fname = outdir + 'ert_monitoring_design_' + year + 'years.png'

            fig = plt.figure(figsize=(12, 6))
            for i in range(nr):
                y = i*dx*10
                p = plt.plot([x1,x2], [y,y], 'k-', lw=0.3)
                plt.scatter(self.xe, [y]*len(self.xe), c='k', s=1)

                # ERT source
                idx = arrays[j][i]
                obs_idx = self.ert_arrays[idx][0]
                a, b, m, n = self.abmn[obs_idx]
                Ax = self.xe[a]
                Bx = self.xe[b]
                nx = round(abs(Ax-Bx)/dx)
                # print(nx, Ax, Bx, dx, year)
                plt.scatter([Ax,Bx], [y,y], c='r', s=30)

                Mx = []; Nx = []
                for k, obs_idx in enumerate(self.ert_arrays[idx]):
                    # s = str(i + 1) + ','
                    a,b,m,n = self.abmn[obs_idx]
                    Mx.append(self.xe[m])
                    Nx.append(self.xe[n])

                # demo sparse array
                MN = list(set(Mx+Nx))
                MN.sort(); MN = MN[::nx]
                # print(Mx); print(Nx); print(MN)
                # plt.scatter(Mx, [y]*len(Mx), c='b', s=20)
                # plt.scatter(Nx, [y]*len(Nx), c='b', s=20)
                plt.scatter(MN, [y]*len(MN), c='b', s=20)

            w = nleaks[j].copy()
            if len(w) > 0: nd = w[-1]
            else: nd = 0

            plt.title(f'Number of Detected Scenarios = {nd}')
            import matplotlib.ticker as ticker
            plt.gca().yaxis.set_major_locator(ticker.NullLocator())
            plt.xlabel('X (m)')
            plt.ylabel(f'ERT Arrays at {yr} Years')
            plt.grid(True, lw=0.2, linestyle='--')
            plt.tight_layout()
            plt.savefig(fname, bbox_inches='tight')
            plt.close()

# ----------------------------------------
if __name__ == "__main__":
    input_file = 'control_file_interface.yaml'
    ert = MonitoringDesignERT(input_file)
    # ert.load_co2_tds_from_h5(ert.params['datadir'])
    ert.load_ert_data(ert.params['datadir'])
    ert.compute_pseudoxyz()
    ert.build_ert_arrays()

    detections_array, detections_sum = ert.evaluate_ert_arrays()
    ert.plot_detection_scatter(detections_sum)

    arrays, leaks, nleaks = ert.find_optimal_design(detections_array)
    ert.plot_nleaks_detected(nleaks)

    # print(leaks)
    print(arrays)
    print(nleaks)

    ert.plot_monitoring_design(arrays, nleaks)

    if 'csv' in ert.params['output_format']:
        ert.save_output_csv(arrays, leaks, nleaks)
    if 'yaml' in ert.params['output_format']:
        ert.save_output_yaml(arrays, leaks)
    if 'h5' in ert.params['output_format']:
        ert.save_output_h5(arrays, leaks)
    print('\nDone')
