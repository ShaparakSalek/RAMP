# -*- coding: utf-8 -*-
"""
Created on 2023-09-22
@author: Xianjin Yang, LLNL (yang25@llnl.gov), Yuan Tian, LLNL (tian7@llnl.gov)

Sensitivity based seismic monitoring design
NRAP RAMP Use Case 2
Contributors:
Xianjin Yang, LLNL
Lianjie Huang, LANL
Erika Gasperikova, LBL
Veronika Vasylkivska
Yuan Tian, LLNL
"""
import os, sys, glob, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
# from ipywidgets import interact, IntSlider, HBox, IntText, BoundedIntText
# import ipywidgets as widgets
import yaml
import argparse
mpl.rcParams.update({'font.size': 20})
DEBUG = False



# -------------------------------


def read_yaml_parameters(file_path):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

def create_cmap():
    '''
    Create a custom linear colormap

    Returns
    -------
    cmap: LinearSegmentedColormap
    '''
    colors = [(0.00, '#ffffff'),
              # (0.03, '#ffff00'),
              (1.00, '#ff0000')]
    cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=64)
    return cmap

class MonitoringDesignSensitivity2D:
    '''
    sensitivity-based 2D seismic monitoring design
    '''
    def __init__(self,nx,nz,dx,dz,ns,nr,ds,dr,yrs,t1,thresholds,ks,kr,sen_nor,datadir,outpre,
                 wavefield,vpvs,units):
        '''

        Parameters
        ----------
        nx: int
            number of horizontal grid points on seismic model mesh
        nz: int
            number of vertical grid points on seismic model mesh
        dx: float
            hotizontal grid interval
        dz: float
            vertical grid interval
        ns: int
            Number of seismic sourcs
        nr: int
            Number of seismic receivers
        ds: float
            Seismic source interval
        dr: float
            Seismic receiver interval
        yrs: list[str]
            A list of time steps, e.g., ['80', '85']
        t1: float
            the year when fault leakage begins
        thresholds: list[float]
            Two thresholds of scaled sensitivity, e.g., 0.2, 0.6
        ks: int
            Scale factor on the source interval in the simulated data
        kr: int
            Scale factor on the receiver interval in the simulated data
        datadir:ir: str
            root directory of sensitivity data
        outpre: str
            output directory
        wavefield: list[str]
            waveform (P or S)
        vpvs: list[str]
            sensitivity wrt Vp or Vs
        units: dict
            units of sensitivity data and model
        wavefield: list[str]
            waveform (P or S)
        vpvs: list[str]
            sensitivity wrt Vp or Vs
        units: dict
            units of sensitivity data and model
        '''

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.nsrc = ns
        self.nrec = nr
        self.dsrc = ds
        self.drec = dr
        self.timestamps = yrs
        self.t1 = t1
        self.thresholds = thresholds
        self.opt_src_sep = ks
        self.opt_rec_sep = kr
        self.sen_nor=sen_nor
        self.wavefield = wavefield
        self.vpvs = vpvs
        self.units = units
        self.datadir=datadir
        self.outpre=outpre
        # plot velocity, density and plume models (images)
        self.modeldir = self.datadir + 'models_plume_mask/'
    
        self.plume_images_dir = outpre + 'model_plume_images/'
        

        # plot sensitivity images per component
        self.sensitivity_images_out_dir = outpre + '/sensitivity_images/'
        if not os.path.exists(self.sensitivity_images_out_dir):
            os.makedirs(self.sensitivity_images_out_dir)

        self.optimal_dir = outpre + '/optimal_design/'
        if not os.path.exists(self.optimal_dir):
            os.makedirs(self.optimal_dir)



    def read_sensitivity_file(self, path, fname):
        '''
        Read one seismic sensitivity file along a surface profile (1D)
        Seismic sensitivity file is a binary file with single precision floats (4B)
        and little endian. The number of floats equals to the number of receivers
        Reference: Gao, K., and L. Huang, 2020, Numerical Modeling of Anisotropic
            Elastic-Wave Sensitivity Propagation for Optimal Design of Time-Lapse Seismic
            Surveys: Communications in Computational Physics, 28, no. LA-UR-18-27229.

        Parameters
        ----------
        path : str
            sensitivity data directory
        fname str
            sensitivity data file name

        Returns
        -------
        data : ndarray (dtype=float, ndim=1)
            an array of sensitivity data

        '''

        p = os.path.normpath(path) + '/'
        data = np.fromfile(p+fname, dtype=np.float32, count=-1, sep='')
        return data

    def load_sensitivity_data(self, datadir, yr, wf, vpvs,norm=0):
        ''' Load sensitivity data of all sources per senstivity component (wf, vpvs)
        and per time step (yr). Scale sensitivity data between 0 and 1

        Parameters
        ----------
        datadir: str
            root directory of sensitivity data
        yr: str
            a time step in years, e.g., '80'
        wf: str
            P or S waveform (data)
        vpvs: str
            sensitivity wrt Vp or Vs (model)

        Returns
        -------
        sens: ndarray(dtype=float, ndim=2)
            an array of sensitivity data in a shape (nSources, nReceivers)

        '''

        sDir = datadir + f'EW_sensitivity_results_ol_y{yr}_leakage/'
        sens = np.zeros((self.nsrc, self.nrec))
        for i in range(1, self.nsrc+1):
            fname = f'receiver_sensitivity_{wf}_wrt_{vpvs}_src_{i:d}.bin'
            sens[i-1,:] = self.read_sensitivity_file(sDir, fname.lower())
        if norm:
            maxSens = np.max(sens)
            minSens = np.min(sens)
            dsens = maxSens - minSens
            sens = (sens - minSens) / dsens
        return sens
    


    # -------------------------------
    def read_seismic_model(self, fname):
        '''
        Read a 2D binary seismic model file, e.g., Vp, Vs, density
        A binary seismic model data is saved in single precision floats (4B)
        and little endian. The number of floats = nz x nx. Data are located
        on an equally-spaced rectangular grid of nz rows and nx columns.
        The binary data are saved in a column-major order in a shape (nx,nz)

        Parameters
        ----------
        fname str
            Seismic model file name with path

        Returns
        -------
        data : ndarray (dtype=float, ndim=2)
            seismic (Vp, Vs or density) model in a 2D numpy array
        '''

        data = np.fromfile(fname, dtype=np.float32, count=-1, sep='')
        data = data.reshape(self.nx, self.nz)  # column major array
        return data

    def plot_model_image(self):
        ''' plot baseline Vp, Vs and density model and plume mask model

        '''
        cmap = create_cmap()
        if not os.path.exists(self.plume_images_dir):
            os.makedirs(self.plume_images_dir)
        fnames = glob.glob(self.modeldir + '*.bin')
        for fname in fnames:
            #seis.plot_model_image(fname, out_dir)
            model = self.read_seismic_model(fname)
            _, basename = os.path.split(fname)
            print(basename)
            tokens = basename.split('.')
            param = tokens[0].split('_')[0]  # vp, vs, den, mask

            fig, ax = plt.subplots(figsize=(12, 5))

            if param.lower() == 'mask':  # mask
                img = ax.imshow(model.T, extent=[0, self.nx * self.dx, self.nz * self.dz, 0], cmap=cmap)
                s = tokens[0][:-8]
                ss = s[-3:]
                if ss[0] == 'y': ss = ss[1:]
                yr = str(int(ss) - self.t1)
                annotate = 'CO$_2$ plume mask at t1+' + yr + ' years'
                label = 'CO$_2$ Plume Mask'
                # Add text using relative positioning
                relative_x_position = 0.03  # 3% from the left edge
                relative_y_position = 0.85  # 15% from the top edge (or 85% from the bottom)
                ax.text(relative_x_position, relative_y_position, annotate, transform=ax.transAxes, color='black', fontsize=24)
            else:
                img = ax.imshow(model.T, extent=[0, self.nx * self.dx, self.nz * self.dz, 0], cmap='jet')
                if len(param)==3: param='density'
                annotate = param.title()
                label = annotate + self.units[param]
                fig.colorbar(img, label=label, orientation='horizontal',
                        fraction=0.06, shrink=0.5, pad=0.18, aspect=30)
                # Add text using relative positioning
                relative_x_position = 0.03  # 3% from the left edge
                relative_y_position = 0.85  # 15% from the top edge (or 85% from the bottom)
                ax.text(relative_x_position, relative_y_position, annotate, transform=ax.transAxes, color='black', fontsize=24)

            ax.set_xlabel('Horizontal Distance (m)')
            ax.set_ylabel('Depth (m)')
            ax.grid(lw=0.2, alpha=0.5)
            plt.tight_layout()
            s = os.path.splitext(basename)[0]
            plt.savefig(self.plume_images_dir + s + '.png', bbox_inches='tight')
            if DEBUG: plt.show()
            plt.cla()
            plt.clf()
            plt.close()


    def plot_sensitivity_image(self, sens, outdir, yr, wf, vpvs, area):
        ''' Create a sensitivity image with x=receivers, y=sources
        per senstivity component and per time step (years)

        Parameters
        ----------
        outdir: str
            output image directory
        yr: str
            a time step in years, e.g., '80'
        wf: str
            P or S waveform (data)
        vpvs: str
            sensitivity wrt Vp or Vs (model)
        area: list[float]
            area of optimal monitoring design correspoding to
            two sensitivity thresholds

        Returns
        -------
        Save sensitivity image

        '''
        prefix = yr + 'y_' + wf + '-wavefield_' + vpvs
        clabel = 'W' + wf.lower() + '/' + vpvs
        fig, ax = plt.subplots(figsize=(10, 8))

        cmap = create_cmap()
        extent = [0, (self.nrec-1)*self.drec, (self.nsrc-1)*self.dsrc, 0]
        img = ax.imshow(sens,extent=extent,vmin=0.0,vmax=1.0,cmap=cmap)
        yrs = 't1+' + str(int(yr)-self.t1) + ' years'
        # Add text using relative positioning
        relative_left_position = 0.03  # 3% from the left edge
        relative_top_position = 0.75  # 15% from the top edge (or 85% from the bottom)
        relative_right_position = 0.86  # 95% from the right edge
        relative_bottom_position = 0.05  # 5% from the bottom edge
        ax.text(relative_left_position, relative_bottom_position, yrs, transform=ax.transAxes, color='k', fontsize=24)
        ax.text(0.6, relative_top_position+0.2, '%data to collect', transform=ax.transAxes, color='k')
        ax.text(relative_right_position, relative_top_position+0.1, f'{area[0]:.1f}%', transform=ax.transAxes, color='brown')
        ax.text(relative_right_position, relative_top_position, f'{area[1]:.1f}%', transform=ax.transAxes, color='m')

        ax.set_ylim(ax.get_ylim()[::-1])
        x = np.linspace(0, (self.nrec-1)*self.drec, self.nrec)
        y = np.linspace(0, (self.nsrc-1)*self.dsrc, self.nsrc)
        X, Y = np.meshgrid(x, y)
        CS = plt.contour(X,Y,sens,self.thresholds,vmin=0.0,vmax=1.0,linestyles='solid',
                    linewidths=[0.5, 1.5], colors=['brown', 'm'])
        ax.set_xlabel('Receiver Location (m)')
        ax.set_ylabel('Source Locations (m)')
        ax.grid(color='k', lw=0.3, alpha=0.2)
        fig.colorbar(img, label='Scaled Sensitivity ' + clabel, orientation='horizontal',
                     fraction=0.05, shrink=0.5, pad=0.125, aspect=30)
        plt.tight_layout()
        plt.savefig(outdir + 'sensitivity_image_' + prefix + '.png', bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()
    
    def find_optimal_seismic_arrays(self,sens,threshold,norm=0):
        '''
        Find optimal sources and receivers with scaled sensitivity
        above a threshold Scale source and reciver intervals

        Parameters
        ----------
        sens2d: ndarray(dtype=float, ndim=2)
            an array of sensitivity data in a shape (nSources, nReceivers)
        threshold: float
            Sensitivity thresholds in fraction between (0, 1)

        Returns
        -------
        out: ndarray(dtype=bool, ndim=2)
            an array of sensitivity masks showing optimal source and receiver locations

        '''
        if norm==1:
            out = sens >= threshold
        else:
            maxSens = np.max(sens)
            minSens = np.min(sens)
            dsens = maxSens - minSens
            out = sens >= (threshold*dsens  + minSens)
        
        sens_sel=sens*out
        # increase source interval by a factor of ks and receiver interval by kr
        out = out[::self.opt_src_sep, ::self.opt_rec_sep]
        sens_sel = sens_sel[::self.opt_src_sep, ::self.opt_rec_sep]

        m, n = out.shape
        opt = np.sum(out)
        area = 100 * opt / (m * n)
        
        return out, area,np.sum(sens_sel)

    def find_max_num_sources(self,src_rec):
        ''' find maximum number of sources for all time steps
        Assume more sources are needed at later time

        Parameters
        ----------
        src_rec: ndarray(dtype=bool, ndim=2)
            an array of sensitivity masks showing optimal source and receiver locations

        Returns
        -------
        ns_max: int
            max number of sources for all time steps
        '''

        ns, nr = src_rec.shape
        ns_max = 0
        for i in range(ns):
            s = np.int32(src_rec[i,:])
            if np.sum(s) <= 0: continue
            ns_max += 1
        return ns_max


    def plot_optimal_design(self,src_rec,modeldir,outdir,wf,ps,yr,ns_max):
        '''
        Plot and save optimal design with source, receivers and plume per time step
        and per sensitivity component

        Parameters
        ----------
        src_rec: ndarray(dtype=bool, ndim=2)
            an array of sensitivity masks showing optimal source and receiver locations
        modeldir: str
            Directory with baseline density and velocity models and plume mask data
        outdir: str
            Directory to save optimal design image
        yr: str
            a time step in years, e.g., '80'
        wf: str
            P or S waveform (data)
        ps: str
            sensitivity wrt Vp or Vs (model)
            Scale factor applied to receiver interval
        ns_max: int
            Max number of sources in optimal design for all time steps

        Returns
        -------
        design: list[float]
            Seismic source coordinate and then receiver coordinates
        '''

        plume_file = modeldir + 'mask_ol_y' + yr + '_leakage.bin'
        model = self.read_seismic_model(plume_file)
        outdir2 = outdir + wf + '_' + ps + '/'
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)

        ns, nr = src_rec.shape
        design = []
        fig, ax = plt.subplots(figsize=(12, 6))
        xMin = 0; xMax = self.nx * self.dx
        cmap = create_cmap()
        ax.imshow(model.T, extent=[0, self.nx * self.dx, self.nz * self.dz, 0], cmap=cmap)
        prefix = wf + '_' + ps + '_y' + yr

        k = 0
        for i in range(ns):
            s = np.int32(src_rec[i,:])
            if np.sum(s) <= 0: continue
            y0 = -100 - k*200
            k += 1
            xs = i * self.dsrc * self.opt_src_sep
            xr = np.float32(np.nonzero(s)) * self.drec * self.opt_rec_sep
            xr = xr[0]  # xr above is a 2D array
            design.append(xs)
            design.append(list(xr))

            plt.plot([xMin, xMax], [y0, y0], 'k--', lw=0.3)
            plt.plot([xs], [y0], 'r*', ms=8)
            plt.plot(xr, [y0]*len(xr), 'bo', ms=3)

        for j in range(k, ns_max):
            y0 = -100 - j*200
            plt.plot([xMin, xMax], [y0, y0], 'k--', lw=0.3)

        ax.set_yticks((0, 1000, 2000))
        ax.set_xlabel('Horizontal Distance (m)')
        ax.set_ylabel('Depth (m)', loc='bottom')
        rel_pos1=0.01
        rel_pos2=0.03
        yrs = str(int(yr) - self.t1)
        annotate = 'CO$_2$ plume mask at t1+' + yrs + ' years'
        ax.text(rel_pos1, rel_pos2, annotate, transform=ax.transAxes, color='black')
        ax.grid(lw=0.2, alpha=0.5)
        ax.set_aspect('auto')

        plt.savefig(outdir2 + prefix + '.png', bbox_inches='tight')
        plt.tight_layout()

        plt.cla()
        plt.clf()
        plt.close()
        return design





# ----------------------------------------
if __name__ == "__main__":
    # Command-line argument parsing to get the YAML configuration file
    parser = argparse.ArgumentParser(description='Seismic Design Optimization Script')
    parser.add_argument('--config', type=str, default='seis_sens_opt_params.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    yaml_path = args.config
    parameters = read_yaml_parameters(yaml_path)
    # read in parameters from YAML file
    nx = parameters['nx']
    nz = parameters['nz']
    dz = parameters['dz']
    dx = parameters['dx']
    ns = parameters['ns']
    nr = parameters['nr']
    dr = parameters['dr']
    ds = parameters['ds']
    t1 = parameters['t1']
    years0 = parameters['years']
    senMax = parameters['senMax']
    thresholds = parameters['thresholds']
    ks = parameters['ks']
    kr = parameters['kr']
    sen_nor=parameters['sen_nor']
    wavefield = parameters['wavefield']
    vpvs = parameters['vpvs']
    units= parameters['units']
    datadir = parameters['datadir']
    outpre = parameters['outpre']
    years = [str(y) for y in years0]
    # create a sensitivity object
    seis = MonitoringDesignSensitivity2D(nx,nz,dx,dz,ns,nr,ds,dr,years,t1,thresholds,ks,kr,sen_nor,datadir,outpre,wavefield,vpvs,units)

    seis.plot_model_image()

    # plot sensitivity images per component
    out_dir = outpre + '/sensitivity_images/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    optimal_dir = outpre + '/optimal_design/'
    if not os.path.exists(optimal_dir):
        os.makedirs(optimal_dir)

    area = [0.0, 0.0]
    for wf in seis.wavefield:
        for ps in seis.vpvs:
            sens2d = seis.load_sensitivity_data(datadir, seis.timestamps[-1], wf, ps)
            design1, _,sen_sel = seis.find_optimal_seismic_arrays(sens2d, seis.thresholds[1],norm=sen_nor)
            ns_max = seis.find_max_num_sources(design1)
            for yr in seis.timestamps:
                print('Optimal design', yr+'y', wf, ps)
                sens2d = seis.load_sensitivity_data(datadir,yr,wf,ps)
                _, area[0],sen_sel = seis.find_optimal_seismic_arrays(sens2d,seis.thresholds[0],norm=sen_nor)
                design1, area[1],sen_sel = seis.find_optimal_seismic_arrays(sens2d,seis.thresholds[1],norm=sen_nor)
                seis.plot_sensitivity_image(sens2d, out_dir,yr,wf,ps, area)
                design2 = seis.plot_optimal_design(design1,seis.modeldir,optimal_dir,wf,ps,yr,ns_max)
                fname = optimal_dir + wf + '_' + ps + '_' + yr + '.txt'
                with open(fname, 'w') as fout:
                    for line in design2:
                        fout.write(str(line) + '\n')
    dtc_flag=1
    if dtc_flag==1:
        sens_dtc_dir = outpre + '/sens_dtc/'
        if not os.path.exists(sens_dtc_dir):
            os.makedirs(sens_dtc_dir)
        
        plt.figure(figsize=(20,15))
        for yr in seis.timestamps[:]:
            plt.subplot(2,3,seis.timestamps.index(yr)+1)
            for wf in seis.wavefield:
                for ps in seis.vpvs:
                    sens2d = seis.load_sensitivity_data(datadir, yr, wf, ps)
                    arae_all=[]
                    sen_all=[]
                    for th in 1/np.arange(1,10):
                        design, area,sen_sel = seis.find_optimal_seismic_arrays(sens2d,th,norm=sen_nor)
                        arae_all.append(area)
                        sen_all.append(sen_sel)
                    plt.plot(arae_all,sen_all,'o-',label=wf+'_'+ps)
            plt.xlabel('data to collect (%)')
            plt.ylabel('Total Sensitivity')
            plt.title('{} yr'.format(yr))
            plt.legend()
        # change default font size
        plt.rcParams.update({'font.size': 20})
        
        plt.savefig(sens_dtc_dir + 'sens_dtc.png', bbox_inches='tight')
        print('Done')



