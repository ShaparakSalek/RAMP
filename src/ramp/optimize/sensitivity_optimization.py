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
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
# from ipywidgets import interact, IntSlider, HBox, IntText, BoundedIntText
# import ipywidgets as widgets

import argparse
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.data_readers.read_input_from_files import read_yaml_parameters,read_sens_from_segy
from utilities.write_output_to_files import convert_matrix_to_segy,write_optimal_design_to_yaml
mpl.rcParams.update({'font.size': 20})
DEBUG = False



# -------------------------------

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
    def __init__(self,yaml_path):
        """
        Initializes seismic parameters from a YAML file.

        Parameters:
            yaml_path (str): Path to the input YAML file.

        """

        parameters = read_yaml_parameters(yaml_path)
        # read in parameters from YAML file
        self.nx = parameters['nx']
        self.nz = parameters['nz']
        self.dz = parameters['dz']
        self.dx = parameters['dx']
        self.nsrc = parameters['ns']
        self.nrec = parameters['nr']
        self.drec = parameters['dr']
        self.dsrc = parameters['ds']
        self.t1 = parameters['t1']
        years0 = parameters['years']
        self.senMax = parameters['senMax']
        self.thresholds = parameters['thresholds']
        self.opt_src_sep = parameters['ks']
        self.opt_rec_sep = parameters['kr']
        self.sen_nor=parameters['sen_nor']
        self.wavefield = parameters['wavefield']
        self.vpvs = parameters['vpvs']
        self.units= parameters['units']
        self.datadir = parameters['datadir']
        self.outpre = parameters['outpre']
        self.timestamps = [str(y) for y in years0]
        self.dtc_flag=parameters['dtc_flag']
        self.output_yaml=parameters['output_yaml']
        self.target_dtc=parameters['target_dtc']
        self.segy_read=parameters['segy_read']
        # create a sensitivity object

        # plot velocity, density and plume models (images)
        self.modeldir = self.datadir + 'models_plume_mask/'
    
        self.plume_images_dir = self.outpre + 'model_plume_images/'

        # plot sensitivity images per component
        self.sensitivity_images_out_dir = self.outpre + '/sensitivity_images/'
        if not os.path.exists(self.sensitivity_images_out_dir):
            os.makedirs(self.sensitivity_images_out_dir)

        self.optimal_dir = self.outpre + '/optimal_design/'
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

    def load_sensitivity_data(self, datadir, yr, wf, vpvs):
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
        if self.sen_nor:
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
        if self.sen_nor==1:
            img = ax.imshow(sens,extent=extent,vmin=0.0,vmax=1.0,cmap=cmap)
        else:
            img = ax.imshow(sens,vmin=0.0,vmax=np.max(sens),extent=extent,cmap=cmap)

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
        if self.sen_nor==1:
            CS = plt.contour(X,Y,sens,self.thresholds,vmin=0.0,vmax=1.0,linestyles='solid',
                        linewidths=[0.5, 1.5], colors=['brown', 'm'])
        else:
            CS = plt.contour(X,Y,sens,np.array(self.thresholds)*np.max(sens),vmin=0.0,vmax=np.max(sens),linestyles='solid',
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
    
    def find_optimal_seismic_arrays(self,sens,threshold):
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
        if self.sen_nor==1:
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
    
    
    def find_design_from_dtc(self,sens2d):
        '''
        Find optimal design given a target data to collect

        Parameters
        ----------
        sens2d: ndarray(dtype=float, ndim=2)
            an array of sensitivity data in a shape (nSources, nReceivers)

        Returns
        -------
        design: list[float]
            Seismic source coordinate and then receiver coordinates

        '''
        arae_all=[]
        sen_all=[]
        th_all=1/np.arange(1,400)
        for th in th_all:
            design, area,sen_sel = self.find_optimal_seismic_arrays(sens2d,th)
            arae_all.append(area)
            sen_all.append(sen_sel)
        arae_all=np.array(arae_all)
        sen_all=np.array(sen_all)
        # given one dtc, find the design and th
        #print(sen_all)
        design_idx = np.argmin(np.abs(arae_all-self.target_dtc))
        th=th_all[design_idx]
        #print(th)
        design, area,sen_sel = self.find_optimal_seismic_arrays(sens2d,th)
        return design

    def plot_sens_to_percent(self):
        '''
        function to plot sensitivity vs data to collect percentage
        '''
        if self.dtc_flag==0:
            return
        sens_dtc_dir = self.outpre + '/sens_dtc/'
        if not os.path.exists(sens_dtc_dir):
            os.makedirs(sens_dtc_dir)
        
        plt.figure(figsize=(20,15))
        for yr in self.timestamps[:]:
            plt.subplot(2,3,self.timestamps.index(yr)+1)
            for wf in self.wavefield:
                for ps in self.vpvs:
                    if self.segy_read==1:
                        segy_path=self.datadir+wf + '_' + ps + '_' + yr + '.sgy'
                        sens2d = read_sens_from_segy(segy_path,self.sen_nor)
                    else:
                        sens2d = self.load_sensitivity_data(self.datadir, yr, wf, ps)
                    arae_all=[]
                    sen_all=[]
                    for th in 1/np.arange(1,400):
                        design, area,sen_sel = self.find_optimal_seismic_arrays(sens2d,th)
                        arae_all.append(area)
                        sen_all.append(sen_sel)
                    if self.sen_nor==1:
                        plt.plot(arae_all,np.array(sen_all)/1250,'o-',label=wf+'_'+ps)                        
                    else:
                        plt.plot(arae_all,sen_all,'o-',label=wf+'_'+ps)
            plt.xlabel('data to collect (%)')
            if self.sen_nor==1:
                plt.ylim(0,1)
                plt.ylabel('Normalized Total Sensitivity (m/s)')
            else:
                plt.ylabel('Total Sensitivity (m/s)')
            year=int(yr)-78
            plt.title('t1+{} yr'.format(year))
            plt.legend()
        # change default font size
        plt.rcParams.update({'font.size': 20})
        
        plt.savefig(sens_dtc_dir + 'sens_dtc.png', bbox_inches='tight')
        print('Done')

    def plot_and_find_opt_arrays(self):
        # plot sensitivity images per component
        out_dir = self.outpre + '/sensitivity_images/'
        if not os.path.exists(out_dir):
            os.makedirs(self.out_dir)

        optimal_dir = self.outpre + '/optimal_design/'
        if not os.path.exists(optimal_dir):
            os.makedirs(optimal_dir)

        area = [0.0, 0.0]
        for wf in self.wavefield:
            for ps in self.vpvs:
                if self.segy_read==1:
                    segy_path=self.datadir+wf + '_' + ps + '_' + self.timestamps[-1] + '.sgy'
                    sens2d = read_sens_from_segy(segy_path,self.sen_nor)
                else:
                    sens2d = self.load_sensitivity_data(self.datadir, self.timestamps[-1], wf, ps)
                
                design1, _,sen_sel = self.find_optimal_seismic_arrays(sens2d, self.thresholds[1])
                ns_max = self.find_max_num_sources(design1)
                for yr in self.timestamps:
                    print('Optimal design', yr+'y', wf, ps)
                    if self.segy_read==1:
                        segy_path=self.datadir+wf + '_' + ps + '_' + yr + '.sgy'
                        sens2d = read_sens_from_segy(segy_path,self.sen_nor)
                    else:
                        sens2d = self.load_sensitivity_data(self.datadir, yr, wf, ps)
                    _, area[0],sen_sel = self.find_optimal_seismic_arrays(sens2d,self.thresholds[0])
                    design1, area[1],sen_sel = self.find_optimal_seismic_arrays(sens2d,self.thresholds[1])
                    self.plot_sensitivity_image(sens2d, out_dir,yr,wf,ps, area)
                    design2 = self.plot_optimal_design(design1,self.modeldir,optimal_dir,wf,ps,yr,ns_max)
                    if self.output_yaml==1:
                        fname = optimal_dir + wf + '_' + ps + '_' + yr + '.yaml'
                        write_optimal_design_to_yaml(design2,fname)
                    else:
                        fname = optimal_dir + wf + '_' + ps + '_' + yr + '.txt'
                        with open(fname, 'w') as fout:
                            for line in design2:
                                fout.write(str(line) + '\n')

    def plot_and_find_opt_arrays_from_dtc(self):
        '''
        given a value of data to collect, find and plot the optimal seismic design
        '''
        optimal_dir = self.outpre + '/optimal_design_from_'+str(self.target_dtc)+'%_of_data/'
        
        if not os.path.exists(optimal_dir):
            os.makedirs(optimal_dir)
        for wf in self.wavefield:
            for ps in self.vpvs:
                if self.segy_read==1:
                    segy_path=self.datadir+wf + '_' + ps + '_' + self.timestamps[-1] + '.sgy'
                    sens2d = read_sens_from_segy(segy_path,self.sen_nor)
                else:
                    sens2d = self.load_sensitivity_data(self.datadir, self.timestamps[-1], wf, ps)
                design1=self.find_design_from_dtc(sens2d)
                ns_max = self.find_max_num_sources(design1)
                for yr in self.timestamps:
                    print('Optimal design from dtc', yr+'y', wf, ps)
                    if self.segy_read==1:
                        segy_path=self.datadir+wf + '_' + ps + '_' + yr + '.sgy'
                        sens2d = read_sens_from_segy(segy_path,self.sen_nor)
                    else:
                        sens2d = self.load_sensitivity_data(self.datadir, yr, wf, ps)
                    design1=self.find_design_from_dtc(sens2d)
                    #seis.plot_sensitivity_image(sens2d, out_dir,yr,wf,ps, area)
                    design2 = self.plot_optimal_design(design1,self.modeldir,optimal_dir,wf,ps,yr,ns_max)
                    if self.output_yaml==1:
                        fname = optimal_dir + wf + '_' + ps + '_' + yr + '.yaml'
                        write_optimal_design_to_yaml(design2,fname)
                    else:
                        fname = optimal_dir + wf + '_' + ps + '_' + yr + '.txt'
                        with open(fname, 'w') as fout:
                            for line in design2:
                                fout.write(str(line) + '\n')
    
                         


# ----------------------------------------
if __name__ == "__main__":
    # Command-line argument parsing to get the YAML configuration file
    parser = argparse.ArgumentParser(description='Seismic Design Optimization Script')
    parser.add_argument('--config', type=str, default='seis_sens_opt_params.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    yaml_path = args.config
    # create a sensitivity object
    seis = MonitoringDesignSensitivity2D(yaml_path)

    seis.plot_model_image()

    seis.plot_and_find_opt_arrays()
    
    seis.plot_and_find_opt_arrays_from_dtc()
    
    seis.plot_sens_to_percent()



