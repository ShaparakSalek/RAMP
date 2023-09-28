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
from ipywidgets import interact, IntSlider, HBox, IntText, BoundedIntText
import ipywidgets as widgets
mpl.rcParams.update({'font.size': 20})
DEBUG = False
from seis_design_optimization import MonitoringDesignSensitivity2D,create_cmap
# -------------------------------


# ----------------------------------------
if __name__ == "__main__":
    nz = 335
    nx = 1467
    dz = dx = 7.5  # m
    ns = 146
    nr = 1467
    dr = 7.5     # Receiver interval (m)
    ds = 75.0    # source interval (m)
    t1 = 78      # the year (int) when fault leakage begins
    years = [str(y) for y in [80, 85, 90, 95, 100, 125]]
    senMax = [1e7, 1e7, 1e7, 1e9]
    thresholds = [0.2, 0.6]
    ks = 4; kr = 4

    datadir = '/Users/tian7/data/NRAP/data_use-case2/'
    outpre='./'
    seis = MonitoringDesignSensitivity2D(nx,nz,dx,dz,ns,nr,ds,dr,years,t1,thresholds,ks,kr)
    # plot velocity, density and plume models (images)
    modeldir = datadir + 'models_plume_mask/'

    out_dir = outpre + 'model_plume_images/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fnames = glob.glob(modeldir + '*.bin')
    for fname in fnames[:1]:
        seis.plot_model_image(fname, out_dir)