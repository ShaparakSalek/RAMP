# -*- coding: utf-8 -*-
"""
Created on 2023-09-22
@author: Xianjin Yang, LLNL (yang25@llnl.gov), Yuan Tian, LLNL (tian7@llnl.gov)
"""
from ipywidgets import interact, IntSlider, HBox, IntText, BoundedIntText
import ipywidgets as widgets


def interactive_plot_optimal_design(source_idx, time_idx):
    '''
    Plot and save optimal design with source, receivers and plume per time step
    and per sensitivity component

    Parameters
    ----------
    source_idx: int
        an index to sources0 list
        
    '''

    yr = timestamps[time_idx]
    wf = 'P'; ps = 'Vp'

    # create CO2 plume image
    plume_file = model_dir + 'mask_ol_y' + yr + '_leakage.bin'
    model = read_seismic_model(plume_file)
    fig, ax = plt.subplots(figsize=(12, 4))
    xMin = 0; xMax = nx * dx
    cmap = create_cmap()
    ax.imshow(model.T, extent=[0, nx*dx, nz*dz, 0], cmap=cmap)

    # Plot source and receivers
    sens2d = load_sensitivity_data(yr, 'P', 'Vp')
    # optimal_design and sources0 are global variables used in the function above
    optimal_design = find_optimal_seismic_arrays(sens2d,sensitivity_threshold[1])
    #optimal_design = find_optimal_seismic_arrays(sens2d,0.4)
    num_source_max, sources0 = find_optimal_sources(optimal_design)
    print('Optimal sources: ', sources0)
    if source_idx >= num_source_max: return

    s = np.int32(optimal_design[source_idx,:])
    y0 = -50
    xs = sources0[source_idx] * source_interval * scale_src_interval
    xr = np.float32(np.nonzero(s)) * receiver_interval * scale_rec_interval
    xr = xr[0]  # xr above is a 2D array

    plt.plot([xMin, xMax], [y0, y0], 'k-', lw=0.3)
    plt.plot([xs], [y0], 'r*', ms=12)
    plt.plot(xr, [y0]*len(xr), 'bo', ms=3)
    plt.xlim(-100, xMax)
    ax.text(200, y0+300, 'Sensitivity_threshold=' + str(sensitivity_threshold[1]), color='k')
    ax.set_yticks((0, 1000, 2000))
    ax.set_xlabel('Horizontal Distance (m)')
    ax.set_ylabel('Depth (m)')

    yrs = str(int(yr) - t1)
    annotate = 'CO$_2$ plume mask at t1+' + yrs + ' years'
    ax.text(200, 500, annotate, color='black')
    ax.grid(lw=0.3, alpha=0.7)
    ax.set_aspect('auto')
    plt.tight_layout()

