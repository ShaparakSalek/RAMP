#
# This code shows how HDF5 files are created. 
# Kimbrelina 1.2 data set
# The NUFT simulation data in ASCII format is found on EDX:
# https://edx.netl.doe.gov/dataset/llnl-kimberlina-1-2-nuft-simulations-june-2018-v2

# This data set was created for testing DREAM in 3/2022, Xianjin Yang, LLNL
# The initial data set consists of P, CO2_saturation and gravity forward modeling results
# Added TDS data, 10/4/2023
# 3D pressure, co2 saturation and tds data. 
# Gravity data on a 2D regular grid on the ground surface

# References
# -----------
# Buscheck, T. A., K. Mansoor, X. Yang, H. M. Wainwright, and S. A. Carroll, 2019, Downhole pressure and chemical monitoring for CO2 and brine leak detection in aquifers above a CO2 storage reservoir: International Journal of Greenhouse Gas Control, 91, 102812.

# Yang, X., T. A. Buscheck, K. Mansoor, Z. Wang, K. Gao, L. Huang, D. Appriou, and S. A. Carroll, 2019, Assessment of geophysical monitoring methods for detection of brine and CO2 leakage in drinking water aquifers: International Journal of Greenhouse Gas Control, 90, 102803.


import numpy as np
import h5py, sys, os, glob
import multiprocessing

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

# ---------------------------------
def read_hdf5_file(h5file):
    '''Example hdf5 reader: read and parse one hdf5 or h5 file 
       attrname, attrval = hdf[ds].attrs.items()'''

    print(h5file)
    hdf = h5py.File(h5file, 'r')

    print('\n --- HDF groups')
    groups = get_groups('/', hdf)
    groups.sort()
    for g in groups: print(g)

    print('\n --- HDF datasets')
    for t in range(nt):
        datasets = get_datasets(f'/t{t}/', hdf)
        datasets.sort()
        for ds in datasets:
            data = hdf[ds]
            print(ds, data.shape, np.min(data), np.max(data))

    hdf.close()


# ---------------------------------
def read_data(sdir, sim, yr):
    '''Read pressure, co2sat and gravity data'''
    if '/co2/' in sdir:
        fname = sdir + 'sim%04d.%03dy.co2.npy' %(sim, yr)
        if os.path.exists(fname):
            return np.load(fname)
        else:
            print(fname)
            return no_co2_P_tds

    elif '/tds/' in sdir:
        fname = sdir + 'sim%04d.%03dy.tds.npy' %(sim, yr)
        if os.path.exists(fname):
            return np.load(fname)
        else:
            print(fname)
            return no_co2_P_tds

    elif '/P/' in sdir:
        fname = sdir + 'sim%04d.%03dy.P.npy' %(sim, yr)
        if os.path.exists(fname):
            return np.load(fname)
        else:
            print(fname)
            return no_co2_P_tds

    elif '/grav_fwd/' in sdir:
        fname = sdir + 'sim%04d.%03dy.out' %(sim, yr)
        if not os.path.exists(fname): 
            print(fname)
            return no_grav
        # four column gravity data: x, y, gz, gy. Select gz
        grav = np.loadtxt(fname, skiprows=2, usecols=2)
        grav = grav.reshape(ngy, ngx)
        return grav

    else:
        print('Invalid dir!')
        return None


# ---------------------------------
def create_h5(iSim):
    hdf5 = h5py.File(h5dir + 'sim%04d.h5'%iSim,'w')
    for i in range(nt):
        # print('Sim%d %d years' %(iSim, years[i]))
        # t0 - t11 for 12 time points (years)
        g1=hdf5.create_group('t%i'%i)

        pres = read_data(rootdir+'P/', iSim, years[i])
        g1.create_dataset('pressure', data=pres, dtype='float32')

        satu = read_data(rootdir+'co2/', iSim, years[i])
        g1.create_dataset('saturation', data=satu, dtype='float32')

        tds = read_data(rootdir+'tds/', iSim, years[i])
        g1.create_dataset('tds', data=tds, dtype='float32')

        grav = read_data(rootdir+'grav_fwd/', iSim, years[i])   
        g1.create_dataset('gravity', data=grav, dtype='float32')

        g1['pressure'].attrs['unit'] = 'Pa'
        g1['saturation'].attrs['unit'] = '1'
        g1['gravity'].attrs['unit'] = 'uGal'
        g1['tds'].attrs['unit'] = 'ppm'

    porosity_fixed = 0.30

    g1=hdf5.create_group('data')
    g1.create_dataset('porosity', data=porosity_fixed*np.ones([nx,ny,nz]),dtype='float32')
    g1.create_dataset('steps',    data=np.array(range(nt)),dtype='float32')
    g1.create_dataset('times',    data=np.array(years),dtype='float32')
    g1.create_dataset('vertex-x', data=np.array(vx),dtype='float32')
    g1.create_dataset('vertex-y', data=np.array(vy),dtype='float32')
    g1.create_dataset('vertex-z', data=np.array(vz),dtype='float32')
    g1.create_dataset('x',        data=np.array(x),dtype='float32')
    g1.create_dataset('y',        data=np.array(y),dtype='float32')
    g1.create_dataset('z',        data=np.array(z),dtype='float32')
    g1.create_dataset('grav_x',   data=np.array(gx),dtype='float32')
    g1.create_dataset('grav_y',   data=np.array(gy),dtype='float32')

    g1['x'].attrs['units'] = 'm'
    g1['y'].attrs['units'] = 'm'
    g1['z'].attrs['units'] = 'm'
    g1['vertex-x'].attrs['units'] = 'm'
    g1['vertex-y'].attrs['units'] = 'm'
    g1['vertex-z'].attrs['units'] = 'm'
    g1['grav_x'].attrs['units'] = 'm'
    g1['grav_y'].attrs['units'] = 'm'

    g1['z'].attrs['postive'] = 'up'
    g1['vertex-z'].attrs['postive'] = 'up'

    g1=hdf5.create_group('statistics')
    g1.create_dataset('pressure',   data=np.array([ np.min(pres),np.mean(pres),np.max(pres) ]),dtype='float32')
    g1.create_dataset('saturation', data=np.array([ np.min(satu),np.mean(satu),np.max(satu) ]),dtype='float32')
    g1.create_dataset('tds',        data=np.array([ np.min(tds), np.mean(tds), np.max(tds) ]), dtype='float32')
    g1.create_dataset('gravity',    data=np.array([ np.min(grav),np.mean(grav),np.max(grav) ]),dtype='float32')

    hdf5.close()




  
