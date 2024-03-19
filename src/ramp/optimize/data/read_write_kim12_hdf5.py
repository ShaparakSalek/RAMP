'''
This script prepares data in HDF5 format for monitoring design in NRAP RAMP
It also provided a reader to load and parse an H5 file

Kimberlina 1.2 simulated CO2 and brine leakage data and geophysical monitoring data
Pressure, CO2 saturation, TDS, gravity and ERT data
EDX Drive: https://edx.netl.doe.gov/workspace/resources/nrap-task-4-monitoring
https://edx.netl.doe.gov/dataset/llnl-kimberlina-1-2-nuft-simulations-june-2018-v2

Created on 2024-01-25
@author: Xianjin Yang, LLNL (yang25@llnl.gov)

References
----------
Yang, X., T. A. Buscheck, K. Mansoor, Z. Wang, K. Gao, L. Huang, D. Appriou,
and S. A. Carroll (2019), Assessment of geophysical monitoring methods for
detection of brine and CO2 leakage in drinking water aquifers, Int J
Greenh Gas Con, 90, 102803.

Buscheck, T. A., K. Mansoor, X. Yang, H. M. Wainwright, and S. A. Carroll (2019),
Downhole pressure and chemical monitoring for CO2 and brine leak detection in
aquifers above a CO2 storage reservoir, Int J Greenh Gas Con, 91, 102812.

'''

import numpy as np
import h5py, sys, os, glob
import multiprocessing
DEBUG = True

# There are 23 incomplete simulations out of 1000. Some time steps are missing
# Parameters at incomplete simulatons are set to zeros

incomplete_simulations = ['0008', '0037', '0092', '0118', '0120', '0127', 
'0136', '0150', '0182', '0197', '0211', '0245', '0397', '0449', '0518', 
'0590', '0598', '0686', '0749', '0863', '0935', '0937', '0970']

nSimulations = 1000
years =     [5, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200]
years_ert = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200] 
nt = len(years)

ertdir = './yang25/nrap/sim_out/ert/'
simdir = './yang25/nrap/sim_out/'
h5dir = 'U:/NRAP/kim12_h5/'

if not os.path.exists(h5dir):
    os.makedirs(h5dir)

# LLNL NUFT models (P, CO2, TDS): npy numpy array shape
nx = 40
ny = 20
nz = 32
no_co2_P_tds = np.zeros((nx, ny, nz))

# bounds for region of interest
xmin,xmax = 4000,8000
ymin,ymax = 1500,3500
zmin,zmax = 0, 1410.80

# vertex: vx, vy and vz
# voxel center: x, y, z
vx = np.linspace(xmin, xmax, nx+1)
vy = np.linspace(ymin, ymax, ny+1)
x = np.linspace( (vx[0]+vx[1])/2.0, (vx[-2]+vx[-1])/2.0, nx)
y = np.linspace( (vy[0]+vy[1])/2.0, (vy[-2]+vy[-1])/2.0, ny)
z = np.array([2.5, 7.5, 34.4, 83.1, 131.9, 180.6, 229.4, 278.1, 326.9,
              375.6, 424.4, 473.1, 521.9, 570.5, 619.0, 667.5, 716.0, 
              764.5, 813.0, 861.5, 910.0, 958.5, 1007.0, 1055.5, 1104.0, 
              1152.5, 1201.0, 1248.5, 1295.0, 1341.5, 1376.3, 1399.6])
vz = np.zeros(nz+1)
for i in range(1, nz+1):
    vz[i] = vz[i-1] + (z[i-1] - vz[i-1]) * 2

# gravity survey grid (gx and gy) on the ground surface
ngx = 41
ngy = 21
no_grav = np.zeros((ngy, ngx))
gx = np.linspace(4000.0, 8000.0, ngx)
gy = np.linspace(1500.0, 3500.0, ngy)

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

# -----------------------
def read_ert_mesh_array():
    '''ERT mesh, electrode location, array schedule'''
    mesh_array_file = ertdir + 'ert_mesh_array.dat'
    with open(mesh_array_file) as fin:
        lines = fin.readlines()
    # nx, ny, nz = [int(x) for x in lines[0].split()]
    dx = [float(x) for x in lines[1].split()]
    dy = [float(x) for x in lines[2].split()]
    dz = [float(x) for x in lines[3].split()]
    y = [0]
    for a in dx:
        y.append(y[-1] + a)
    x = [0]
    for a in dy:
        x.append(x[-1] + a)
    z = [0]
    for a in dz:
        z.append(z[-1] + a)

    ix0, iy0 = [int(a) for a in lines[4].split()]
    xn = np.array(x) - x[ix0-1]
    yn = np.array(y) - y[iy0-1]
    zn = np.array(z)

    ne = int(lines[5])
    xe=np.zeros(ne)
    ye=np.zeros(ne)
    ze=np.zeros(ne)
    for i, line in enumerate(lines[6:6+ne]):
      tokens = line.split(',')
      xe[i] = xn[int(tokens[1])-1]
      ye[i] = yn[int(tokens[0])-1]

    xe += 3000   # Yang et al. 2019
    ye += 1500

    k = ne + 6
    ABMN = []   # 0-based electrode ID: 0 ~ ne-1
    for line in lines[k:]:
        tokens = line.split()
        if len(tokens) < 9: break
        abmn = [int(x)-1 for x in tokens[2::2]]
        ABMN.append(abmn)

    return xe, ye, ze, np.array(ABMN)

# ---------------------------------
def read_ert_data(sdir, sim, yr, dtype):
    '''Read ert data in /p/lustre2/yang25/k12hi/ghgt14/ert/fwd/
    example file name: sim0999.W31-0.2.100y.dat'''
    
    no_ert = np.zeros(10000)  # 5 2D lines, 2000/line

    fname = sdir + 'sim%04d.W31-0.2.%03dy.dat' %(sim, yr)
    if not os.path.exists(fname): 
        #print(fname)
        return no_ert

    if dtype == 'AppRes':
        ert = np.loadtxt(fname, skiprows=0, usecols=10)
    else:
        ert = np.loadtxt(fname, skiprows=0, usecols=9)

    return ert

# ---------------------------------
def read_sim_data(sdir, sim, yr):
    '''Read pressure, co2sat and gravity data'''
    if '/co2/' in sdir:
        fname = sdir + 'sim%04d.%03dy.co2.npy' %(sim, yr)
        if os.path.exists(fname):
            return np.load(fname)
        else:
            #print(fname)
            return no_co2_P_tds

    elif '/tds/' in sdir:
        fname = sdir + 'sim%04d.%03dy.tds.npy' %(sim, yr)
        if os.path.exists(fname):
            return np.load(fname)
        else:
            #print(fname)
            return no_co2_P_tds

    elif '/P/' in sdir:
        fname = sdir + 'sim%04d.%03dy.P.npy' %(sim, yr)
        if os.path.exists(fname):
            return np.load(fname)
        else:
            #print(fname)
            return no_co2_P_tds

    elif '/grav_fwd/' in sdir:
        fname = sdir + 'sim%04d.%03dy.out' %(sim, yr)
        if not os.path.exists(fname): 
            #print(fname)
            return no_grav
        # four column gravity data: x, y, gz, gy. Select gz
        grav = np.loadtxt(fname, skiprows=2, usecols=2)
        grav = grav.reshape(ngy, ngx)
        return grav

    else:
        print('Invalid dir!')
        return None

# ---------------------------------
def create_hdf5_file(iSim):
    xe, ye, ze, abmn = read_ert_mesh_array()
    porosity_fixed = 0.30
    hdf5 = h5py.File(h5dir + 'sim%04d.h5'%iSim,'w')
    for i in range(nt):
        if DEBUG: print('Sim%d %d years' %(iSim, years[i]))
        # t0 - t11 for 12 time points (years)
        g1=hdf5.create_group('t%i'%i)

        pres = read_sim_data(simdir+'P/', iSim, years[i])
        g1.create_dataset('pressure', data=pres, dtype='float32')
        g1['pressure'].attrs['unit'] = 'Pa'

        satu = read_sim_data(simdir+'co2/', iSim, years[i])
        g1.create_dataset('saturation', data=satu, dtype='float32')
        g1['saturation'].attrs['unit'] = '1'

        tds = read_sim_data(simdir+'tds/', iSim, years[i])
        g1.create_dataset('tds', data=tds, dtype='float32')
        g1['tds'].attrs['unit'] = 'ppm'

        grav = read_sim_data(simdir+'grav_fwd/', iSim, years[i])
        g1.create_dataset('gravity', data=grav, dtype='float32')
        g1['gravity'].attrs['unit'] = 'uGal'

        AppRes = read_ert_data(ertdir+'fwd/', iSim, years_ert[i], 'AppRes')
        g1.create_dataset('ert_appres', data=AppRes, dtype='float32')
        g1['ert_appres'].attrs['unit'] = 'Ohm-m'


    g1=hdf5.create_group('data')
    g1.create_dataset('porosity', data=porosity_fixed*np.ones([nx,ny,nz]),
                       dtype='float32')
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

    g1.create_dataset('times_ert', data=np.array(years_ert),dtype='float32')
    g1.create_dataset('ert_abmn',  data=abmn,dtype='int')
    g1.create_dataset('x_ert',     data=xe,dtype='float32')
    g1.create_dataset('y_ert',     data=ye,dtype='float32')
    g1.create_dataset('z_ert',     data=ze,dtype='float32')
    g1['x_ert'].attrs['units'] = 'm'
    g1['y_ert'].attrs['units'] = 'm'
    g1['z_ert'].attrs['units'] = 'm'
 
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

    g1=hdf5.create_group('statistics')  # this is not needed
    g1.create_dataset('pressure',   
        data=np.array([ np.min(pres),np.mean(pres),np.max(pres)]), dtype='float32')
    g1.create_dataset('saturation', 
        data=np.array([ np.min(satu),np.mean(satu),np.max(satu)]),dtype='float32')
    g1.create_dataset('tds',        
        data=np.array([ np.min(tds), np.mean(tds), np.max(tds)]),dtype='float32')
    g1.create_dataset('gravity',    
        data=np.array([ np.min(grav),np.mean(grav),np.max(grav)]),dtype='float32')
    g1.create_dataset('ert_appres',    
        data=np.array([ np.min(AppRes),np.mean(AppRes),np.max(AppRes)]),dtype='float32')

    hdf5.close()

# ---------------------------------
def read_hdf5_file(h5file):
    '''Example hdf5 reader: read and parse one hdf5 or h5 file
       attrname, attrval = hdf[ds].attrs.items()'''

    print('\n\n *** ' + h5file + ' ***')
    hdf5_data = h5py.File(h5file, 'r')

    print('\n --- HDF groups')
    groups = get_groups('/', hdf5_data)
    groups.sort()
    for g in groups: print(g)

    print('\n --- HDF datasets')
    for t in range(nt):  # nt is defined at the top
        datasets = get_datasets(f'/t{t}/', hdf5_data)
        datasets.sort()
        for ds in datasets:
            data = np.array(hdf5_data[ds])
            print(ds, data.shape, np.min(data), np.max(data))

    hdf5_data.close()

# ---------------------------------
if __name__ == "__main__":

    create_h5 = False
    if create_h5:
        #Serial processing
        for i in range(1, nSimulations+1):
            create_hdf5_file(i)

        # Parallel processing
        # pool = multiprocessing.Pool()
        # pool.map(create_hdf5_file, range(1, nSimulations+1))

    # read and parse an H5 file
    read_h5 = True
    if read_h5:
        h5files = glob.glob(h5dir + '*.h5')
        h5files.sort()
        # inspect three simulations
        for i in [0, 97, 999]:
            read_hdf5_file(h5files[i])


  
