# Using Starduster to gen mags
# the first edition used to gen mags, the luminosity functions exceed the observations heavily.
import sys

sys.path.append('/home/zltan/.local/lib/python3.6/site-packages')
import starduster
import torch
import sedpy
import numpy as np
import pandas as pd
import time
import h5py
# import psutil
import os
from scipy.integrate import quad


# from scipy.optimize import minimize


def z2r(z):
    func2 = lambda y: 1 / (O_M * ((1 + y) ** 3) + O_K * ((1 + y) ** 2) + O_L) ** 0.5
    v2 = quad(func2, 0, z)
    r_c = c * v2[0] / 1e5  # comoving distance in Mpc/h
    return r_c


def z2t(z):
    func1 = lambda x: 1 / ((1 + x) * (O_M * ((1 + x) ** 3) + O_K * ((1 + x) ** 2) + O_L) ** 0.5)
    v1 = quad(func1, 0, z)
    t_L = t_H * v1[0]  # look back time in Gyr/h
    return t_L


print('Program start:', time.ctime())
# pid = os.getpid()
# process = psutil.Process(pid)

# parallel computing by galaxy bins
ivol = 3  ###

# load filters
csst_names = ['csst_nuv', 'csst_u', 'csst_g', 'csst_r', 'csst_i', 'csst_z', 'csst_y']  # 0~6
sdss_names = ['sdss_u0', 'sdss_g0', 'sdss_r0', 'sdss_i0', 'sdss_z0', 'wise_w1', 'wise_w2']  # 7~13
desi_names = ['DECaLS_g', 'DECaLS_r', 'DECaLS_z', 'BASS_g', 'BASS_r', 'MzLS_z_corrected']  # 14~19
# hsc_names = ['HSC_g_mean', 'HSC_r2_mean', 'HSC_i2_mean', 'HSC_z_mean', 'HSC_y_mean']  # 18~22
filters = sedpy.observate.load_filters(csst_names, directory="/home/zltan/CSST/filters/")
filters += sedpy.observate.load_filters(sdss_names)
filters += sedpy.observate.load_filters(desi_names, directory="/home/zltan/CSST/filters/")
# filters += sedpy.observate.load_filters(hsc_names, directory="/home/zltan/CSST/filters/")
output_list = []
for j in ['_nodust', '_dust']:
    for i in csst_names + sdss_names + desi_names:  ### + hsc_names:
        output_list.append(i + j)
data_types = ['<f8' for _ in range(len(output_list))]
dtypes = [(output_list[i], data_types[i]) for i in range(len(data_types))]

# load redshift table
redshift_path = '/home/zltan/9t/redshift.csv'
redshifts = pd.read_table(redshift_path, header=0)
zcp = np.array(redshifts['z'])  # 从大到小
# zcp[69] = zcp[68]
# zcp[76] = zcp[75]

# setting paras
O_M = 0.3111
O_L = 1 - O_M
O_K = 0
t_H = 9785641806 * 1e-9  # Hubble time in Gyr/h 要根据模拟输出调整
c = 299792458
pc = 3.08568025e16
h = 0.6766

datadir = "/home/cossim/Jiutian/M1000/GAEA-SAM/%03d/Gal_%03d_%03d.hdf5"
sfh_path = '/home/cossim/Jiutian/M1000/lightcones/gal-tab/SFH_mag/%03d/sfh%d_%d.npy'
subfile = '/home/cossim/Jiutian/M1000/lightcones/gal-tab/SFH_mag/%03d/subfile%d_%d.npy'
ssn = 128
need_fields = ['NMetalsColdGas', 'StellarDiskRadius', 'GasDiskRadius', 'StellarMass', 'BulgeMass', 'BulgeSize',
               'NMetalsStellarMass', 'NMetalsBulgeMass']
finished = np.load('mag_finished.npy')
not_finished = np.load('fat_not_finished.npy')
thissub = np.array([], dtype='int')
for i in range(ivol, len(not_finished), 7):
    thissub = np.append(thissub, not_finished[i])
print('THIS BIN:', thissub)
normal_max = np.loadtxt('/home/cossim/Jiutian/M1000/GAEA-SAM/M1000_subvolume/subvolume_%03d' % 10, dtype=np.int32)[0]
for isubvol in thissub:
    if isubvol in finished:
        continue
    normal_now = np.loadtxt('/home/cossim/Jiutian/M1000/GAEA-SAM/M1000_subvolume/subvolume_%03d' % isubvol, dtype=np.int32)[0]
    # if normal_now > normal_max:  # if larger, memory killed
    #     continue
    print('subvolume now:', isubvol)

    subvol_st = np.load('/home/zltan/9t/galaxy/SFH/subvol_st.npy')[isubvol]
    for isnap in range(subvol_st + 1, ssn):  ###
        if isnap == 69 or isnap == 76:  # cannot equals to 69 or 76
            continue
        save_path = '/home/cossim/Jiutian/M1000/lightcones/gal-tab/SFH_mag/%03d/mag%d_%d.npy' % (isnap, isnap, isubvol)
        if os.path.exists(save_path):  # already generated
            continue
        redshift = zcp[isnap]
        if redshift < 1e-5:
            distmod = 0
        else:
            distmod = 25 + 5 * np.log10(z2r(redshift) * (1 + redshift) / h)
        print(isnap, redshift, distmod)

        if not os.path.exists(datadir % (isnap, isnap, isubvol)):
            continue
        snapcut = h5py.File(datadir % (isnap, isnap, isubvol))['Galaxies'][:][need_fields]
        result = np.load(sfh_path % (isnap, isnap, isubvol))
        if len(result['GalID']) < 2:  # only one galaxy in sub file, jump this
            continue
        rstate = np.random.RandomState(111)
        n_gal = len(snapcut)
        theta = np.rad2deg(np.arccos(rstate.uniform(0, 1, n_gal)))

        # parameters to Starduster
        # Stellar age bins [yr]
        age_bins = []
        for i in range(isnap, subvol_st - 2, -1):
            # if i == 69 or i == 76:
            #     continue
            age_bins.append(z2t(zcp[i]) * 1e9 / h)
        age_bins = np.array(age_bins)
        age_bins -= age_bins[0]
        age_bins[0] = 0

        # Dust mass [M_sol]
        m_dust = 0.33 * (np.sum(snapcut['NMetalsColdGas'], axis=1)) * 1e10 / h  # before: metal bulgemass
        # Stellar disk radius [kpc]
        r_disk = snapcut['StellarDiskRadius'] * 1e3 / 3 / h
        # Dust disk radius [kpc]
        r_dust = snapcut['GasDiskRadius'] * 1e3 / 3 / h
        # Bulge radius [kpc]
        r_bulge = snapcut['BulgeSize'] * 1e3 / h

        # Mass in each stellar age bin [M_sol]
        diskmass1 = (snapcut['StellarMass'] - snapcut['BulgeMass']) * 1e10 / h
        sfh_mass_disk1 = np.array(result['StellarDisk'] * 1e10 / h)  # [:, :isnap+1]
        sfh_mass_bulge1 = np.array(result['StellarBulge'] * 1e10 / h)  # [:, :isnap+1]
        # Metal mass in each stellar age bin [M_sol]
        metal_diskmass1 = (np.sum(snapcut['NMetalsStellarMass'], axis=1) - np.sum(snapcut['NMetalsBulgeMass'],
                                                                                  axis=1)) * 1e10 / h
        sfh_metal_mass_disk1 = np.array(result['Metal'] - result['Metal_bulge']) * 1e10 / h  # [:, :isnap+1]
        sfh_metal_mass_bulge1 = np.array(result['Metal_bulge'] * 1e10 / h)  # [:, :isnap+1]

        sed_model = starduster.MultiwavelengthSED.from_builtin()
        converter = starduster.SemiAnalyticConventer(sed_model, age_bins)
        gp, sfh_disk, sfh_bulge = converter(
            theta=theta,
            m_dust=np.zeros_like(m_dust),
            r_dust=r_dust,
            r_disk=r_disk,
            r_bulge=r_bulge,
            sfh_mass_disk=sfh_mass_disk1,
            sfh_metal_mass_disk=sfh_metal_mass_disk1,
            sfh_mass_bulge=sfh_mass_bulge1,
            sfh_metal_mass_bulge=sfh_metal_mass_bulge1
        )
        sed_model.configure(filters=filters, redshift=redshift, distmod=distmod, ab_mag=True)
        with torch.no_grad():
            # generate sed without dust
            mags_nodust = sed_model(gp, sfh_disk, sfh_bulge, return_ph=True, component='dust_free')
            # generate sed with dust
            mags_dust = sed_model(gp, sfh_disk, sfh_bulge, return_ph=True)

        # compile data
        emptyarr = np.zeros((len(snapcut),), dtype=dtypes)
        dtypenames = emptyarr.dtype.names
        mags_nodust = mags_nodust.numpy()
        mags_dust = mags_dust.numpy()
        len_filter = len(filters)
        for i2 in range(len_filter):
            emptyarr[dtypenames[i2]] = mags_nodust[:, i2]
        for i3 in range(len_filter, 2 * len_filter):
            emptyarr[dtypenames[i3]] = mags_dust[:, i3 - len_filter]


        np.save(save_path, emptyarr)
        print('mag save done, ', isnap, 'Len: ', len(emptyarr), 'len gal:', len(snapcut), time.ctime())
        # Print the memory usage
        # memory_usage = round(process.memory_info().rss / 1024**3, 2)
        # print("Memory usage:", memory_usage, "Gbytes")
    print('%d,%d: output_docu=' % (isnap, isubvol), save_path)
