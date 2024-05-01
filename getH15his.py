import numpy as np
import readgal_fun as readgal
import time
import pandas as pd


def snap_to_age(snap, finalsnap):
    return tt[snap] - tt[finalsnap]


def append_sfr(history, nextp, finalsnap=63):
    dt = tt[nextp.snapnum - 1] - tt[nextp.snapnum]
    df = pd.DataFrame(
        {'Mstar': nextp.sfr * dt * 0.1,  # total mass of new formed stars, follow chabrier IMF; 10^10 [Msun];
         'Mstar_bulge': nextp.SfrBulge * dt * 0.1,
         'Mstar_disk': (nextp.sfr - nextp.SfrBulge) * dt * 0.1,
         'metallicity': np.sum(nextp.MetalsStellarMass),  # all new formed stars have uniform metallicity;
         'metal_budge': np.sum(nextp.MetalsBulgeMass),
         'metal_disk': np.sum(nextp.MetalsStellarMass) - np.sum(nextp.MetalsBulgeMass),
         'snap': nextp.snapnum,
         'age': snap_to_age(nextp.snapnum, finalsnap)}, index=[0])  # age of the stars
    history = pd.concat([history, df], ignore_index=True)
    return history


def tracedesc(glist, p, finalsnap=63):
    d = -1
    if (p.snapnum < finalsnap):
        for i in range(p.snapnum, finalsnap):
            aa = np.argwhere(glist.galaxyID == p.descendantId).squeeze()
            if aa is None:
                d = -1
                break
            else:
                p = glist[aa]
            if (p.snapnum == finalsnap):
                d = aa
                break
    if (d == -1):
        return None
    else:
        return glist[d]


def get_History(gid):  # gid is the galaxyID of the target galaxy
    allprogs = []
    history = pd.DataFrame(columns=['Mstar', 'metallicity', 'age'])
    ig = 0
    flag = 1
    if (gtree[ig].galaxyID != gid):
        ig = np.argwhere(gtree.galaxyID == gid).squeeze()
        if (ig.size == 0):
            print('can not find the galaxy %d', gid)
            flag = 0
    if flag:
        g = gtree[ig]
        # print(g)
        Nsnap = g.snapnum
        allprogs.append(g)
        glist = gtree[(gtree.galaxyID >= gid) & (gtree.galaxyID <= g.lastProgenitorId)]
        for g in glist:
            desc = tracedesc(glist, g, finalsnap=Nsnap)
            if (desc is None):
                continue
            elif (desc.galaxyID == gid):
                allprogs.append(g)
        for g in allprogs:
            history = append_sfr(history, g, Nsnap)
    return history


def compile_arrays(input_array):
    max_len = max(len(arr) for arr in input_array)

    # Create a structured array with the maximum length
    result_dtype = [(name, _) for name, _ in input_array[0].dtype.descr]
    result = np.zeros((len(input_array), max_len), dtype=result_dtype)

    for i, arr in enumerate(input_array):
        # print(int(arr['snap'][0]), len(arr))
        result[i, 0:len(arr)] = arr

    return result


Galtype = [
        ('galaxyID',    np.int64),
        ('haloID',    np.int64),
        ('firstProgenitorId',    np.int64),
        ('nextProgenitorId',    np.int64),
        ('lastProgenitorId',    np.int64),
        ('fofCentralId',    np.int64),
        ('FileTreeNr',    np.int64),
        ('descendantId',    np.int64),
        ('mainLeafID',    np.int64),
        ('treeRootID',    np.int64),
        ('SubID',    np.int64),
        ('MMSubID',    np.int64),
        ('PeanoKey',    np.int32),
        ('redshift',    np.float32),
        ('type',    np.int32),
        ('snapnum',    np.int32),
        ('LookBackTimeToSnap',    np.float32),
        ('CentralMvir',    np.float32),
        ('CentralRvir',    np.float32),
        ('DistanceToCentralGal',    (np.float32, 3)),
        ('Pos',    (np.float32, 3)),
        ('Vel',    (np.float32, 3)),
        ('Len',    np.int32),
        ('mvir',    np.float32),
        ('rvir',    np.float32),
        ('vvir',    np.float32),
        ('vmax',    np.float32),
        ('GasSpin',    (np.float32, 3)),
        ('StellarSpin',    (np.float32, 3)),
        ('InfallVmax',    np.float32),
        ('InfallVmaxPeak',    np.float32),
        ('InfallSnap',    np.int32),
        ('InfallHotGas',    np.float32),
        ('HotRadius',    np.float32),
        ('OriMergTime',    np.float32),
        ('MergTime',    np.float32),
        ('ColdGas',    np.float32),
        ('stellarMass',    np.float32),
        ('bulgeMass',    np.float32),
        ('DiskMass',    np.float32),
        ('HotGas',    np.float32),
        ('EjectedMass',    np.float32),
        ('blackHoleMass',    np.float32),
        ('ICM',    np.float32),
        ('MetalsColdGas',    np.float32),
        ('MetalsStellarMass',    np.float32),
        ('MetalsBulgeMass',    np.float32),
        ('MetalsDiskMass',    np.float32),
        ('MetalsHotGas',    np.float32),
        ('MetalsEjectedMass',    np.float32),
        ('MetalsICM',    np.float32),
        ('PrimordialAccretionRate',    np.float32),
        ('CoolingRadius',    np.float32),
        ('CoolingRate',    np.float32),
        ('CoolingRate_beforeAGN',    np.float32),
        ('QuasarAccretionRate',    np.float32),
        ('RadioAccretionRate',    np.float32),
        ('sfr',    np.float32),
        ('SfrBulge',    np.float32),
        ('XrayLum',    np.float32),
        ('BulgeSize',    np.float32),
        ('StellarDiskRadius',    np.float32),
        ('GasDiskRadius',    np.float32),
        ('CosInclination',    np.float32),
        ('disruptionOn',    np.int32),
        ('mergeOn',    np.int32),
        ('MagDust',    (np.float32, 20)), # see ./Filter_Names_my_new.txt
        ('Mag',    (np.float32, 20)),
        ('MagBulge',    (np.float32, 20)),
        ('MassWeightAge',    np.float32),
        ('rbandWeightAge',    np.float32),
        ('sfh_ibin',    np.int32),
        ('sfh_numbins',    np.int32),
        ('sfh_DiskMass',    (np.float32, 20)),
        ('sfh_BulgeMass',    (np.float32, 20)),
        ('sfh_ICM',    (np.float32, 20)),
        ('sfh_MetalsDiskMass',    (np.float32, 20)),
        ('sfh_MetalsBulgeMass',    (np.float32, 20)),
        ('sfh_MetalsICM',    (np.float32, 20))
]

Galtype=np.dtype(Galtype)


props= [
        ['MagDust', '[:,15:20]',5],
        ['Mag',     '[:,15:20]',5],
        ['MagBulge','[:,15:20]',5],
        'MassWeightAge',
        ]
# props= list(Galtype.names[:-13]) + props
props = ['galaxyID', 'firstProgenitorId', 'lastProgenitorId', 'descendantId', 'snapnum', 'ColdGas', 'stellarMass', 'bulgeMass', 'DiskMass', 'MetalsStellarMass', 'sfr', 'SfrBulge', 'MetalsBulgeMass', 'StellarDiskRadius', 'BulgeSize'] + props

# parallel computing by galaxy bins
for istart_read in range(13, 26):
    start_read = int(istart_read * 20)
    end_read = min(start_read+19, 511)
    print('Now read from %d to %d' % (start_read, end_read))

    gals=readgal.readgal_h15('/home/cossim/Millennium/Henriques15/SA_galtree_',start_read,end_read,Galtype, props=props)
    print(gals[:5],'\n\n',gals.dtype.names)

    isnap = 63
    # z0 snapnum63, z1 snapnum41, z2 snapnum32 2.07, z3 snapnum27 3.06
    masscut = 0.5  #  * 1e10 solar mass
    gals = np.sort(gals, order='snapnum')
    left = np.searchsorted(gals['snapnum'], isnap, side='left')
    right = np.searchsorted(gals['snapnum'], isnap, side='right')
    snapcut = gals[left:right]
    snapcut = snapcut[snapcut['stellarMass'] > masscut]
    print(len(snapcut))
    print(snapcut[:5])


    redshift_time_dir='/home/lzxie/SAM_data/MR_forCSST/redshift_time_mil.txt'
    aa=np.loadtxt(redshift_time_dir)
    global tt
    tt=aa[:,1]

    # 10 subfiles need 1 hour to run, using 72 cores
    # counts: 67500
    from multiprocessing.pool import Pool
    # from multiprocessing import get_context

    # ctx = get_context('spawn')
    gtree = gals

    def parallel_get_history(galid):
        # firstly check gtree and gals read by same files！
        his = get_History(galid)
        return his.groupby('age').sum().reset_index().to_records(index=None)

    process = 72
    # pool = Pool(context=ctx)
    pool = Pool(processes=process)
    print('pool star:', time.ctime())
    res = pool.map(parallel_get_history, snapcut.galaxyID)
    pool.close()
    pool.join()
    print('pool end:', time.ctime())

    max_len = max(len(arr) for arr in res)
    for i in range(len(res)):
        if len(res[i]['age']) == max_len:
            age = res[i]['age']
            break
    age = np.append(age, snap_to_age(isnap-len(age),isnap))
    age = age*1e9
    print(age, len(age))

    result = compile_arrays(res)
    print(len(result))
    np.save('./rerunH15/result_%d.npy' % start_read, result)
    np.save('./rerunH15/age_%d.npy' % start_read, age)
    print('Success genereated!')
    print('output_docu path=' + './rerunH15/result_%d.npy' % start_read)
