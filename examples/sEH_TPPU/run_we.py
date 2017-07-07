import multiprocessing as mulproc

import numpy as np
import simtk.openmm.app  as app

from wepy.WExplore import Walker_chr, run_walker, Calculate

print("HELLO")
print('Reading psf file ..')
psf = app.CharmmPsfFile('fixed_seh.psf')

print('Reading pdb file ..')
pdb = app.PDBFile('seh_tppu_mdtraj.pdb')

print('Reading ff parameters')
params = app.CharmmParameterSet('top_all36_cgenff.rtf', 'par_all36_cgenff.prm',
                                'top_all36_prot.rtf', 'par_all36_prot.prm',
                                'tppu.str', 'toppar_water_ions.str')
# set WE parameters
n_walkers = 3
n_workers = 2
n_cycles = 2
n_atoms = 5097

initial = True
manager = mulproc.Manager()
walkers = manager.list()

# initialize each Walker object
for i in range (n_walkers):
    new_walker = Walker_chr()
    new_walker.walker_id = i
    new_walker.weight = 1 / n_walkers
    new_walker.restart_point = None
    walkers.append(new_walker)

# merging cutoff
mergedist = 0.25 #nm ==  2.5 A

# Make list of Walkers
walker_weight = []

# main loop
for i in range(n_cycles):
    # a list of walker processes
    walkers_pool = [run_walker(params, psf.topology, i , initial)
                     for i in range(n_walkers)]

    # queu for putting processes to workers
    free_workers= mulproc.Queue()
    # lock for number of workers in the work pool, this is assumed to be global
    lock = mulproc.Semaphore(n_workers)

    # putting core indices into the worker queue
    for i in range(n_workers):
        free_workers.put(i)

    # start all walkers
    for p in walkers_pool:
        p.start()

    # wait for processes to end
    for p in walkers_pool:
        p.join()

    # DEBUG
    for w in walkers:
        print ('Result ID= {} and Weight = {}\n'.format(w.walker_id ,w.weight))

    # no longer the first segments
    initial = False

    # all to all distance matrix
    a2a = np.zeros((n_walkers,n_walkers))

    # Calculating a2a Distance Matrix
    for i in range(n_walkers):
        walker_weight.append(walkers[i].weight)
        for j in range (i+1, n_walkers):
            Cal = Calculate()
            a2a[i][j] = Cal.Calculate_Rmsd(walkers[i].positions, walkers[j].positions)


    print(a2a)

    # merge and clone!
    mcf= mergeclone.decision_maker(a2a, walkerwt, n_walkers, mergedist)
    mcf.mergeclone()
    #for i in range(n_walkers):
     #   print (' WalkerId ={ }  Weight = {}  amp= {}  parent= {} '.format( i, mcf.walkerwt[i], mcf.amp[i] , mcf#.copy_struct[i]))

#   print (a2a)
