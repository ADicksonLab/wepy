import numpy as np 

from wepy.resampling.wexplore2_old  import WExplore2Resampler 
from wepy.resampling.wexplore2 import WExplore2Resampler as newsampler 
import mdtraj as mdj 
   
    
if __name__ == "__main__":

    # load the coordinates
    pdb = mdj.load_pdb('sEH_TPPU_system.pdb')
    n_walkers = 8
    
    distance_matrix =  [[ 0.,         0.18137853,  0.16746621,  0.17419305,  0.20474735,  0.15551431, 0.22712255,  0.16141559],
                        [ 0.18137853,  0.,          0.17614205,  0.12979625,  0.14421478,  0.17100027, 0.18107967,  0.19682084],
                        [ 0.16746621,  0.17614205,  0.,          0.20918009,  0.23642683,  0.24217756, 0.2291024,   0.24796152],
                        [ 0.17419305,  0.12979625,  0.20918009,  0.,    0.12338937,  0.14516354,       0.15453847,  0.18512847],
                        [ 0.20474735,  0.14421478,  0.23642683,  0.12338937,  0.,     0.18476842, 0.15496477,  0.22384655],
                        [ 0.15551431,  0.17100027,  0.24217756,  0.14516354,  0.18476842,  0., 0.20447403,  0.10472166],
                        [ 0.22712255,  0.18107967,  0.2291024,   0.15453847,  0.15496477,  0.20447403, 0.,        0.24945486],
                        [ 0.16141559,  0.19682084,  0.24796152,  0.18512847,  0.22384655,  0.10472166, 0.24945486,  0.        ]]
    


    resampler = WExplore2Resampler(reference_traj=pdb,seed=5000,pmax=0.2)
    walkerwt = [ 1/n_walkers for i in range(n_walkers)]
    amp = [ 1 for i in range(n_walkers)]
    resampler.decide_clone_merge(n_walkers,walkerwt,amp,distance_matrix,debug_prints=True)
    print ("second----------------------------------------------")
    resampler = newsampler(reference_traj=pdb,seed=5000,pmax=0.2)
    walkerwt = [ 1/n_walkers for i in range(n_walkers)]
    amp = [ 1 for i in range(n_walkers)]
    resampler.decide_clone_merge(n_walkers,walkerwt,amp,distance_matrix,debug_prints=True)

    
    n_cycles = -1
    for i in range(n_cycles):
        print ("cycle", i)
        distance_matrix = np.random.rand(n_walkers,n_walkers)
        resampler.decide_clone_merge(n_walkers,walkerwt,amp,distance_matrix,debug_prints=True)
