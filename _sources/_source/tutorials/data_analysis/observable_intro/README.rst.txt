Introducing Observables
=========================

Observables are quantities calculated from simulation data that provide insights into the system's behavior. A common observable is RMSD (Root Mean Square Deviation). In this guide, we'll walk through the process of defining and using observables within a simulation.

.. code:: python

    import mdtraj as mdj

    # Load the PDB file
    pdb_path = 'path/to/your/file.pdb'
    pdb = mdj.load_pdb(pdb_path)


To calculate observables, we often need to select specific residues or atoms. Here’s how to select residues using MDTraj.

.. TODO: Update it according to tutorial structure

.. code:: python

    # Select specific residues based on segment names and residue numbers
    active_1_res = pdb.top.select('(segname PROA and (residue 399)) or (segname PROB and (residue 366))')
    active_2_res = pdb.top.select('(segname PROB and (residue 399)) or (segname PROC and (residue 366))')
    active_3_res = pdb.top.select('(segname PROC and (residue 399)) or (segname PROA and (residue 366))')

    CARA_active_res = pdb.top.select('segname CARA and (residue 24 25)')
    CARB_active_res = pdb.top.select('segname CARB and (residue 24 25)')
    CARC_active_res = pdb.top.select('segname CARC and (residue 24 25)')


An observable function takes simulation data as input and computes a specific property. You can define your own functions based on your research needs.

Here’s an example of how you might define an observable function to compute centroid distances:

.. code:: python

    import numpy as np

    def centroid_distance(fields_d, *args, **kwargs):
        centroid_distances = []
        for i in range(len(fields_d['positions'])):
            active_1_centroid = np.mean(fields_d['positions'][i][args[0]['active_1_res']], axis=0)
            active_2_centroid = np.mean(fields_d['positions'][i][args[0]['active_2_res']], axis=0)
            active_3_centroid = np.mean(fields_d['positions'][i][args[0]['active_3_res']], axis=0)
            
            CARA_active_centroid = np.mean(fields_d['positions'][i][args[0]['CARA_active_res']], axis=0)
            CARB_active_centroid = np.mean(fields_d['positions'][i][args[0]['CARB_active_res']], axis=0)
            CARC_active_centroid = np.mean(fields_d['positions'][i][args[0]['CARC_active_res']], axis=0)
            
            centroid_distances.append(np.array([
                np.linalg.norm(active_1_centroid - CARA_active_centroid),
                np.linalg.norm(active_3_centroid - CARB_active_centroid),
                np.linalg.norm(active_2_centroid - CARC_active_centroid)
            ]))
        
        return np.array(centroid_distances)


Once we have the observable function, we can use it to compute properties during the simulation. Wepy provides tools to manage simulation data and compute observables.

.. code:: python

    from wepy.hdf5 import WepyHDF5

    # Path to the WEPY results files
    outputs_dir = ['path/to/output1', 'path/to/output2']
    wepy_results = [WepyHDF5(output_dir + '/wepy.results.h5', mode='r+') for output_dir in outputs_dir]

    for wepy_result in wepy_results:
        with wepy_result:
            args = [{
                'active_1_res': active_1_res, 
                'active_2_res': active_2_res, 
                'active_3_res': active_3_res, 
                'CARA_active_res': CARA_active_res, 
                'CARB_active_res': CARB_active_res,
                'CARC_active_res': CARC_active_res
            }]
            
            obs = wepy_result.compute_observable(
                centroid_distance,
                ['positions'],  # Specify the required fields
                args=(args),    # Pass custom arguments
                save_to_hdf5='centroid',  # Save results to the HDF5 file
                return_results=True
            )


Now you have centroid distance data that you can use to analyze the system's behavior. You can also use the data to visualize the system's behavior over time.