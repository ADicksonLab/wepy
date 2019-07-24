"""Module that provides functionality for performing parallel
distributed calculations using the dask suite of tools.

Because these tools are to be used in tandem with each other here is an overview of how they fit together:


Dask provides parallelism over arbitrary (serializable) python
objects through the bag object. Because WepyHDF5 objects (or any
HDF5 dataset reference) are not serializable (because they contain
file handles to an underlying HDF5 file) we instead provide a
simple datastructure that provides the specifications for which
dataset to read from and what piece of the data is to be read
in. These the 'items' (which are returned by this function) are a
simple tuple containing:

- file path to WepyHDF5 file
- run index
- trajectory index
- frame indices
- field names/paths

This is everything we need to call the `get_traj_field` method and
generate a traj fields, which can then be operated on by various
other functions, e.g. `traj_fields_to_mdtraj` or any function that
could be passed to `WepyHDF5.compute_observable`.

This function does not directly create a dask bag object, but can
easily be done, via
`dask.bag.from_sequence(traj_fields_bag_items(*args))`. After
which we only need to actually load the data into memory using the
other helper function `load_frames_fields` which takes as a single
argument a single 'item' from this functions output.

I.e.:
`dask.bag.from_sequence(traj_fields_bag_items(*args)).map(load_frames_fields)`

This will create a dask bag which can then be used to perform
various distributed, parallel operations on it, such as
`dask.bag.map`. Think of it as a distributed version of
`WepyHDF5.compute_observable` (except we can't save directly to
the WepyHDF5 in the same invocation).

For example:

```
import dask
from wepy.util.mdtraj import traj_fields_to_mdtraj
from wepy.analysis.distributed import traj_fields_chunk_items, load_traj_chunks

bag = dask.bag.from_sequence(
         traj_fields_bag_items(
                'results.wepy.h5',
                ['positions', 'box_vectors'],
                chunk_size=500
                              )
                            )

# load the trajectory fields "chunks" into distributed memory
a = bag.map(load_traj_chunks)
# convert our "trajectory fields" chunks to an mdtraj Trajectory object
b = bag.map(traj_fields_to_mdtraj)
# compute the solvent accessible surface area of the trajectory chunks
c = bag.map(mdj.shrake_rupley)
results = b.compute()

```

The intermediate values a, b, and c are just place holders for
delayed operations since, dask is 'lazy' in the sense that it
doesn't actually do any of the work in the 'map' calls until you
call 'compute'.

The choice of the chunk size is also of high importance for getting
efficient calculations. To choose the chunk size appropriately
consider the resources and parallelism available to each of your
workers and the amount of data that you have.

The smaller the chunk, the more tasks that will be generated for
dask. Each task has an intrinsic overhead associated with it, may
require serialization and communication if operating in separate
processes or hosts, and furthermore needs to be scheduled for
execution by dask. An excessively large number of tasks will cause the
scheduler to grind to a halt so we want to increase the chunk size to
large enough that the scheduler can handle it and that communication
doesn't become more expensive than actual calculation. However, we are limited in a few ways to the size of the chunks:

1. length of individual trajectories in a run
2. memory of a worker process
3. throughput of a worker process

Firstly we are limited in a strong sense by the fact that the largest
possible chunk for a single trajectory is the whole trajectory. Even
if larger chunks would be theoretically, possible. Probably there are
advanced optimizations that could be made if our trajectories happened
to be very numerous and very short, we assume that trajectories are
reasonably long and not numbering in the hundreds of thousands or
millions. This is also dependent on the actual size of a single frame,
which may vary greatly in size for different domains. In any case you
should very likely do some dimensionality reduction before performing
calculations (i.e. stripping out waters in molecular dynamics
simulations).

Secondly, a chunk must be able to fit into the memory of a worker
process. This is a straightforward and intuitive limitation.
Thirdly, we must not make the chunks so large that the degree of
parallelism is diminished.

We leave off here because optimizing parallel calculation is a huge
topic and will change from problem to problem. Thus we encourage trial
and error. In our experience however, a nonresponsive (but non-error
producing calculation) is probably due to the scheduler being
inundated with too many tasks (chunk sizes too small, say of only 1
frame).

"""

import numpy as np

from wepy.hdf5 import WepyHDF5

def load_frames_fields(args):

    from wepy.hdf5 import WepyHDF5

    # unpack the arguments
    wepy_h5_path, run_idx, traj_idx, frame_idxs, fields = args

    wepy_h5 = WepyHDF5(wepy_h5_path, mode='r')

    with wepy_h5:
        frame_fields = {}
        for field in fields:
            frame_fields[field] = wepy_h5.get_traj_field(run_idx, traj_idx, field,
                                                         frames=frame_idxs)
    return frame_fields

def traj_fields_chunk_items(wepy_h5_path, fields, chunk_size=1):
    """Generate items that can be used to create a dask.bag object.



    Arguments
    ---------

    wepy_h5_path : str
        The file path to the WepyHDF5 file that will be read from.

    fields : list of str
        The field names/paths for the data to be retrieved.

    chunk_size : int
        This is the size of the chunk (i.e. number of frames) that
        will be retrieved from each trajectory. This is the unit of
        data for which a single task will work on. Dask will also
        partition these chunks as it sees fit.

    """



    # open the HDF5
    try:
        wepy_h5 = WepyHDF5(wepy_h5_path, mode='r')
    except OSError:
        print("Failed to open HDF5")
        return None

    with wepy_h5:
        items = []
        # DEBUG
        for run_idx in [0]: #wepy_h5.run_idxs:
            for traj_idx in wepy_h5.run_traj_idxs(run_idx):

                num_frames = wepy_h5.num_traj_frames(run_idx, traj_idx)

                # split it allowing for an unequal chunk sizes
                chunks = np.array_split(range(num_frames),
                                        num_frames // chunk_size)

                for frame_idxs in chunks:
                    items.append((wepy_h5_path, run_idx, traj_idx, frame_idxs, fields))

    bag = dbag.from_sequence(items)

    return bag

