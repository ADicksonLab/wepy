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
import time

from collections import defaultdict
from copy import deepcopy

import numpy as np

import dask.bag as dbag

from wepy.hdf5 import WepyHDF5
from wepy.util.util import concat_traj_fields

RESULT_FIELD_NAME = 'observable'

def traj_fields_chunk_items(wepy_h5_path, fields,
                            run_idxs=Ellipsis,
                            chunk_size=Ellipsis):
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

    Returns
    -------
    chunk_specs : list of dict of str : value

    """

    # open the HDF5
    try:
        wepy_h5 = WepyHDF5(wepy_h5_path, mode='r')
    except OSError:
        print("Failed to open HDF5")
        return None

    with wepy_h5:

        # choose the run idxs
        if run_idxs is not Ellipsis:
            assert all([run_idx in wepy_h5.run_idxs for run_idx in run_idxs]), "run_idx not in runs"
        else:
            run_idxs = wepy_h5.run_idxs

        chunk_specs = []
        for run_idx in run_idxs:
            for traj_idx in wepy_h5.run_traj_idxs(run_idx):

                num_frames = wepy_h5.num_traj_frames(run_idx, traj_idx)

                # determine the specific frame indices in the chunks

                # if the chunk size is either larger than the
                # trajectory, or chunk size is Ellipsis we take the
                # whole trajectory
                if chunk_size is Ellipsis:
                    chunks = [range(num_frames)]
                elif chunk_size > num_frames:
                    chunks = [range(num_frames)]
                else:
                    # split it allowing for an unequal chunk sizes
                    chunks = np.array_split(range(num_frames),
                                            num_frames // chunk_size)

                for frame_idxs in chunks:
                    chunk_spec = {
                        'wepy_h5_path' : wepy_h5_path,
                        'run_idx' : run_idx,
                        'traj_idx' : traj_idx,
                        'frame_idxs' : frame_idxs,
                        'fields' : fields,
                    }
                    chunk_specs.append(chunk_spec)

    return chunk_specs

def load_chunk(chunk_spec):

    wepy_h5 = WepyHDF5(chunk_spec['wepy_h5_path'], mode='r')

    with wepy_h5:
        frame_fields = {}
        for field in chunk_spec['fields']:
            frame_fields[field] = wepy_h5.get_traj_field(chunk_spec['run_idx'],
                                                         chunk_spec['traj_idx'],
                                                         field,
                                                         frames=chunk_spec['frame_idxs'])

    # combine the chunk spec with the traj_fields data
    chunk_spec['traj_fields'] = frame_fields

    return chunk_spec

def chunk_func_funcgen(func,
                       input_keys=['traj_fields'],
                       result_name=None,
                       keep_inputs=False):

    # if the result name wasn't given use the function name
    if result_name is None:
        result_name = func.__name__
    else:
        assert isinstance(result_name, str)

        result_name = result_name


    def chunk_func(chunk_spec):


        assert not result_name in chunk_spec.keys()

        fields = []
        for key in input_keys:

            # pop the trajectory fields off if we are to keep them or not
            if keep_inputs:
                field = chunk_spec[key]
            else:
                field = chunk_spec.pop(key)

            fields.append(field)

        chunk_spec[result_name] = func(*fields)

        return chunk_spec

    return chunk_func

# NOTE: probably shouldn't need this. THere is a dask function called
# 'pluck' which does the same thing, except only for 1 key
def unwrap_chunk_funcgen(keys=None):
    """Consider first using the bag.pluck method"""

    assert keys is not None, "must give at least one key or this is useless"
    assert len(keys) > 0, "must give at least one key or this is useless"

    def unwrap_chunk_func(chunk_struct):

        return {key: value for key, value in chunk_struct.items()
                if key in keys}

    return unwrap_chunk_func

# for reduction
def chunk_key_func(chunk_spec):
    return (chunk_spec['run_idx'], chunk_spec['traj_idx'])

def init_chunk():

    return {'wepy_h5_path' : None,
            'run_idx' : None,
            'traj_idx' : None,
            'fields' : None,
            'frame_idxs' : np.array([])}

def chunk_concat_funcgen(*concat_funcs):

    def chunk_concat(cum_chunk_spec, new_chunk_spec):

        new_chunk = deepcopy(cum_chunk_spec)

        # check to see that the accumulator chunk struct has been
        # initialized, if not initialize it with the values from the new_chunk
        if (
                (cum_chunk_spec['wepy_h5_path'] is None) or
                (cum_chunk_spec['run_idx'] is None) or
                (cum_chunk_spec['traj_idx'] is None) or
                (cum_chunk_spec['fields'] is None)
        ):
            cum_chunk_spec['wepy_h5_path'] = new_chunk_spec['wepy_h5_path']
            cum_chunk_spec['run_idx'] = new_chunk_spec['run_idx']
            cum_chunk_spec['traj_idx'] = new_chunk_spec['traj_idx']
            cum_chunk_spec['fields'] = new_chunk_spec['fields']

        # concatenate the frame indices in this chunk
        new_chunk['frame_idxs'] = np.concatenate(
            [cum_chunk_spec['frame_idxs'], new_chunk_spec['frame_idxs']])

        # for each extra concat function feed it the two chunk specs
        for concat_func in concat_funcs:

            # update the cumulative chunk spec in-place
            new_chunk = concat_func(new_chunk,
                                    new_chunk_spec)


        return new_chunk

    return chunk_concat


def chunk_array_concat_funcgen(field):

    def func(cum_chunk_spec, new_chunk_spec):

        # only add it if it has been initialized in the cum_chunk
        if field in cum_chunk_spec:
            cum_chunk_spec[field] = np.concatenate([cum_chunk_spec[field], new_chunk_spec[field]])

        # otherwise set just the new chunk
        else:
            cum_chunk_spec[field] = new_chunk_spec[field]

        return cum_chunk_spec

    return func

def chunk_traj_fields_concat(cum_chunk_spec, new_chunk_spec):
    """Binary operation for dask foldby reductions for concatenating chunk
    specs with a traj_fields payload"""

    # concatenate the traj fields
    cum_chunk_spec['traj_fields'] = concat_traj_fields(
        [cum_chunk_spec['traj_fields'], new_chunk_spec['traj_fields']])

    return cum_chunk_spec


# this needs to be tested etc.
# def compute_observable_gen(func,
#                            wepy_h5_path,
#                            dask_client,
#                            fields,
#                            chunk_size=Ellipsis,
#                            # TODO replace with traj_sels
#                            run_idxs=Ellipsis):

#     with dask_client:

#         chunks = traj_fields_chunk_items(wepy_h5_path,
#                                          fields,
#                                          chunk_size=chunk_size,
#                                          run_idxs=run_idxs)

#         # DEBUG
#         # TODO add to logging
#         print("generated {} chunks".format(len(chunks)))

#         frame_fields_bag = dbag.from_sequence(chunks)

#         last_step = compute_observable_graph(func, frame_fields_bag, chunk_size)

#         # then we can just iterate through the values (JIT compute)
#         # and set them into the appropriate trajectory
#         while True:
#             traj_id, struct = last_step.take(1)
#             yield (traj_id, struct['observable'])

def _by_traj_to_multidimensional(traj_d):
    """Convert a dictionary of keys (run_idx, traj_idx) and values (of
    dimension val_dim) as arrays to a list of lists of the value
    arrays.

    """

    # get all of the unique run_idxs and sort them
    run_idxs = sorted(list(set([traj_id[0] for traj_id in traj_d.keys()])))

    # then get which trajectories each run has
    run_trajs = defaultdict(list)
    for run_idx, traj_idx in traj_d.keys():
        run_trajs[run_idx].append(traj_idx)

    for run_idx in run_trajs.keys():
        # sort the traj indices (these are unique already within the
        # run)
        run_trajs[run_idx] = sorted(list(run_trajs[run_idx]))

    # then just iterate in order over the run_idxs and inside each the
    # traj indices, as we build these add them to the big structure
    runs_arr = []
    for run_idx in run_idxs:
        run_arr = []
        for traj_idx in run_trajs[run_idx]:
            traj_id = (run_idx, traj_idx)
            run_arr.append(traj_d[traj_id])
        runs_arr.append(run_arr)

    return runs_arr


def compute_observable(func,
                       wepy_h5_path,
                       dask_client,
                       fields,
                       chunk_size=Ellipsis,
                       num_partitions=100,
                       # TODO replace with traj_sels
                       run_idxs=Ellipsis):

    with dask_client:

        chunks = traj_fields_chunk_items(wepy_h5_path,
                                         fields,
                                         chunk_size=chunk_size,
                                         run_idxs=run_idxs)

        # DEBUG
        # TODO add to logging
        print("generated {} chunks".format(len(chunks)))

        frame_fields_bag = dbag.from_sequence(chunks, npartitions=num_partitions)

        last_step = compute_observable_graph(func, frame_fields_bag, chunk_size)

        chunk_results = last_step.compute()

        # get just the result value with the traj id
        results_d = {traj_id : chunk_struct[RESULT_FIELD_NAME]
                   for traj_id, chunk_struct in chunk_results}

        results_arr = _by_traj_to_multidimensional(results_d)

        return results_arr


def compute_observable_graph(func, chunk_bag, chunk_size):

    # since the inputs to any mapped function will be chunk
    # structs (i.e. having metadata about the identity of the
    # chunk, plus any data it drags along, namely traj_fields and
    # computed observables, but perhaps other intermediates in
    # more complex pipelines) we need to wrap any function that we
    # want to call on this data with a function which gets the
    # field of data it needs, and the name of the field it will
    # save data in, optionally we can get rid of the input data if
    # it is no longer needed

    chunk_func = chunk_func_funcgen(func,
                                    input_keys=['traj_fields'],
                                    result_name=RESULT_FIELD_NAME,
                                    keep_inputs=False)

    # start constructing computational graph

    # load chunks into distributed memory
    load_step = chunk_bag.map(load_chunk)

    # run the function over the chunks
    map_step = load_step.map(chunk_func)

    # optionally, defrag the chunks into chunks which are the same
    # as trajectories using a reduce step

    # TODO currently we just always reduce, since that gives us the
    # structure we need at the end, and don't have that for the
    # non-reduction option i.e. when chunk size is same as
    # trajectories. We didn't want to code up the function to
    # structure it that way (in dask) and so leave this for now and if
    # it becomes a performance issue we can make that
    if True: #chunk_size is not Ellipsis:

        # generate the concatenation function for the result so we can
        # reduce and defrag it, this only deals with the part of the chunk of the field name
        result_concat = chunk_array_concat_funcgen(RESULT_FIELD_NAME)

        # now generate the entire chunk concatenation which aggregates
        # multiple other specific field concatenation functions,
        # fundamentally this really only deals with concatenating the
        # frame indices for the chunk
        chunk_concat = chunk_concat_funcgen(result_concat)

        init_cum_chunk = init_chunk()

        # reduce the results by de-fragging the chunks into trajectory
        # chunks
        defrag_step = map_step.foldby(chunk_key_func, chunk_concat,
                                      init_cum_chunk)
    else:
        defrag_step = map_step

    return defrag_step
