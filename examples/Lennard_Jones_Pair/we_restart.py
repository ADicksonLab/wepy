import pickle

# to unpickle the distance metric you need to import the class
# definition from the script it was defined in
from we import PairDistance

with open('./outputs/restart.pkl', 'rb') as rf:
    restarter = pickle.load(rf)

sim_manager = restarter.new_sim_manager()

def main(continue_run, n_cycles, steps, filepath=None, debug_prints=False, seed=None):

    # if another filepath is given we want to change the file where
    # the reporters report to
    if filepath is not None:
        pass

    ### RUN the simulation
    print("Starting run")
    sim_manager.continue_run_simulation(continue_run, n_cycles, steps,
                                        debug_prints=True)
    print("Finished run")


if __name__ == "__main__":

    import time
    import sys

    if sys.argv[1] == "--help" or sys.argv[1] == '-h':
        print("arguments: continue_run_idx, n_cycles, n_steps")
    else:

        continue_run = int(sys.argv[1])
        n_cycles = int(sys.argv[2])
        n_steps = int(sys.argv[3])
        try:
            filepath = str(sys.argv[4])
        except KeyError:
            filepath = None

        print("Number of steps: {}".format(n_steps))
        print("Number of cycles: {}".format(n_cycles))

        steps = [n_steps for i in range(n_cycles)]

        start = time.time()
        main(continue_run, n_cycles, steps, filepath=filepath, debug_prints=True)
        end = time.time()

        print("time {}".format(end-start))
