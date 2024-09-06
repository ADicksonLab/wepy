Using SLURM to Start a Wepy Simulation
======================================

To start a Wepy simulation using SLURM, you can use the following SLURM
script template. Customize the script according to your specific
resource requirements and environment setup.

Example SLURM Script
--------------------

.. code:: bash

   #!/bin/bash --login
   ########## SBATCH Lines for Resource Request ##########

   #SBATCH --time=TIME_LIMIT            # limit of wall clock time - how long the job will run (e.g., 168:00:00 for 168 hours)
   #SBATCH -A YOUR_ALLOCATION_ACCOUNT   # replace with your allocation account
   #SBATCH --nodes=NUMBER_OF_NODES      # the number of nodes requested (e.g., 1)
   #SBATCH --ntasks=NUMBER_OF_TASKS     # the number of tasks to run (e.g., 1)
   #SBATCH --cpus-per-task=CPUS_PER_TASK # the number of CPUs (or cores) per task (e.g., 8)
   #SBATCH --gpus-per-task=GPUS_PER_TASK # request GPUs per task (e.g., v100:4)
   #SBATCH --mem=MEMORY_PER_NODE        # total memory per node (e.g., 64G)
   #SBATCH --job-name=JOB_NAME          # give your job a name for easier identification (e.g., wepy_run)
   #SBATCH --output=%x.out              # standard output file (e.g., wepy_run.out)
   #SBATCH --error=%x.err               # standard error file (e.g., wepy_run.err)

   ########## Command Lines to Run ##########

   # Load necessary modules (customize as needed)
   module load Conda/3                  # load the Conda module (modify if needed)

   # Initialize Conda (replace with the path to your Conda installation)
   eval "$(/path/to/conda/bin/conda shell.bash hook)"  # e.g., /mnt/home/username/anaconda3/bin/conda

   # Activate your Conda environment (replace with your environment name)
   conda activate your_environment_name # e.g., wepy_env

   # Set your home directory for the project (replace with your home directory)
   HOME_DIR="/path/to/your/home/directory"  # e.g., /mnt/home/username/project_dir
   JOBNAME=${SLURM_JOB_ID}

   # Set your log directory (replace with your log directory)
   LOG_DIR="$HOME_DIR/logs"  # e.g., $HOME_DIR/logs

   # Log the beginning of the run
   echo 'Beginning of the run' 1>> "$LOG_DIR/$JOBNAME.log" 2>> "$LOG_DIR/$JOBNAME.log"

   # Load CUDA module if necessary (customize as needed)
   module load centos7/lib/cuda/12  # modify CUDA version if needed

   # Change to the home directory
   cd "$HOME_DIR"

   # Log the SLURM_JOB_ID number
   echo "SLURM_JOB_ID: $SLURM_JOB_ID" 1>> "$LOG_DIR/$JOBNAME.log" 2>> "$LOG_DIR/$JOBNAME.log"

   # Running the Wepy simulation script (replace with your script name)
   echo "Running Wepy simulation script" 1>> "$LOG_DIR/$JOBNAME.log" 2>> "$LOG_DIR/$JOBNAME.log"
   python wepy_run.py  # replace with your Wepy simulation script name

Instructions for Customization
------------------------------

#. **Resource Requests:**

   -  Replace ``TIME_LIMIT`` with the desired wall clock time limit
      (e.g., ``168:00:00`` for 168 hours).
   -  Replace ``YOUR_ALLOCATION_ACCOUNT`` with your specific allocation
      account.
   -  Replace ``NUMBER_OF_NODES`` with the number of nodes you need
      (e.g., ``1``).
   -  Replace ``NUMBER_OF_TASKS`` with the number of tasks to run (e.g.,
      ``1``).
   -  Replace ``CPUS_PER_TASK`` with the number of CPUs per task (e.g.,
      ``8``).
   -  Replace ``GPUS_PER_TASK`` with the type and number of GPUs per
      task (e.g., ``v100:4``).
   -  Replace ``MEMORY_PER_NODE`` with the total memory per node (e.g.,
      ``64G``).
   -  Replace ``JOB_NAME`` with a name for your job (e.g.,
      ``wepy_run``).

#. **Conda Setup:**

   -  Replace ``/path/to/conda/bin/conda`` with the actual path to your
      Conda installation.
   -  Replace ``your_environment_name`` with the name of your Conda
      environment.

#. **Home and Log Directory:**

   -  Set ``HOME_DIR`` to the directory where your project files are
      located.
   -  Ensure ``LOG_DIR`` points to where you want the log files to be
      saved.

#. **CUDA Module:**

   -  Adjust the ``module load centos7/lib/cuda/12`` line depending on
      your cuda path.

#. **Wepy Simulation Script:**

   -  Replace ``wepy_run.py`` with the name of your Wepy simulation
      script.

This template provides flexibility for users to customize the SLURM
script according to their specific needs while maintaining a general
structure for running a Wepy simulation.
