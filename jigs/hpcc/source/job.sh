#!/bin/bash

#SBATCH -e _output/sub_logs/lysozyme_test.%J.err
#SBATCH -o _output/sub_logs/lysozyme_test.%J.out

#SBATCH -J lysozyme_test

#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --mem=20gb

#SBATCH --mail-type="BEGIN,END,FAIL"

#SBATCH --constraint="[intel18|intel16]"

#SBATCH --gres=gpu:8

trap teardown SIGINT

teardown () {
    echo "caught signal tearing processes down"

    kill "$node_exporter_pid"
    echo "Killed node_exporter proc"

    kill "$nvidia_exporter_pid"
    echo "Killed nvidia_exporter proc"

    exit 1
}


echo "----------------------"
echo "Running on host: $(hostname)"
echo "----------------------"

# run the node exporters

./_bin/node_exporter &
node_exporter_pid="$!"
echo "Started the node_exporter as PID: $node_exporter_pid"

./_bin/nvidia_gpu_prometheus_exporter &
nvidia_exporter_pid="$!"
echo "Started the nvidia_gpu_prometheus_exporter as PID: $nvidia_exporter_pid"

module purge
module load GCC/8.3.0
module load CUDA/10.1.243

module list


export ANACONDA_DIR="$HOME/.pyenv/versions/miniconda3-latest"
. ${ANACONDA_DIR}/etc/profile.d/conda.sh

which python

# activating python env
conda activate ./_env || teardown

which python

python -m simtk.testInstallation

# then run the job
echo "Running main process"
python source/lysozyme_we.py 1000 10000 48 8 'CUDA' 'WExploreResampler' 'TaskMapper' 'debug_monitor' || teardown

teardown

