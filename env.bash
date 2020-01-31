# figure out which directory is you anaconda dir
export ANACONDA_DIR=$(dirname $(dirname $(which conda)))


# enable tab completion for invoke
source <(inv --print-completion-script bash)
