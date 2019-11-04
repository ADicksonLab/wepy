# source me!!

# i.e. run:

# . tooling.sh

# or

# source tooling.sh

# figure out which directory is you anaconda dir
export ANACONDA_DIR=$(dirname $(dirname $(which conda)))

# install invoke here
$ANACONDA_DIR/bin/pip install -r requirements_tooling.txt
