#!/bin/sh

## used in provisioning for LXDock when you craete a container from
## scratch to get the bare minimum configuration

apt install -y \
    ssh \
    mg \
    rsync \
    git \
    tmux

# password: 'password'
usermod --password '$6$lZKBQ/eOBzntgg8j$QlbpSwx4tr5IOfYV7GdfKtLXB0BB8AjXeTBa6h.XCCp/seKj207okOpK0ZHq2mwFxlyOrFmVfs5ak77ec/y1f1' root

usermod -aG sudo ${USER}
