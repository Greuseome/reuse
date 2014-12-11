#!/bin/bash

if [[ ! -d ./pyenv ]]; then
    if [[ "$(which virtualenv)" == "" ]]; then
        echo "Cannot find virtualenv. Run setup script using machine that has virtualenv (e.g. turtles)."
        exit
    fi
    virtualenv pyenv
    source ./pyenv/bin/activate
    easy_install -U distribute
    pip install -r requirements.txt
else
    source ./pyenv/bin/activate
fi

#/bin/bash -c ". ./pyenv/bin/activate; exec /bin/bash -i"

# ATARI images and roms
[[ ! -d ./images ]] && ln -s /u/mhollen/sift/ale-assets/images images
[[ ! -d ./roms ]] && ln -s /u/mhollen/sift/ale-assets/roms roms

# CONDOR binaries
export PATH=/lusr/opt/condor/bin:$PATH


