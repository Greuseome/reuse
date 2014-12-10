#!/bin/bash

if [[ ! -d ./pyenv ]]; then
    if [[ "$(which virtualenv)" == "" ]]; then
        easy_install virtual_env
    fi
    virtualenv pyenv
    source ./pyenv/bin/activate
    pip install -r requirements.txt
fi

# ATARI images and roms
[[ ! -d ./images ]] && ln -s /u/mhollen/sift/ale-assets/images images
[[ ! -d ./roms ]] && ln -s /u/mhollen/sift/ale-assets/roms roms

# CONDOR binaries
export PATH=/lusr/opt/condor/bin:$PATH


