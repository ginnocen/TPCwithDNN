#!/bin/bash

###############################################################################
# This is a specific script which works with a specific python istallation at #
# /usr/bin/python3.6                                                          #
# Furthermore, virtualenv must be installed on the system                     #
# With exported paths works only on AILCEML machine                           #
###############################################################################


VIRTUALENV_PATH=~/.virtualenvs/tpcwithdnn
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH;
export ALICE_ROOT=/home/pyadmin/alice/sw/ubuntu1804_x86-64/AliRoot/master-1

python3 -c 'import sys; sys.exit(1 if sys.version_info < (3, 6) else 0)' || { printf "You need to have at least Python 3.6 installed"; exit 1; }

create-virtualenv ()
{
    local FORCE=;
    while [[ $# > 0 ]]; do
        case "$1" in
            --force)
                FORCE=1
            ;;
            *)
                echo "ERROR: unknown option: $1";
                return 1
            ;;
        esac;
        shift;
    done;
    mkdir -p ~/.virtualenvs;
    if [[ -d $VIRTUALENV_PATH ]]; then
        if [[ -n $FORCE ]]; then
            rm -rf $VIRTUALENV_PATH;
        else
            echo 'ERROR: virtual environment already exists, use `--force` to recreate it';
            return 1;
        fi;
    fi;
    virtualenv -p /usr/bin/python3 $VIRTUALENV_PATH
}

activate-virtualenv ()
{
    if [[ -e $VIRTUALENV_PATH/bin/activate ]]; then
        source $VIRTUALENV_PATH/bin/activate;
        echo "Now using $(python -V) from $(which python)";
    else
        echo 'ERROR: no default virtualenv found`';
    fi
}

check-active()
{
    local deact=$(typeset -F | cut -d " " -f 3 | grep "deactivate$")
    if [[ "$deact" != "" ]]
    then
        echo "active"
    fi
}

deactivate-virtualenv()
{
    local deact=$(typeset -F | cut -d " " -f 3 | grep "deactivate$")
    if [[ "$deact" != "" ]]
    then
        echo "Deactivate virtualenv, goodbye :)"
        deactivate > /dev/null 2>&1
    fi
}


#############
# Main part #
#############
option=$1

# Must be sourced
if [[ $_ != $0 ]]
then

    if [[ "$(check-active)" ]]
    then
        [[ "$option" != "" ]] && echo "Options are not available in active virtualenv."
        deactivate-virtualenv
    else
        if [[ "$option" == "--recreate" || ! -d $VIRTUALENV_PATH ]]
        then
            echo "Creating virtual environment"
            create-virtualenv --force
        fi
        activate-virtualenv

        ml-activate-root > /dev/null 2>&1

        VIRT_LIBS=$(find $VIRTUALENV_PATH/lib/python*/site-packages -maxdepth 0)
        export PYTHONPATH=$VIRT_LIBS:$PYTHONPATH
    fi
fi
