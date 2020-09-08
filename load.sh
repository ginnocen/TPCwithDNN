#!/bin/bash

###############################################################################
# This is a specific script which works with a specific python istallation at #
# /usr/bin/python3.6                                                          #
# Furthermore, virtualenv must be installed on the system                     #
# With exported paths works only on AILCEML machine                           #
###############################################################################


VIRTUALENV_PATH=~/.virtualenvs/tpcwithdnn

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
    virtualenv -p /usr/bin/python3.6 $VIRTUALENV_PATH
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
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH;
        export FLUCTUATIONDIR="/data/tpcml/";
        activate-virtualenv

        # If not on aliceml inform user to source own ROOT
        ml-activate-root > /dev/null 2>&1
        [[ "$?" != "0" ]] &&  echo "PLEASE SOURCE YOUR OWN ROOT PACKAGE"
    fi
fi
