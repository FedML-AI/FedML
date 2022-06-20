#!/bin/bash
set -eu

MPI=$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')

case $(uname) in

    Linux)
        MPI="${MPI:-mpich}"
        echo "Installing $MPI with apt"
        sudo apt update
        case $MPI in
            mpich)
                sudo apt install -y -q mpich libmpich-dev
                ;;
            openmpi)
                sudo apt install -y -q openmpi-bin libopenmpi-dev
                ;;
            *)
                echo "Unknown MPI implementation:" $MPI
                exit 1
                ;;
        esac
        ;;

    Darwin)
        MPI="${MPI:-mpich}"
        echo "Installing $MPI with brew"
        case $MPI in
            mpich)
                brew install mpich
                ;;
            openmpi)
                brew install openmpi
                ;;
            *)
                echo "Unknown MPI implementation:" $MPI
                exit 1
                ;;
        esac
        ;;

    Windows* | MINGW* | MSYS*)
        MPI="${MPI:-msmpi}"
        echo "Installing $MPI"
        case $MPI in
            msmpi)
                sdir=$(dirname "${BASH_SOURCE[0]}")
                pwsh "${sdir}\\setup-${MPI}.ps1"
                ;;
            *)
                echo "Unknown MPI implementation:" $MPI
                exit 1
                ;;
        esac
        ;;

    *)
        echo "Unknown OS kernel:" $(uname)
        exit 1
        ;;
esac

echo "::set-output name=mpi::${MPI}"

if [ $MPI == openmpi ]; then
    openmpi_mca_params=$HOME/.openmpi/mca-params.conf
    mkdir -p $(dirname $openmpi_mca_params)
    echo plm=isolated >> $openmpi_mca_params
    echo rmaps_base_oversubscribe=true >> $openmpi_mca_params
    echo btl_base_warn_component_unused=false >> $openmpi_mca_params
    echo btl_vader_single_copy_mechanism=none >> $openmpi_mca_params
    if [[ $(uname) == Darwin ]]; then
        # open-mpi/ompi#7516
        echo gds=hash >> $openmpi_mca_params
        # open-mpi/ompi#5798
        echo btl_vader_backing_directory=/tmp >> $openmpi_mca_params
    fi
fi
