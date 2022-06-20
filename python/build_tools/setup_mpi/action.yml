---
name: 'Setup MPI'
description: 'Set up a specific MPI implementation.'
author: 'Lisandro Dalcin'
inputs:
  mpi:
    description: "MPI implementation name."
    required: false
    default: '' # Linux/macOS: 'mpich', Windows: 'msmpi'
outputs:
  mpi:
    description: "The installed MPI implementation name."
    value: ${{ steps.setup-mpi.outputs.mpi }}
runs:
  using: 'composite'
  steps:
    - id: setup-mpi
      run: ${GITHUB_ACTION_PATH}/setup-mpi.sh "$MPI"
      shell: bash
      env:
        MPI: ${{ inputs.mpi }}
