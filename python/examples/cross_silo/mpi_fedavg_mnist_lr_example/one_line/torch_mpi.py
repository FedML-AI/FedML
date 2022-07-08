import fedml
from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD
process_id = comm.Get_rank()

sys.argv.extend(["--rank",str(process_id)])

if process_id == 0:
    print("Running Server")
    fedml.run_cross_silo_server()
else:
    print("Running Client")
    fedml.run_cross_silo_client()
