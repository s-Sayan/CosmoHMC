from mpi4py import MPI

# Initialize the MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # Get the rank of the process
size = comm.Get_size() # Get the total number of processes

# Suppose we have an array of tasks to be executed by multiple processes
tasks = range(10) # This could be any iterable of tasks

# Distribute tasks evenly among the processes
for i in tasks:
    if i % size == rank:
        # This task is assigned to this process
        print(f"Process {rank} is handling task {i}")
