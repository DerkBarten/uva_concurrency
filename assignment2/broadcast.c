#include <mpi.h>

int MYMPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {
    // No init here?
    int numtasks, rank;
    MPI_Comm_size(communicator, &numtasks);
    MPI_Comm_rank(communicator, &rank);
    //MPI_Request reqs[numtasks];    

    if (rank == root) {
        MPI_Isend(buffer, count, datatype, i, 0, communicator, &reqs[i]);
    }
    return 0;
}

int MYMPI_Bcast_recv(void);