#include <mpi.h>

int MYMPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {
    // No init here?
    int numtasks, rank;
    MPI_Comm_size(communicator, &numtasks);
    MPI_Comm_rank(communicator, &rank);
    //MPI_Request reqs[numtasks];    

    if (rank == root) {
        int i;
        // if the rank is the root then send to all other processes
        for( i = 0; i < numtasks; i++) {
            if( i != rank) {
                MPI_send(buffer, count, datatype, i, 0, communicator);
            }
        }
    } else {
        MPI_recv(buffer, count, datatype, root, 0, communicator, MPI_Status_Ignore);
    }
    return 0;
}

int MYMPI_Bcast_recv(void);