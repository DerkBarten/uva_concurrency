#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
/*
################################################################################
run:
mpicc -c broadcast_test.c
mpicc broadcast_test.o -o broadcast
mpirun -np  N broadcast -quiet

TODO:
    - should finish when last message is received
*/

int MYMPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {
    int numtasks, rank, left, right;
    MPI_Comm_size(communicator, &numtasks);
    MPI_Comm_rank(communicator, &rank);
    MPI_Request reqs[numtasks];  


    if (rank == root) {
        left = numtasks - 1;
        right = root + 1;
        printf("process %i receives from to %i\n", rank, left );
        MPI_Recv(buffer, count, datatype, numtasks - 1, numtasks-1, communicator, MPI_STATUS_IGNORE);
        MPI_Isend(buffer, count, datatype, 1, 0, communicator, &reqs[rank]);
        printf("process %i send to %i\n", rank, right );
        
    } else if (rank == numtasks - 1) {
        left = rank - 1;
        right = root;
        printf("process %i receives from to %i\n", rank, left );
        MPI_Isend(buffer, count, datatype, root, 0 , communicator, &reqs[rank]);
        MPI_Recv(buffer, count, datatype, rank - 1 , numtasks - 1, communicator, MPI_STATUS_IGNORE);
        printf("process %i send to %i\n", rank, right );
    } else {
        right = rank + 1;
        left = rank - 1;
        printf("process %i receives from to %i\n", rank, left );
        MPI_Recv(buffer, count, datatype, rank - 1, 0, communicator, MPI_STATUS_IGNORE);
        MPI_Isend(buffer, count, datatype, rank + 1, 0, communicator, &reqs[rank]);
        printf("process %i send to %i\n", rank, right );
    }

    MPI_Finalize();
    return 0;
}

void main() {

    MPI_Init(NULL, NULL);
    int data = 1000;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MYMPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return;

}