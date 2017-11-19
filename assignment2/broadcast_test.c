#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>
/*
################################################################################
run:
mpicc -c broadcast_test.c
mpicc broadcast_test.o -o broadcast
mpirun -np  3 broadcast -quiet

*/

int MYMPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm communicator) {
    int numtasks, rank;
    MPI_Comm_size(communicator, &numtasks);
    MPI_Comm_rank(communicator, &rank);
    //MPI_Request reqs[numtasks];    

    if (rank == root) {
        int i;
        printf("%i tasks\n", numtasks);
        for( i = 0; i < numtasks; i++) {
            if( i != rank) {
                printf("rank %i send message\n", i);
                MPI_Send(buffer, count, datatype, i, 0, communicator);
            }
        }
    } else {
        MPI_Recv(buffer, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
        printf("rank %i received message\n", rank);
    }
    MPI_Finalize();
    return 0;
}

void main() {

    MPI_Init(NULL, NULL);
    int data = 100;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MYMPI_Bcast (&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

}