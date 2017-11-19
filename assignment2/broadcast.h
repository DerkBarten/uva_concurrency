#include <mpi.h>

/*
 * INOUT  : buffer address
 * IN     : buffer size
 * IN     : datatype of entry
 * IN     : root process (sender)
 * IN     : communicator
 */
int MYMPI_Bcast (void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm communicator);

int MYMPI_Bcast_recv(void);