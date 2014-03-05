#include "stdio.h"
#include <stdlib.h>

#include <mpi.h>

void PrintMPIError(int errnumber) {
	switch (errnumber) {
		case MPI_ERR_COMM:
		case MPI_ERR_COUNT:
		case MPI_ERR_TYPE:
		case MPI_ERR_TAG:
		case MPI_ERR_RANK:
			printf("Error in MPI Communication\n");
			break;
	//No error; MPI routine completed successfully. 
    //Invalid communicator. A common error is to use a null communicator in a call (not even allowed in MPI_Comm_rank). 
    //Invalid count argument. Count arguments must be non-negative; a count of zero is often valid. 
    //Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Type_commit). 
    //Invalid tag argument. Tags must be non-negative; tags in a receive (MPI_Recv, MPI_Irecv, MPI_Sendrecv, etc.) may also be MPI_ANY_TAG. The largest tag value is available through the the attribute MPI_TAG_UB. 
	}
	return;
}

int main(int argc, char *argv[]) {
	int rank,size;
	char *cpu_name;

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);	//master has tid 0
	MPI_Comm_size(MPI_COMM_WORLD, &size);	//number of processes
	if (rank==0)
		printf("Number of threads: %i\n",size);
	
	//on EVERY process, allocate space for the machine name
	cpu_name    = (char *)calloc(80,sizeof(char));
	gethostname(cpu_name,80);

	if (rank==0) {
		int test = 123;
		PrintMPIError( MPI_Send(&test,1, MPI_INT, 1, 0, MPI_COMM_WORLD));
	}
	if (rank == 1) {
		MPI_Status status;
		int test_rcv = 0;
		MPI_Recv(&test_rcv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		printf("Received Message: Int=%i\n",test_rcv);
	}

	printf("hello MPI user: from process = %i on machine=%s, Processes: %i\n",
         rank, cpu_name, size);
	MPI_Finalize();
	return(0);
}
