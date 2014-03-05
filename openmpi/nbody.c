#include "stdio.h"
#include <stdlib.h>
#include <time.h>

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
    //Invalid communicator. A common error is to use a null communicator in a c$
    //Invalid count argument. Count arguments must be non-negative; a count of $
    //Invalid datatype argument. May be an uncommitted MPI_Datatype (see MPI_Ty$
	        }
        return;
}

/*template <class T> 
int sgn(T val) {
    return (((T)(0)) < val) - (val < ((T)(0)));
}*/
int sgn(float val) {
    return ((0 < val) - (val < 0));
}

struct particle{
        int id; //should be numbered beginning from zero without omitting ids as it is used to reference the force matrix!
        float x0,v0;
};

int main(int argc, char *argv[]) {      //This is executed 'size' times
	int rank,size;
	char *cpu_name;
	const int N_part = 2;
	
	//MPI Initialisation
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);   //master has tid 0
	MPI_Comm_size(MPI_COMM_WORLD, &size);   //number of processes
	if (rank==0)
			printf("Number of threads: %i\n",size);
	cpu_name    = (char *)calloc(80,sizeof(char));
	gethostname(cpu_name,80);
	
	//Initialize Initial velocities and locations of N particles -> Total particles: N*size, which is also the matrix size
	//struct particle* part_data = calloc(sizeof((struct particle)*N_part));
	struct particle* part_data = calloc(N_part, sizeof(struct particle));
	srand(time(NULL)*rank);
	//printf("Rand: %i out of Randmax: %i\n", rand(),RAND_MAX);
	int i;
	for (i=0; i<N_part; i++) {
		part_data[i].x0 = rand()/((float)RAND_MAX);
		part_data[i].v0 = rand()/((float)RAND_MAX);
		part_data[i].id = rank*N_part+i;
	}
	float forces[N_part*size][N_part*size];  //should be deleted every loop so that one can differentiate whether an element or the symmetric partner was already calculated
	
	//systolic loop
	struct particle part_tmp[N_part];
	for (i=0; i<N_part; i++) {
		part_tmp[i].x0 = part_data[i].x0;
		part_tmp[i].v0 = part_data[i].v0;
		part_tmp[i].id = part_data[i].id;
	}
	int i_sys=-1;
	for (i_sys=0; i_sys<size; i_sys++) {
		//Calculate force and sort into matrix
		int m,n;
		for (m=0; m<N_part; m++) 
			for(n=0; n<N_part; n++) 
				if (m != n) {
					float r = part_tmp[m].x0 - part_data[n].x0 ;
					forces[part_tmp[m].id][part_data[n].id] = sgn(r)/(r*r);
				} else
					forces[part_tmp[m].id][part_data[n].id] = 0;
		MPI_Request request;
		int rank_send = (rank+1 > N_part-1) ? (0) : (rank+1);
		MPI_Isend( part_tmp, sizeof(struct particle)*N_part, MPI_CHAR, rank_send, 0, MPI_COMM_WORLD, &request);
		MPI_Status status;
		int rcv_rank = (rank-1 < 0) ? (N_part-1) : (rank-1);
		MPI_Recv( part_tmp, sizeof(struct particle)*N_part, MPI_CHAR, rcv_rank, 0, MPI_COMM_WORLD, &status);
		MPI_Wait (&request, &status);
		/*if (rank == 0) {
			int test = 123;
			PrintMPIError( MPI_Send(&test,1, MPI_INT, 1, 0, MPI_COMM_WORLD));
		}
		if (rank == 1) {
			MPI_Status status;
			int test_rcv = 0;
			MPI_Recv(&test_rcv, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
			printf("Received Message: Int=%i\n",test_rcv);
		}*/
		//printf("isys: %i, hello MPI user: from process = %i on machine=%s, Processes: %i\n", i_sys, rank, cpu_name, size);
	}
	
	if (rank==0) {
		printf("isys: %i, hello MPI user: from process = %i on machine=%s, Processes: %i\n", i_sys, rank, cpu_name, size);
		int m,n;
		for (m=0; m<N_part; m++) {
			for(n=0; n<N_part; n++) 
				if (m==n)
					printf("0.00 ");
				else
					printf("%1.2f", forces[m][n]);
			printf("\n");
		}
		int code = 123;
		int rank_send = (rank+1 > N_part-1) ? (0) : (rank+1);
		MPI_Send(&code, 1, MPI_INT, rank_send, 1, MPI_COMM_WORLD);
	} else {
		int code_rcv;
		int rank_rcv = (rank-1 < 0) ? (N_part-1) : (rank-1);
		MPI_Status status;
		MPI_Recv(&code_rcv, 1, MPI_INT, rank_rcv, 1, MPI_COMM_WORLD, &status);
		if (code_rcv != 123)
			printf("Wrong print signal received!\n");
		printf("isys: %i, hello MPI user: from process = %i on machine=%s, Processes: %i\n", i_sys, rank, cpu_name, size);
		int m,n;
		for (m=0; m<N_part; m++) {
			for(n=0; n<N_part; n++) 
				if (m==n)
					printf("0.00\t");
				else
					printf("%1.2f\t", forces[m][n]);
			printf("\n");
		}
	}
	
	//Finalize
	MPI_Finalize();
	free(cpu_name);
	free(part_data);
	return(0);
}
