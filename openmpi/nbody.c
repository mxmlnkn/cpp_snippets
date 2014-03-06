#include "stdio.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <netdb.h>
#include <unistd.h>	//for gethostname

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
	}
	return;
}

void PrintMatrix(float *matrix, int xmax, int ymax, int rank, const char* headline) {
	const int OUT_SR_LEN = 1024;
	char sr[OUT_SR_LEN],tmpsr[200];
	snprintf(sr, OUT_SR_LEN, "\n%s\nI am process: %i\n", headline, rank);
	for (int m=0; m<xmax; m++) {
		for(int n=0; n<ymax; n++) {
			sprintf(tmpsr,"%1.2f ", *(matrix + m*ymax+n));
			strcat(sr, tmpsr);
		}
		sprintf(tmpsr,"\n\0");
		strncat(sr, tmpsr, strlen(tmpsr));
	}
	printf("%s",sr);
	fflush(stdout);
}

//Not possible in raw C
/* template<class T> 
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
	
	//MPI Initialization
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);   //master has tid 0
	MPI_Comm_size(MPI_COMM_WORLD, &size);   //number of processes
	if (rank==0)
			printf("Number of threads: %i\n",size);
	cpu_name    = (char *)calloc(80,sizeof(char));
	gethostname(cpu_name,80);
	
	//Initialize initial velocities and locations of N particles -> Total particles: N*size, which is also the matrix size
	//struct particle* part_data = calloc(sizeof((struct particle)*N_part));
	struct particle* part_data = calloc(N_part, sizeof(struct particle));
	srand(time(NULL)*(rank+1));
	printf("Time: %i, Wtime: %1.4f\n",(int)time(NULL),MPI_Wtime());
	//printf("Rand: %i out of Randmax: %i\n", rand(),RAND_MAX);
	for (int i=0; i<N_part; i++) {
		part_data[i].x0 = rand()/((float)RAND_MAX);
		part_data[i].v0 = rand()/((float)RAND_MAX);
		part_data[i].id = rank*N_part+i;
	}
	
	float partf[N_part][N_part*size];	//forces: cols for all particles and rows only for the particles in the process-cell
	//concatenating alle partial force matrices gives complete force matrix
	//partf is in memory like [0,0] [0,1] [0,2] ... [0,cols-1] [1,0] [1,1] [1,2] ... meaning: column after column
	//this structure makes cocatenating the partial force matrices easy!!!
	memset(partf, 0, sizeof(float)*N_part*size*N_part);
	/*for (int m=0; m<N_part; m++) 
		for(int n=0; n<N_part*size; n++) 
			partf[m][n] = 0;*/
	
	//float** + 1 results in +8 of pointeradress, float* +1 results in +4 of adress!
	/*if (rank==0) {
		float* fp = (float*)partf;
		printf("=> floatsize: %i\n",sizeof(float));
		printf("=> partf:%p [1]:%p [1][1]:%p [1][1]=%1.2f\n",partf,&(partf[1]),&(partf[1][1]),partf[1][1]);
		printf("=>     fp:%p [1]:%p [1][1]:%p [1][1]=%1.2f\n",fp,fp+1,fp+(N_part*size*1+1),*(fp+(N_part*size*1+1)));
	}*/
	if (rank==0) PrintMatrix((float*)partf, N_part, N_part*size, rank, "========= Force Matrix as initialized =========");
	
	//=================== Time Step Loop ===================//
	/* it: discrete time counter
	   dt: time step
	   itmax: itmax*dt ist endtime of simulation
	   L: length of system
	   dx: length of one cell in x-direction
	   ix: equivalent to cell ID
	*/
	const int itmax = 10;
	for (int it=0; it<itmax; it++) {
		int systolic_check = 0;
		struct particle part_tmp[N_part];
		memcpy(part_tmp, part_data, sizeof(struct particle)*N_part);
		/*for (int i=0; i<N_part; i++) {
			part_tmp[i].x0 = part_data[i].x0;
			part_tmp[i].v0 = part_data[i].v0;
			part_tmp[i].id = part_data[i].id;
		}*/
		//=================== Systolic Loop ===================//
		for (int i_sys=0; i_sys<size; i_sys++) {
			//Calculate force and sort into matrix
			for (int m=0; m<N_part; m++) {		//Go through particle data from other cells (including this cell itself)
				for(int n=0; n<N_part; n++)  {	//Go through particles in cell
					if (part_tmp[m].id != part_data[n].id) {
						float r = part_tmp[m].x0 - part_data[n].x0; //difference + length of one cell (without modulo condition)
						partf[n][part_tmp[m].id] = r;	//sgn(r)/(r*r);
						//forces[part_data[n].id][part_tmp[m].id] = -r;	//force matrix is symmetric
					} else
						partf[n][part_tmp[m].id] = 0;	//diagonal of force matrix
				}
				//Add up IDs to test correct working of systolic loop
				systolic_check += part_tmp[m].id;
			}
			//Send to next rank and receive from prior one
			MPI_Request request;
			int rank_send = (rank+1 > size-1) ? (0) : (rank+1);
			MPI_Isend( part_tmp, sizeof(struct particle)*N_part, MPI_CHAR, rank_send, 0, MPI_COMM_WORLD, &request);
			MPI_Status status;
			int rank_rcv = (rank-1 < 0) ? (size-1) : (rank-1);
			
				/*--------------------------------------------------------------------
				//Print Location Matrix to test whether these get transferred correctly
				int k;
				const int OUT_SR_LEN = 1024;
				char sr[OUT_SR_LEN],tmpsr[200];
				snprintf(sr, OUT_SR_LEN, "=== isys: %i, hello MPI user: from process = %i ===\nsend to rank:%i receive from rank:%i\n", i_sys, rank, rank_send, rank_rcv);
				for (k=0; k<N_part; k++) {
					sprintf(tmpsr,"Particle ID:%i x0:%1.2f\n\0", part_tmp[k].id, part_tmp[k].x0);
					strncat(sr, tmpsr, strlen(tmpsr));
				}
				printf("%s",sr);
				fflush(stdout);
				//------------------------------------------------------------------*/
			
			MPI_Recv( part_tmp, sizeof(struct particle)*N_part, MPI_CHAR, rank_rcv, 0, MPI_COMM_WORLD, &status);
			MPI_Wait (&request, &status);	//Wait until every sent package is received (necessary?)
		}
		//Sum up all the forces on one Particle (only N_part columns in the force matrix are filled in!)
		//The note in braces means summing up all force matrices results in the correct an complete one
		float netforces[N_part];
		for (int i=0; i<N_part; i++) {
			netforces[i]=0;
			for(int j=0; j<N_part*size; j++)
				netforces[i] += partf[i][j];
		}
		PrintMatrix((float*)netforces, 1, N_part, rank, "========= Net Forces =========");
		MPI_Barrier(MPI_COMM_WORLD);
		//Notice that IDs begin counting from 0, not 1, up to N_part*size-1, not N_part*size
		//printf("=>Systolic Loop check: %i should be %i",systolic_check,(N_part*size-1)*N_part*size/2);
		if (systolic_check != (N_part*size-1)*N_part*size/2)
			printf("SYSTOLIC LOOP INTEGRITY CHECK FAILED!\n");
		//print partial force matrix from every process
		//PrintMatrix((float*)partf, N_part, N_part*size, rank, "========= Partial Force Matrix =========");

		//Collect all partial force matrices
		float* forces[N_part*size][N_part*size];
		//if (rank==0)
		//	forces = malloc(sizeof(float)*N_part*size*N_part*size);
		//Buffersize should be same in receive and send! Don't know why that is even possible to choose idp
		MPI_Gather( partf, N_part*size*N_part, MPI_FLOAT, forces,
					N_part*size*N_part, MPI_FLOAT, 0, MPI_COMM_WORLD );
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			//sizeof doesn't seem to work with MPI_Types
			//printf("=> sizeof: MPI_FLOAT:%i, float:%i, MPI_BYTE:%i, MPI_CHAR:%i, MPI_LONG_DOUBLE:%i\n",sizeof(MPI_FLOAT),sizeof(float),sizeof(MPI_BYTE),sizeof(MPI_CHAR),sizeof(MPI_LONG_DOUBLE));
			PrintMatrix((float*)forces, N_part*size, N_part*size, rank, "========= Total! Force Matrix =========");
			//free(forces);
		}
	}
		
	//Finalize
	MPI_Finalize();
	free(cpu_name);
	free(part_data);
	return(0);
}
