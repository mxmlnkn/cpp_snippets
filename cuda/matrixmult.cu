//compile with: nvcc matrixmult.c -o matrixmult --compiler-options "-std=c99"
//or under windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe matrixmult.cu -o matrixmult --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" --machine 32
//Notes: snprintf, %zu don't work with cl.exe >:O

#include <stdio.h>
#include <string.h>
#include <stdlib.h>	//malloc,free,calloc
#include "cuda.h"

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line) {
   if (code != cudaSuccess)
      printf("GPUassert at Line %d: %s\n", line, cudaGetErrorString(code));
}

void PrintMatrix(float *matrix, int stride, int xmax, int ymax, const char* headline) {
	const int OUT_SR_LEN = 1024;
	char sr[OUT_SR_LEN],tmpsr[200];
	sprintf(sr, "\n%s\n\n", headline);
	for (int m=0; m<xmax; m++) {
		for(int n=0; n<ymax; n++) {
			sprintf(tmpsr,"%1.2f \0", matrix[m*stride+n]);
			strcat(sr, tmpsr);
		}
		sprintf(tmpsr,"\n\0");
		strncat(sr, tmpsr, strlen(tmpsr));
	}
	printf("%s",sr);
}


void SquareMatrixMult(int rows1, int cols1, float* m1,
                      int rows2, int cols2, float* m2, float** m3) {
	if (cols1 != rows2)
		return;
	const int rows3 = rows1, cols3 = cols2, scpsize = cols1;
	*m3 = (float*)realloc( *m3, rows3*cols3* sizeof(float) );
	for (int i=0; i<rows3; i++)
		for (int j=0; j<cols3; j++) {
			float scp = 0;
			for (int k=0; k<scpsize; k++) 
				scp += m1[i*rows1+k] * m2[k*rows2+j];
			(*m3)[i*rows3+j] = scp;
		}
	return;
}


__global__ void cudaKernelSquareMatrixMult( int rows1, 
                float* cm1, int rows2, float* cm2, float* cm3) {
	const int rows3 = rows1, scpsize = rows2;
	const int i = threadIdx.x;
	const int j = threadIdx.y;
	float scp = 0;
	for (int k=0; k<scpsize; k++) 
		scp += cm1[i*rows1+k] * cm2[k*rows2+j];
	cm3[i*rows3+j] = scp;
}

void cudaSquareMatrixMult( int rows1, int cols1,
                float* m1, int rows2, int cols2,
                float* m2, float** m3) {
	if (cols1!=rows2)
		return;
	const int rows3 = rows1, cols3 = cols2;
	float *cm1,*cm2,*cm3;
	const int sz_m1 = rows1*cols1* sizeof(float);
	
	printf("cudaMalloc Matrix one with %i Bytes\n",sz_m1);
	gpuErrchk( cudaMalloc( (void**)&cm1, sz_m1 ) );
	printf("cudaMemcpy\n");
	gpuErrchk( cudaMemcpy( cm1, m1, sz_m1, cudaMemcpyHostToDevice ) );
	const int sz_m2 = rows2*cols2* sizeof(float);
	cudaMalloc( &cm2, sz_m2 );
	cudaMemcpy( cm2, m2, sz_m2, cudaMemcpyHostToDevice );
	const int sz_m3 = rows1*cols2* sizeof(float);
	cudaMalloc( &cm3, sz_m3 );
	cudaMemset( cm3, 0, sz_m3 );
	
	dim3 blocks(1,1);
	dim3 threads(rows3,cols3);
	
	printf("Start CUDA-Kernel\n");
	//similar to mpirun -n blocks*threads
	cudaKernelSquareMatrixMult<<<blocks,threads>>>( rows1, cm1, rows2, cm2, cm3 );
	/*is it faster to write many identical arguments first to global 
	 *memory and read from that global memory from the kernel ?
	 *Folllowing would also work, but means something a bit different (better)
	 *rows3 blocks with each cols3 threads will be launched
	 *As the number of threads is limited to 512 (architecture independent?)
	 *this method can calculate bigger matrices
	 *cudaKernelSquareMatrixMult<<<rows3,cols3>>>( rows1, cm1, rows2, cm2, cm3 ); */
	printf("CUDA-Kernel terminated\n");

	*m3 = (float*)realloc(*m3, sz_m3);
	cudaMemcpy( m1, cm3, rows3*cols3*sizeof(float), cudaMemcpyDeviceToHost );
	cudaFree(m1);
	cudaFree(m2);
	cudaFree(m3);
}

int main(int argc, char** args) {
	const int size = 32;
	float m1[size][size];
	float m2[size][size];
	float* m3 = NULL;
	
	for (int i=0; i<size; i++)
		for (int j=0; j<size; j++) {
			//int cols = size;
			m1[i][j] = i;
			m2[i][j] = i;
		}
	
	printf("Getting Device Informations. As this is the first command, "
	       "it can take ca.30s, because the GPU must be initialized.\n");
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Devices supporting CUDA: %i\n",deviceCount);
	for (int device = 0; device < deviceCount; device++) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		if (device == 0 & deviceProp.major == 9999 && deviceProp.minor == 9999)
			printf("There is no device supporting CUDA.\n");
		printf("\n=========== Device Number %i ===========\n"
		       "Device name  : %s\n"
		       "Global Memory: %lu Bytes\n"
		       "Shared Memory per Block: %lu Bytes\n"
		       "Registers per Block    : %i\n"
		       "Max Threads per Block  : %i\n"
		       "Max threads dimension  : (%i,%i,%i)\n"
		       "Max Grid Size: (%i,%i,%i)\n"
		       "Clock Rate   : %i kHz\n"
		       "Computability: %i.%i\n", device, deviceProp.name,
		       deviceProp.totalGlobalMem, deviceProp.sharedMemPerBlock,
		       deviceProp.regsPerBlock, deviceProp.maxThreadsPerBlock,
		       deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
		       deviceProp.maxThreadsDim[2], deviceProp.maxGridSize[0],
		       deviceProp.maxGridSize[1], deviceProp.maxGridSize[2],
		       deviceProp.clockRate, deviceProp.major, deviceProp.minor);
	}
	/*size_t size = 1024 * sizeof(float);
	cudaSetDevice(0);
	float* p0;
	cudaMalloc(&p0, size);
	// Allocate memory on device 0
	MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
	cudaSetDevice(1);
	// Set device 1 as current
	float* p1;
	cudaMalloc(&p1, size);	// Allocate memory on device 1
	MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
	*/
	cudaEvent_t start,end;
	cudaEventCreate (&start);
	cudaEventCreate (&end);
	cudaEventRecord(start,0);
	
	//PrintMatrix((float*)m1, size,size,size, "Matrix 1 initialized");
	//SquareMatrixMult( size,size, (float*)m1, size,size,(float*)m2, &m3 );
	//PrintMatrix(m3, size,size,size, "Result Matrix");
	cudaSquareMatrixMult( size,size, (float*)m1, size,size,(float*)m2, &m3 );
	PrintMatrix(m3, size, 5,5, "Result Matrix");
	
	//FINISH RECORDING
	gpuErrchk( cudaEventRecord(end,0) );
	cudaEventSynchronize(end);
	float elapsedTime;
	gpuErrchk( cudaEventElapsedTime(&elapsedTime,start,end)) ;
	printf("Elapsed Time: %f ms\n",elapsedTime);
	return 0;
}


/*
===================== Device Number 0 =====================
Device name  : Tesla C2070
Global Memory: 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size: (65535,65535,65535)
Clock Rate   : 1147000 kHz
Major: 2 Minor: 0

===================== Device Number 1 =====================
Device name  : Tesla C2070
Global Memory: 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size: (65535,65535,65535)
Clock Rate   : 1147000 kHz
Major: 2 Minor: 0

===================== Device Number 2 =====================
Device name  : Tesla C2070
Global Memory: 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size: (65535,65535,65535)
Clock Rate   : 1147000 kHz
Major: 2 Minor: 0

===================== Device Number 3 =====================
Device name  : Tesla C2070
Global Memory: 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size: (65535,65535,65535)
Clock Rate   : 1147000 kHz
Major: 2 Minor: 0

@Home
=========== Device Number 0 ===========
Device name  : GeForce GTX 760
Global Memory: 2147483648 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 65536
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size: (2147483647,65535,65535)
Clock Rate   : 1150000 kHz
Computability: 3.0
*/
