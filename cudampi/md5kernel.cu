//compile with: nvcc matrixmult.c -o matrixmult --compiler-options "-std=c99"
//or under windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe matrixmult.cu -o matrixmult --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" --machine 32
//Notes: snprintf, %zu don't work with cl.exe >:O

#include <stdio.h>	//printf, fopen, fprintf
#include <string.h>	//strlen
#include <stdlib.h>	//malloc,free,calloc
#include "cuda.h"
#include "statistics.hpp"
#include "md5kernel.hpp"
#include <stdint.h>	//uint32_t

const uint32_t s[64] = 
	{ 7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
	  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
	  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
	  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21 };
const uint32_t K[64] =
	{3614090360u, 3905402710u,  606105819u, 3250441966u,
	 4118548399u, 1200080426u, 2821735955u, 4249261313u,
	 1770035416u, 2336552879u, 4294925233u, 2304563134u,
	 1804603682u, 4254626195u, 2792965006u, 1236535329u,
	 4129170786u, 3225465664u,  643717713u, 3921069994u,
	 3593408605u,   38016083u, 3634488961u, 3889429448u,
	  568446438u, 3275163606u, 4107603335u, 1163531501u,
	 2850285829u, 4243563512u, 1735328473u, 2368359562u,
	 4294588738u, 2272392833u, 1839030562u, 4259657740u,
	 2763975236u, 1272893353u, 4139469664u, 3200236656u,
	  681279174u, 3936430074u, 3572445317u,   76029189u,
	 3654602809u, 3873151461u,  530742520u, 3299628645u,
	 4096336452u, 1126891415u, 2878612391u, 4237533241u,
	 1700485571u, 2399980690u, 4293915773u, 2240044497u, 
	 1873313359u, 4264355552u, 2734768916u, 1309151649u,
	 4149444226u, 3174756917u,  718787259u, 3951481745u };

__host__ __device__ uint32_t rol(uint32_t num, int bits) {
	return (num << bits) | (num >> (32-bits));
}

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line) {
   if (code != cudaSuccess)
      printf("GPUassert at Line %d: %s\n", line, cudaGetErrorString(code));
}

__device__ unsigned int cudaStrlen(const char *s) {
	unsigned int i;
	for(i=0; i<2^31; i++) {
		if (s[i] == 0)
			break;
	}
	return i;
}
__global__ void cudaMd5Kernel( const hashstruct hash, char *original_message, uint32_t respawn_number = 0);

/******************************************************************************
 *********************************** CUDA MD5 *********************************
 ******************************************************************************/
//defining s uint_32_t instead of char, reduces 611ms to 546ms! for 65535*1024*4
__constant__ uint32_t md5s[64];
__constant__ uint32_t md5K[64];
	 
md5struct cudaUnmd5( const hashstruct hash ) {
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
		       "Device name            : %s\n"
		       "Global Memory          : %lu Bytes\n"
		       "Shared Memory per Block: %lu Bytes\n"
		       "Registers per Block    : %i\n"
			   "Warp Size              : %i Threads\n"
			   "Multiprocessors (MP)   : %i\n"
			   "Max Threads per MP     : %i\n"
		       "Max Threads per Block  : %i\n"
		       "Max threads dimension  : (%i,%i,%i)\n"
		       "Max Grid Size          : (%i,%i,%i)\n"
		       "Clock Rate             : %i kHz\n"
		       "Computability          : %i.%i\n", device, deviceProp.name,
		       deviceProp.totalGlobalMem, deviceProp.sharedMemPerBlock,
		       deviceProp.regsPerBlock, deviceProp.warpSize, 
			   deviceProp.multiProcessorCount, deviceProp.maxThreadsPerMultiProcessor,
			   deviceProp.maxThreadsPerBlock,
		       deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
		       deviceProp.maxThreadsDim[2], deviceProp.maxGridSize[0],
		       deviceProp.maxGridSize[1], deviceProp.maxGridSize[2],
		       deviceProp.clockRate, deviceProp.major, deviceProp.minor);
	}

	gpuErrchk( cudaMemcpyToSymbol(md5s, s, sizeof(s)));
	gpuErrchk( cudaMemcpyToSymbol(md5K, K, sizeof(K)));

	//Allocate slot for found raw text
	md5struct original;
	memset(original.data, 0, 64);
	//global memory slot for result
	char *result;
	gpuErrchk( cudaMalloc(&result, 64) );
	gpuErrchk( cudaMemcpy( result, original.data, 64, cudaMemcpyHostToDevice ));
	
	//will start 2^(16+8*4) processes, which will derive the message from their
	//block and thread IDs and hash it
	
	#define BENCHMARK
	#ifdef BENCHMARK
	FILE *pTimeLog = fopen("times.dat", "w");
	fprintf(pTimeLog, "#Benchmark done 5 times per Data Entry\n");
	fprintf(pTimeLog, "#Blocks\tThreads\tMean/ms\tStdDev/ms\n");
	for(int i=1; i<=1024; i++) {
		for(int j=1; j<=1024; j++) {
			cudaEvent_t start,end;
			cudaEventCreate (&start); cudaEventCreate (&end);
			const int RUNS = 5;
			float elapsedTimes[RUNS];
			for (int k=0; k<RUNS; k++) {
				cudaEventRecord(start,0);
				cudaMd5Kernel<<<i,j>>>(hash, result);
				cudaEventRecord(end,0); cudaEventSynchronize(end);
				gpuErrchk( cudaEventElapsedTime(&(elapsedTimes[k]),start,end) );
			}
			fprintf(pTimeLog, "%i\t%i\t%f\t%f\n",i,j,Mean(elapsedTimes,RUNS),StdDev(elapsedTimes,RUNS));
			if ( (i % 32 == 0) && (j % 512 == 0) ) {
				printf("Blocks: %i, Threads: %i, Time: %f +- %f\n",i,j,Mean(elapsedTimes,RUNS),StdDev(elapsedTimes,RUNS));
				fflush(pTimeLog);
			}
		}
	}
	fclose(pTimeLog);
	#else
		cudaEvent_t start,end;
		cudaEventCreate (&start); cudaEventCreate (&end);
		#define RUNS 1
		float elapsedTimes[RUNS];
		for (int k=0; k<RUNS; k++) {
			cudaEventRecord(start,0);
			dim3 blocks(1024,1,1);
			dim3 threads=768;
				for (int m=0; m<256; m++) {
					cudaMd5Kernel<<<blocks,threads>>>(hash, result, m);
					gpuErrchk( cudaMemcpy( original.data, result, 64, cudaMemcpyDeviceToHost ) );
					gpuErrchk( cudaPeekAtLastError()   );
					gpuErrchk( cudaDeviceSynchronize() );
					if (original.data[56] != 0)
						break;
					//printf("%i: Current String to test: %s\n", m, MD5_STRING((uint64_t)m*(uint64_t)blocks.x*threads.x).data);
				}
			cudaEventRecord(end,0); cudaEventSynchronize(end);
			gpuErrchk( cudaEventElapsedTime(&(elapsedTimes[k]),start,end) );
		}
		printf( "65535 Blocks, 1024 Threads, Time/ms:%f +- %f\n",
		        Mean(elapsedTimes,RUNS),StdDev(elapsedTimes,RUNS));
	#endif
	
	memset(original.data, 0, 64);
	gpuErrchk( cudaMemcpy( original.data, result, 64, cudaMemcpyDeviceToHost ) );
	gpuErrchk( cudaFree(result) );
	return original;
}

__global__ void cudaMd5Kernel( const hashstruct hash, char *original_message, uint32_t respawn_number) {
	#define a0 0x67452301
	#define b0 0xEFCDAB89
	#define c0 0x98BADCFE
	#define d0 0x10325476
	
	//actually only using MD5_STRING because of it's constructor
	//MD5_STRING message(blockIdx.x*blockDim.x + threadIdx.x);
	char m[64];
	//uin64_t instead of uint32_t saves 8ms (536->528) Oo? (thought it was 32bit)
	uint64_t num = (uint64_t)respawn_number*(uint64_t)gridDim.x*(uint64_t)blockDim.x + blockIdx.x*blockDim.x + threadIdx.x;
	//like operator++ the string is treated like little endian, meaning
	//the overflow bits of data[i]+num are summed up in data[i+1]
	/*Problem: Unlike normal numbers, we also want to cycle through 00*
	 *which would be 'aa' in this system!                             *
	 *Solution: determine string length from number and combinatorics *
	 * and use that lengtht to pad with "zeros" (e.g. 'a')            */
	uint64_t pow_tmp = 1;	//64bit suffices for 13 chars in lowercase
	#define BASE 26
	/*if (this->type == LOWERCASE)
		base = 26;*/
	//including the constructor directly here saves 10ms (546ms->536ms) 65535*1024*4
	memset(m,0,64);
	for (int p=1; p<54; p++) {	//54 is max length of string
		pow_tmp *= BASE;
		if (p > 1)
			num -= pow_tmp/BASE;
		if (num < pow_tmp) {
			for (int k=0; k<p; k++) {
				//if this->type == LOWERCASE:
				m[k] = 'a' + num % BASE;
				num /= BASE;
			}
			m[p]   = (char)128;
			m[56]  = 8*p;
			break;
		}
	}
	
	uint32_t hashed[4];
	uint32_t &A = hashed[0];
	uint32_t &B = hashed[1];
	uint32_t &C = hashed[2];
	uint32_t &D = hashed[3];
	A = a0; B = b0; C = c0; D = d0;
	uint32_t F,g;
	
	#pragma unroll	//saves 100ms(267->170) per 65535*1024 hashes
	for (int i=0; i<64; i++) {
		if (i<16) {
			//F = (B & C) | ((~B) & D);
			F = D ^ (B & (C ^ D));
			g = i;
		} else if(i<32) {
			//F = (B & D) | ((~D) & C);
			F = C ^ (D & (B ^ C));
			g = (5*i+1) % 16;
		} else if(i<48) {
			F = (B ^ C) ^ D;
			g = (3*i+5) % 16;
		} else {
			F = C ^ (B | ~D);
			g = (7*i) % 16;
		}
		uint32_t temp = D;
		D = C;
		C = B;
		//check (g > m[56]) ? (0) : (...) makes program slower by 90ms
		B += rol( A + F + md5K[i] + ((uint32_t*)m)[g], md5s[i] );
		A = temp;
	}
	A+=a0; B+=b0; C+=c0; D+=d0;
	
	//if (!(hashed!=hash) || (blockIdx.x==338 && threadIdx.x==123)) {
	/*if (!(hashed!=hash) || (blockIdx.x==784 && threadIdx.x==395)) {
		printf("Found Hash (Thread ID: %i): %s\n", threadIdx.x, message.data);
		printf("Hash to Find: "); hash.print();
		printf("Hashed: "); hashed.print();
		printf("From Message: \n"); message.print();
	}*/
	
	//comparing 32bits saves 70/4ms per 65535*1024 hashes compared to bytewise!
	//64bits saves 1ms compared to 32bits
	uint32_t equal = 0;
	for(int i=0; i<2; i++)
		equal += ((uint64_t*)hashed)[i] == ((uint64_t*)hash.data)[i];
	if (equal == 2)
		memcpy(original_message, m, 64);
	return;
}
