//compile with: nvcc matrixmult.c -o matrixmult --compiler-options "-std=c99"
//or under windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe matrixmult.cu -o matrixmult --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" --machine 32
//Notes: snprintf, %zu don't work with cl.exe >:O

#include <stdio.h>	//printf, fopen, fprintf
#include <string.h>	//strlen
#include <stdlib.h>	//malloc,free,calloc
#include "cuda.h"
#include <time.h>
#include "md5cuda.h"
#include <assert.h>
#include <stdint.h>	//uint32_t

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

		HASH& HASH::operator= (HASH const &src) {
			memcpy( this->data, src.data, 16 );
			return *this;
		}
		HASH::HASH( ) {
			//memset(this->data, 0, 16);
		}
		HASH::HASH( const char hex[32]) {
			for (int i=0; i<16; i++) {
				unsigned int a = hex[2*i];
				unsigned int b = hex[2*i+1];
				if (a >= 'a') a = a-'a'+10;
				if (b >= 'a') b = b-'a'+10;
				if (a >= '0') a = a-'0';
				if (b >= '0') b = b-'0';
				this->data[i] = a*16+b;
			}
		}
		void HASH::print(const char* msg = "Hash: ") const {
			printf("%s",msg);
			for (int i=0; i<16; i++) 
				printf("%02x", (unsigned char) this->data[i]);
			printf("\n");
		}
		int HASH::operator!=(const HASH &b) const {
			int equal = 0;
			for(int i=0; i<16; i++)
				equal += this->data[i] == b.data[i];
			return 16-equal;	//if all 16 chars are equal 0 is returned
		} 
		int HASH::operator==(const HASH &b) const {
			return !(*this != b);
		}
		
		void MD5_STRING::print() const {
			#ifdef __CUDA_ARCH__
				unsigned int len = cudaStrlen(this->data);
			#else
				unsigned int len = strlen(this->data);
			#endif
			printf("\nMessage Length: %u\nSize of unsigned long int: %lu"
			       "Bytes\n", len, sizeof(uint32_t) );
			printf("In Memory:");
			for (int i=0; i<64; i++) {
				if (i%16 == 0)
					printf("\n");
				else if (i%4 == 0)
					printf("\t");
				printf("%x\t", (unsigned char) this->data[i]);
			}
			printf("\n");
			return;
		}
		/**************************** Constructors ****************************/
		MD5_STRING::MD5_STRING( ) {
			memset(this->data, 0, 64);
			this->data[0] = (char)128;
		}
		MD5_STRING::MD5_STRING(const char *src,MD5_STRING_TYPES type=LOWERCASE){
			#ifdef __CUDA_ARCH__
				int len = cudaStrlen(src);
			#else
				int len = strlen(src);
			#endif
			memset( this->data,0,64 );
			memcpy( this->data, src, len%64 );
			this->data[len] = (char)128;
			this->data[56] = 8*len;
			this->type = type;
		}
		MD5_STRING::MD5_STRING( const MD5_STRING &src ) {
			(*this) = src;
		}
		MD5_STRING::MD5_STRING( uint64_t num ) {
			memset(this->data,0,64);
			//like operator++ the string is treated like little endian, meaning
			//the overflow bits of data[i]+num are summed up in data[i+1]
			/*Problem: Unlike normal numbers, we also want to cycle through 00*
			 *which would be 'aa' in this system!                             *
			 *Solution: determine string length from number and combinatorics *
			 * and use that lengtht to pad with "zeros" (e.g. 'a')            */
			uint64_t pow_tmp = 1;
			int base = 26;
			if (this->type == LOWERCASE)
				base = 26;
			for (int p=1; p<54; p++) {	//54 is max length of string
				pow_tmp *= base;
				if (p > 1)
					num -= pow_tmp/base;
				if (num < pow_tmp) {
					for (int k=0; k<p; k++) {
						//if this->type == LOWERCASE:
						this->data[k] = 'a' + num % base;
						num /= base;
					}
					this->data[p]   = (char)128;
					this->data[p+1] = 0;
					this->data[56]  = 8*p;
					break;
				}
			}
		}
		/****************************** Operators *****************************/
		void MD5_STRING::operator=(const MD5_STRING &src) {
			memcpy(this->data, src.data, 64);
		}
		MD5_STRING& MD5_STRING::operator++() {
			for (int i=0; i<54; i++) {
				if ( this->data[i] == (char)128 ) {
					this->data[i] = 'a';
					this->data[i+1] = (char)128;
					this->data[i+2] = 0;
					#ifdef __CUDA_ARCH__
						this->data[56] = 8*(cudaStrlen(this->data)-1);
					#else
						this->data[56] = 8*(strlen(this->data)-1);
					#endif
					break;
				} else if( ++(this->data[i]) > 'z' ) {
					//Todo: implement other kinds of strings
					//if (type-LOWERCASE != 0)
					this->data[i]   = 'a';
				} else
					break;
			}
			return *this;
		}
		MD5_STRING MD5_STRING::operator++(int) {
			MD5_STRING tmp(*this);
			++(*this);
			return tmp;
		}

/******************************************************************************
 *********************************** CPU MD5 **********************************
 ******************************************************************************/
MD5_STRING unmd5( const HASH hash ) {
	MD5_STRING original("", LOWERCASE);
	while( md5( (++original).data ) != hash ) {
		if (original.data[56] > 4)
			break;
	}
	return original;
}

HASH md5(const char* original_message) {
	#define a0 0x67452301
	#define b0 0xEFCDAB89
	#define c0 0x98BADCFE
	#define d0 0x10325476
	//uint32_t K[64]; for (int i=0; i<64; i++) K[i] = floor(abs(sin(i + 1)) * 2^32);
	
	/*original_message.data last bit of bit message must be 1 (or last char 128)
	 *in last 64-bit lies the bitlength of the message in little endian:
	 *Number 02AB13 is in memory: 13 AB 02                                    */
	const char* msg = original_message;

	uint32_t A = a0;
	uint32_t B = b0;
	uint32_t C = c0;
	uint32_t D = d0;
	uint32_t F;
	unsigned char g;

	//optimize by laying out all operations and memory accesses
	//in program code like done manually in md5.asm
	//#pragma unroll
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
		B += rol( A+F+K[i]+((uint32_t*)msg)[g], s[i] );
		A = temp;
		/*printf( "i:%i  A:%X B:%X, C:%X, D:%X, msg:%X, roladd:%X\n",i,A,B,C,D,
		        ((uint32_t*)msg)[g], rol( A+F+K[i]+((uint32_t*)msg)[g], s[i] ) );*/
	}
	
	HASH hash;
	((uint32_t*)(hash.data))[0] = a0+A;
	((uint32_t*)(hash.data))[1] = b0+B;
	/*if uint32_t accidentally 64 bit, then this is out of bound memory access   *
	 *Because of Compiler optimization, it seems, that the error only occurs, *
	 *if after writing to this memory, it is also read from. That's why the   *
	 *error only occured when reading hash.data[8+]!!!                        */
	((uint32_t*)(hash.data))[2] = c0+C;
	((uint32_t*)(hash.data))[3] = d0+D;
	return hash;
}

template <typename T> 
float Mean( T *array, unsigned int count ) {
	float sum = 0;
	for (int i=0; i<count; i++)
		sum += array[i];
	return sum/count;
}
template <typename T> 
float StdDev( T *array, unsigned int count ) {
	if (count <= 1)
		return 0;
	float sum = 0, sum2 = 0;
	for (int i=0; i<count; i++) {
		sum  += array[i];
		sum2 += array[i]*array[i];
	}
	return sqrt((abs(sum2-sum*sum/count))/(count-1));
}

/******************************************************************************
 *********************************** CUDA MD5 *********************************
 ******************************************************************************/
//defining s uint_32_t instead of char, reduces 611ms to 546ms! for 65535*1024*4
__constant__ uint32_t md5s[64];
__constant__ uint32_t md5K[64];
	 
MD5_STRING cudaUnmd5( const HASH hash ) {

	gpuErrchk( cudaMemcpyToSymbol(md5s, s, sizeof(s)));
	gpuErrchk( cudaMemcpyToSymbol(md5K, K, sizeof(K)));

	//Allocate slot for found raw text
	MD5_STRING original("Wrong Result!", LOWERCASE);
	original.data[56] = 0;	//wrong for md5-specifications!
	char *result;		//global memory slot for result
	gpuErrchk( cudaMalloc(&result, 64) );
	gpuErrchk( cudaMemcpy( result, original.data, 64, cudaMemcpyHostToDevice ));
	
	//will start 2^(16+8*4) processes, which will derive the message from their
	//block and thread IDs and hash it
	
	//#define BENCHMARK
	#ifdef BENCHMARK
	FILE *pTimeLog = fopen("Benchmark/times1024Threads.dat", "w");
	fprintf(pTimeLog, "#Blocks\tMean/ms\tStdDev/ms\n");
	for(int i=1; i<=1024; i++) {
		cudaEvent_t start,end;
		cudaEventCreate (&start); cudaEventCreate (&end);
		float elapsedTimes[5];
		for (int k=0; k<5; k++) {
			cudaEventRecord(start,0);
			cudaMd5Kernel<<<i,1024-128>>>(hash, result);
			cudaEventRecord(end,0); cudaEventSynchronize(end);
			gpuErrchk( cudaEventElapsedTime(&(elapsedTimes[k]),start,end) );
		}
		fprintf(pTimeLog, "%i\t%f\t%f\n",i,Mean(elapsedTimes,5),StdDev(elapsedTimes,5));
	}
	fclose(pTimeLog);
	#else
		cudaEvent_t start,end;
		cudaEventCreate (&start); cudaEventCreate (&end);
		#define RUNS 1
		float elapsedTimes[RUNS];
		for (int k=0; k<RUNS; k++) {
			cudaEventRecord(start,0);
			dim3 blocks(10*1024,1,1);
			dim3 threads=512;
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
	//while( md5( (++original).data ) != hash ) {
		//printf("%s ", original.data );
	//}
	return original;
}

__global__ void cudaMd5Kernel( const HASH hash, char *original_message, uint32_t respawn_number = 0) {
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
	
	HASH hashed;
	uint32_t &A = ((uint32_t*)(hashed.data))[0];
	uint32_t &B = ((uint32_t*)(hashed.data))[1];
	uint32_t &C = ((uint32_t*)(hashed.data))[2];
	uint32_t &D = ((uint32_t*)(hashed.data))[3];
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
		equal += ((uint64_t*)hashed.data)[i] == ((uint64_t*)hash.data)[i];
	if (equal == 2)
		memcpy(original_message, m, 64);
	return;
}

/******************************************************************************
 ************************************* MAIN ***********************************
 ******************************************************************************/
int main(int argc, char** args) {
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
	//char* sr = "test\0";
	printf("test->");
	MD5_STRING ToHash("test");
	HASH hash = md5(ToHash.data);
	hash.print();
	HASH hashtest = "098f6bcd4621d373cade4e832627b4f6";
	if (hashtest != hash)
		printf(" MD5-CPU-HASHER DOESN'T WORK!!!\n");
	
	const float stddevdbg[] = {1.0f, 2.0f,3,5,7};
	assert( sizeof(uint32_t)==4 ); //or else, MD5 Hash will be wrong and out of
	                            //bound errors will happen
	assert( round(Mean(stddevdbg,5)*1000) == round(3.6*1000) );
	assert( round(StdDev(stddevdbg,5)*1000) == round(sqrt(145)/5*1000) );
	
	MD5_STRING DbgInc;
	for (int i=0; i<=26*26*26+26*26+26; i++) {
		DbgInc++;
		MD5_STRING DbgCons(i);
		/*printf("%s=%s|",DbgCons.data,DbgInc.data);
		if (i % 26 == 0) printf("\n");*/
		if (strcmp(DbgCons.data,DbgInc.data) != 0)
			printf("Something wrong with num->string or Incrementor!\n");
	}
	
	/*for(int i=0; i<5; i++) {
		clock_t start_clocks = clock();
		MD5_STRING cracked = unmd5(HashToCrack);
		printf( "Cracked String: \"%s\" found in %f seconds (%li ticks per second)\n",
				cracked.data, (float)(clock()-start_clocks)/CLOCKS_PER_SEC, CLOCKS_PER_SEC );
	}*/
	
	HASH HashToCrack = "7a68f09bd992671bb3b19a5e70b7827e";	//5 chars testa
	MD5_STRING result = cudaUnmd5(HashToCrack);//HashToCrack);
	printf("Cracked String \"%s\" found!\n",result.data);
	return 0;
}

/*******************************************************************************
ToDo:
-for short messages the number of memory accesses could be reduced by adding a
 conditional check, because all the values after the short string are 0 or the
 length of the string!
	-> doesn't help, if conditions slows down process
	-> don't do this leck depending on external var, but depending on for loop
	   and threadid or argument :S
-More Charactersets
-Cancel all jobs when hash found
	-> would slow down kernel. Because of Watchdog we have to respawn kernel
	   anyway (on windows)

Learned:
-memcpy, memset work in __device__ and __host__! cudaMemcpy only on __host__!
-cudaMemcpyDeviceToDevice means from graphic card to other graphic card!
-Debug infos compiler directives slow down program by factor 16 !!!
-One error message occured only if something was in a certain if statement.
 That was, because the error was in the condition, and the compiler optimized
 the condition away, as it wouldn't result in anything anyway!
-"test"-Hash takes (44116.472656 ms, 44171.015625 ms) to crack in hzdr k20f with
 cudaMd5Kernel<<<60*1024, 768>>> 
-why is hzdr so slow?
	-> -g -G disables optimization, cuda-memcheck slows program down
-how to know how to align
	-> Cuda Programming Guide Section 5.1

Questions:
-how many blocks are physically calculated in parallel? Should be deducable,
 as up to this block count the execution time should not increase!
 .200ms for one md5 run -> 256 time measurements in ~50s possible -> graph
	-> 14 SM * MaxThreadsPerMP
-MaxThreads seems to be exactly 896 in hzdr even though deviceprop says 1024
	-> Calculate total RegistersPerMP !
	
Notes:
-md5("test") should be 098f6bcd 4621d373 cade4e83 2627b4f6
-test is string number 337811

Benchmarks at Home:
-const memory ~242ms vs 492ms local memory
-crack be73... hash -> ~900ms

Benchmark Tesla C2070
-10240 blocks, 512 threads launched 256 times -> 921 ms
-(10*1024,64,1) blocks 512 threads takes 14s ... it's just wrong :(
 surely because for the above he actually finds the string at m~100 and breaks
 the loop

=========== Device Number 0 ===========
Device name            : Tesla C2070
Global Memory          : 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Warp Size              : 32 Threads
Multiprocessors (MP)   : 14
Max Threads per MP     : 1536
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size          : (65535,65535,65535)
Clock Rate             : 1147000 kHz
Computability          : 2.0

=========== Device Number 1 ===========
Device name            : Tesla C2070
Global Memory          : 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Warp Size              : 32 Threads
Multiprocessors (MP)   : 14
Max Threads per MP     : 1536
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size          : (65535,65535,65535)
Clock Rate             : 1147000 kHz
Computability          : 2.0

=========== Device Number 2 ===========
Device name            : Tesla C2070
Global Memory          : 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Warp Size              : 32 Threads
Multiprocessors (MP)   : 14
Max Threads per MP     : 1536
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size          : (65535,65535,65535)
Clock Rate             : 1147000 kHz
Computability          : 2.0

=========== Device Number 3 ===========
Device name            : Tesla C2070
Device name            : Tesla C2070
Global Memory          : 5636554752 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 32768
Warp Size              : 32 Threads
Multiprocessors (MP)   : 14
Max Threads per MP     : 1536
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size          : (65535,65535,65535)
Clock Rate             : 1147000 kHz
Computability          : 2.0


@Home
=========== Device Number 0 ===========
Device name  : GeForce GTX 760
Global Memory: 2147483648 Bytes
Shared Memory per Block: 49152 Bytes
Registers per Block    : 65536
Warp Size              : 32 Threads
Multiprocessors (MP)   : 6
Max Threads per MP     : 2048
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size: (2147483647,65535,65535)
Clock Rate   : 1150000 kHz
Computability: 3.0
*/
