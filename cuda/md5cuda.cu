//compile with: nvcc matrixmult.c -o matrixmult --compiler-options "-std=c99"
//or under windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe matrixmult.cu -o matrixmult --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" --machine 32
//Notes: snprintf, %zu don't work with cl.exe >:O

#include <stdio.h>	//printf, fopen, fprintf
#include <string.h>	//strlen
#include <stdlib.h>	//malloc,free,calloc
#include "cuda.h"
#include <time.h>
#include "md5cuda.h"

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
			memset(this->data, 0, 16);
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
		bool HASH::operator!=(const HASH &b) const {
			int equal = 0;
			for(int i=0; i<16; i++)
				equal += this->data[i]== b.data[i];
			return equal<16;
		}
		
		void MD5_STRING::print() const {
			#ifdef __CUDA_ARCH__
				int len = cudaStrlen(this->data);
			#else
				int len = strlen(this->data);
			#endif
			printf("\nMessage Length: %lu\nSize of unsigned long int: %lu Bytes\n", len, sizeof(DWORD) );
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
		MD5_STRING::MD5_STRING( MD5_STRING &src ) {
			(*this) = src;
		}
		MD5_STRING::MD5_STRING( unsigned long int num ) {
			memset(this->data,0,64);
			//like operator++ the string is treated like little endian, meaning
			//the overflow bits of data[i]+num are summed up in data[i+1]
			/*Problem: Unlike normal numbers, we also want to cycle through 00*
			 *which would be 'aa' in this system!                             *
			 *Solution: determine string length from number and combinatorics *
			 * and use that lengtht to pad with "zeros" (e.g. 'a')            */
			int pow_tmp = 1;
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
		void MD5_STRING::operator=(MD5_STRING &src) {
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
	const DWORD s[64] = 
		{ 7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
		  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
		  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
		  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21 };
	const DWORD K[64] =
		{3614090360, 3905402710,  606105819, 3250441966,
		 4118548399, 1200080426, 2821735955, 4249261313,
		 1770035416, 2336552879, 4294925233, 2304563134,
		 1804603682, 4254626195, 2792965006, 1236535329,
		 4129170786, 3225465664,  643717713, 3921069994,
		 3593408605,   38016083, 3634488961, 3889429448,
		  568446438, 3275163606, 4107603335, 1163531501,
		 2850285829, 4243563512, 1735328473, 2368359562,
		 4294588738, 2272392833, 1839030562, 4259657740,
		 2763975236, 1272893353, 4139469664, 3200236656,
		  681279174, 3936430074, 3572445317,   76029189,
		 3654602809, 3873151461,  530742520, 3299628645,
		 4096336452, 1126891415, 2878612391, 4237533241,
		 1700485571, 2399980690, 4293915773, 2240044497, 
		 1873313359, 4264355552, 2734768916, 1309151649,
		 4149444226, 3174756917,  718787259, 3951481745};
	#define a0 0x67452301
	#define b0 0xEFCDAB89
	#define c0 0x98BADCFE
	#define d0 0x10325476
	//DWORD K[64]; for (int i=0; i<64; i++) K[i] = floor(abs(sin(i + 1)) * 2^32);
	
	/*original_message.data last bit of bit message must be 1 (or last char 128)
	 *in last 64-bit lies the bitlength of the message in little endian:
	 *Number 02AB13 is in memory: 13 AB 02                                    */
	const char* msg = original_message;

	DWORD A = a0;
	DWORD B = b0;
	DWORD C = c0;
	DWORD D = d0;
	DWORD F,g;

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
		DWORD temp = D;
		D = C;
		C = B;
		B += rol( A+F+K[i]+((DWORD*)msg)[g], s[i] );
		A = temp;
		/*printf( "i:%i  A:%X B:%X, C:%X, D:%X, msg:%X, roladd:%X\n",i,A,B,C,D,
		        ((DWORD*)msg)[g], rol( A+F+K[i]+((DWORD*)msg)[g], s[i] ) );*/
	}
	HASH hash;
	((DWORD*)(hash.data))[0] = a0+A;
	((DWORD*)(hash.data))[1] = b0+B;
	((DWORD*)(hash.data))[2] = c0+C;
	((DWORD*)(hash.data))[3] = d0+D;
	return hash;
}

template <typename T> 
double Mean( T *array, int count ) {
	double sum = 0;
	for (int i=0; i<count; i++)
		sum += array[i];
	return sum/count;
}
template <typename T> 
double StdDev( T *array, unsigned int count ) {
	if (count <= 1)
		return 0;
	double sum = 0, sum2 = 0;
	for (int i=0; i<count; i++) {
		sum  += array[i];
		sum2 += array[i]*array[i];
	}
	return sqrt((abs(sum2-sum*sum)/count)/(count-1));
}

/******************************************************************************
 *********************************** CUDA MD5 *********************************
 ******************************************************************************/
MD5_STRING cudaUnmd5( const HASH hash ) {
	MD5_STRING original("wrongtest", LOWERCASE);
	
	MD5_STRING *result;		//global memory slot for result
	gpuErrchk( cudaMalloc(&result, sizeof(MD5_STRING)) );
	cudaMemcpy( result, &original, sizeof(MD5_STRING), cudaMemcpyHostToDevice );
	
	//will start 2^(16+8*4) processes, which will derive the message from their
	//block and thread IDs and hash it
	
	#ifdef BENCHMARK
	FILE *pTimeLog = fopen("Benchmark/times33Block.dat", "w");
	fprintf(pTimeLog, "Blocks\tMean/ms\tStdDev/ms\n");
	for(int i=1; i<=1024; i++) {
		cudaEvent_t start,end;
		cudaEventCreate (&start); cudaEventCreate (&end);
		float elapsedTimes[5];
		for (int k=0; k<5; k++) {
			cudaEventRecord(start,0);
				cudaMd5Kernel<<<33,i>>>(hash, result);
			cudaEventRecord(end,0); cudaEventSynchronize(end);
			gpuErrchk( cudaEventElapsedTime(&(elapsedTimes[k]),start,end) );
		}
		fprintf(pTimeLog, "%i\t%f\t%f\n",i,Mean(elapsedTimes,5),StdDev(elapsedTimes,5));
	}
	fclose(pTimeLog);
	#else
		cudaMd5Kernel<<<65535,1024>>>(hash, result);
	#endif
	
	cudaMemcpy( &original, result, sizeof(MD5_STRING), cudaMemcpyDeviceToHost );
	gpuErrchk( cudaFree(result) );
	//while( md5( (++original).data ) != hash ) {
		//printf("%s ", original.data );
	//}
	return original;
}

__global__ void cudaMd5Kernel( const HASH hash, MD5_STRING *original_message) {
	const DWORD s[64] = 
		{ 7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
		  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
		  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
		  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21 };
	const DWORD K[64] =
		{3614090360, 3905402710,  606105819, 3250441966,
		 4118548399, 1200080426, 2821735955, 4249261313,
		 1770035416, 2336552879, 4294925233, 2304563134,
		 1804603682, 4254626195, 2792965006, 1236535329,
		 4129170786, 3225465664,  643717713, 3921069994,
		 3593408605,   38016083, 3634488961, 3889429448,
		  568446438, 3275163606, 4107603335, 1163531501,
		 2850285829, 4243563512, 1735328473, 2368359562,
		 4294588738, 2272392833, 1839030562, 4259657740,
		 2763975236, 1272893353, 4139469664, 3200236656,
		  681279174, 3936430074, 3572445317,   76029189,
		 3654602809, 3873151461,  530742520, 3299628645,
		 4096336452, 1126891415, 2878612391, 4237533241,
		 1700485571, 2399980690, 4293915773, 2240044497, 
		 1873313359, 4264355552, 2734768916, 1309151649,
		 4149444226, 3174756917,  718787259, 3951481745};
	#define a0 0x67452301
	#define b0 0xEFCDAB89
	#define c0 0x98BADCFE
	#define d0 0x10325476

	MD5_STRING message(blockIdx.x*blockDim.x + threadIdx.x);
	char *msg = message.data;

	DWORD A = a0;
	DWORD B = b0;
	DWORD C = c0;
	DWORD D = d0;
	DWORD F,g;

	#pragma unroll
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
		DWORD temp = D;
		D = C;
		C = B;
		B += rol( A+F+K[i]+((DWORD*)msg)[g], s[i] );
		A = temp;
	}
	HASH hashed;
	((DWORD*)(hashed.data))[0] = a0+A;
	((DWORD*)(hashed.data))[1] = b0+B;
	((DWORD*)(hashed.data))[2] = c0+C;
	((DWORD*)(hashed.data))[3] = d0+D;
	
	//if (!(hashed!=hash) || (blockIdx.x==338 && threadIdx.x==123)) {
	/*if (!(hashed!=hash) || (blockIdx.x==784 && threadIdx.x==395)) {
		printf("Found Hash (Thread ID: %i): %s\n", threadIdx.x, message.data);
		printf("Hash to Find: "); hash.print();
		printf("Hashed: "); hashed.print();
		printf("From Message: \n"); message.print();
	}*/
	if (hashed==hash) 
		*original_message = message;
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
		       "Device name  : %s\n"
		       "Global Memory: %lu Bytes\n"
		       "Shared Memory per Block: %lu Bytes\n"
		       "Registers per Block    : %i\n"
			   "Warp Size              : %i Threads\n"
			   "Multiprocessors (MP)   : %i\n"
			   "Max Threads per MP     : %i\n"
		       "Max Threads per Block  : %i\n"
		       "Max threads dimension  : (%i,%i,%i)\n"
		       "Max Grid Size: (%i,%i,%i)\n"
		       "Clock Rate   : %i kHz\n"
		       "Computability: %i.%i\n", device, deviceProp.name,
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
	MD5_STRING ToHash("test");
	HASH hash = md5(ToHash.data);
	hash.print();
	HASH HashToCrack = "098f6bcd4621d373cade4e832627b4f6";
	
	MD5_STRING DbgInc;
	for (int i=0; i<=26*26*26+26*26+26; i++) {
		DbgInc++;
		MD5_STRING DbgCons(i);
		/*printf("%s=%s|",DbgCons.data,DbgInc.data);
		if (i % 26 == 0) printf("\n");*/
		if (strcmp(DbgCons.data,DbgInc.data) != 0)
			printf("Something wrong with num->string or Incrementor!\n");
	}
	
	for(int i=0; i<5; i++) {
		clock_t start_clocks = clock();
		MD5_STRING cracked = unmd5(HashToCrack);
		printf( "Cracked String: \"%s\" found in %f seconds (%i ticks per second)\n",
				cracked.data, (float)(clock()-start_clocks)/CLOCKS_PER_SEC, CLOCKS_PER_SEC );
	}
	
	HashToCrack = "7a68f09bd992671bb3b19a5e70b7827e";
	MD5_STRING result = cudaUnmd5(HashToCrack);
	printf("Cracked String \"%s\" found!\n",result.data);
	return 0;
}

/*******************************************************************************
ToDo:
-for short messages the number of memory accesses could be reduced by adding a
 conditional check, because all the values after the short string are 0 or the
 length of the string!
-More Charactersets, optimize with unrolling, memory choice
-Cancel all jobs when hash found
-increase possible length by implementing multiply: MD5_MESSAGE *= 343;
-or find another way to get a succesive message from threadIdx and blockIdx

Learned:
-Program fatal error: memcpy instead of memset written!
-memcpy, memset work in __device__ and __host__! cudaMemcpy only on __host__!
-cudaMemcpyDeviceToDevice means from graphic card to other graphic card!

Questions:
-how many blocks are physically calculated in paralleh? Should be deducable,
 as up to this block count the execution time should not increase!
 .200ms for one md5 run -> 256 time measurements in ~50s possible -> graph
 
Notes:
-md5("test") should be 098f6bcd 4621d373 cade4e83 2627b4f6
-test is string number 337811
-input num for threads is 32 bit => max reachable string: vxlrmxn 

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
Warp Size              : 32 Threads
Multiprocessors (MP)   : 6
Max Threads per MP     : 2048
Max Threads per Block  : 1024
Max threads dimension  : (1024,1024,64)
Max Grid Size: (2147483647,65535,65535)
Clock Rate   : 1150000 kHz
Computability: 3.0
*/
