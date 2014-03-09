#ifndef MD5_H
#define MD5_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>	//malloc,free,calloc
#include "cuda.h"
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }

typedef unsigned long int DWORD;	//has to be 32 bit
__host__ __device__ DWORD rol(DWORD num, int bits) {
	return (num << bits) | (num >> (32-bits));
}

__device__ unsigned int cudaStrlen(const char*);

class HASH {
	public:
		__align__(4) char data[16];
		__host__ __device__ HASH& operator= (HASH const &src);
		__host__ __device__ HASH();
		__host__ __device__ HASH( const char hex[32]);
		__host__ __device__ bool operator!=(const HASH &b) const;
		__host__ __device__ bool operator==(const HASH &b) const {
			return !(*this != b); }
		__host__ __device__ void print(const char* msg) const;
};

enum MD5_STRING_TYPES {LOWERCASE=1, UPPERCASE=2, LOWUP=3, DIGITS=4};//, 6:UPDIGITS, 5:LOWDIGITS, 7:LOWUPDIGITS};
class MD5_STRING {
	private:
		enum MD5_STRING_TYPES type;
	public:
		#ifdef __CUDA_ARCH__
			char data[64];
		#else
			char data[64];
		#endif
		__host__ __device__ void operator=(MD5_STRING &src);
		__host__ __device__ MD5_STRING( );
		__host__ __device__ MD5_STRING( const char* src, MD5_STRING_TYPES type );
		__host__ __device__ MD5_STRING( MD5_STRING &src );
		__host__ __device__ MD5_STRING( unsigned long int num );
		__host__ __device__ MD5_STRING& operator++();
		__host__ __device__ MD5_STRING operator++(int);
		__host__ __device__ void print() const;
};
MD5_STRING unmd5( const HASH hash );
HASH md5(const char* original_message);

MD5_STRING cudaUnmd5( const HASH hash );
__global__ void cudaMd5Kernel( const HASH hash, MD5_STRING *original_message);

#endif
