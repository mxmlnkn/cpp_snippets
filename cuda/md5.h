#ifndef MD5_H
#define MD5_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>	//malloc,free,calloc
#include "cuda.h"
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }
inline void gpuAssert(cudaError_t code, int line);

typedef unsigned long int DWORD;	//has to be 32 bit
__host__ __device__ DWORD rol(DWORD num, int bits) {
	return (num << bits) | (num >> (32-bits));
}

__device__ unsigned int cudaStrlen(const char*);

class HASH {
	public:
		char data[16];
		HASH& operator= (HASH const &src);
		HASH();
		HASH( const char hex[32]);
		bool operator!=(const HASH &b) const;
		bool operator==(const HASH &b) const {
			return !(*this != b); }
		void print(const char* msg);
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
		void operator=(MD5_STRING &src);
		MD5_STRING( );
		MD5_STRING( const char* src, MD5_STRING_TYPES type );
		MD5_STRING( MD5_STRING &src );
		MD5_STRING( unsigned long int num );
		MD5_STRING& operator++();
		MD5_STRING operator++(int);
};
MD5_STRING unmd5( const HASH hash );
HASH md5(const char* original_message);

MD5_STRING cudaUnmd5( const HASH hash );
__global__ void cudaMd5Kernel( const HASH hash, MD5_STRING *original_message);

#endif
