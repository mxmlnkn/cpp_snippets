#ifndef MD5_H
#define MD5_H

#include "cuda.h"
#include <stdint.h>	//uint32_t

#define gpuErrchk(ans) { gpuAssert((ans), __LINE__); }

__host__ __device__ uint32_t rol(uint32_t num, int bits) {
	return (num << bits) | (num >> (32-bits));
}

__device__ unsigned int cudaStrlen(const char*);

class HASH {
	public:
		#ifdef __CUDA_ARCH__
			__align__(8) char data[16];
		#else
			char data[16];	//or else the copied argument hashToCrack
			//will not be aligned resulting in a memory error in operator==
		#endif
		__host__ __device__ HASH& operator= (HASH const &src);
		__host__ __device__ HASH();
		__host__ __device__ HASH( const char hex[32]);
		__host__ __device__ int operator!=(const HASH &b) const;
		__host__ __device__ int operator==(const HASH &b) const;
		__host__ __device__ void print(const char* msg) const;
};

enum MD5_STRING_TYPES {LOWERCASE=1, UPPERCASE=2, LOWUP=3, DIGITS=4};
//, 6:UPDIGITS, 5:LOWDIGITS, 7:LOWUPDIGITS};
class MD5_STRING {
	private:
		enum MD5_STRING_TYPES type;
	public:
		#ifdef __CUDA_ARCH__
			__align__(8) char data[64];
		#else
			char data[64];
		#endif
		__host__ __device__ void operator=(const MD5_STRING &src);
		__host__ __device__ MD5_STRING( );
		__host__ __device__ MD5_STRING( const char* src, MD5_STRING_TYPES type );
		__host__ __device__ MD5_STRING( const MD5_STRING &src );
		__host__ __device__ MD5_STRING( uint64_t num );
		__host__ __device__ MD5_STRING& operator++();
		__host__ __device__ MD5_STRING operator++(int);
		__host__ __device__ void print() const;
};
	
const uint32_t s[64] = 
	{ 7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
	  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
	  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
	  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21 };
const uint32_t K[64] =
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
	 
MD5_STRING unmd5( const HASH hash );
HASH md5(const char* original_message);

MD5_STRING cudaUnmd5( const HASH hash );
__global__ void cudaMd5Kernel( const HASH hash, char *original_message, uint32_t respawn_number);

#endif

//-compiles with gcc only if copy and assignment operator have const in their arguments!