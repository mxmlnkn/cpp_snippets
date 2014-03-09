//compile with: nvcc matrixmult.c -o matrixmult --compiler-options "-std=c99"
//or under windows: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v5.5\bin\nvcc.exe matrixmult.cu -o matrixmult --compiler-bindir "C:\Program Files (x86)\Microsoft Visual Studio 11.0\VC\bin" --machine 32
//Notes: snprintf, %zu don't work with cl.exe >:O

#include <stdio.h>
#include <string.h>	//strlen
#include <stdlib.h>	//malloc,free,calloc
#include "cuda.h"
#include <time.h>
#include "md5.h"

inline void gpuAssert(cudaError_t code, int line) {
   if (code != cudaSuccess)
      printf("GPUassert at Line %d: %s\n", line, cudaGetErrorString(code));
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
		void HASH::print(const char* msg = "Hash: ") {
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
		void MD5_STRING::operator=(MD5_STRING &src) {
			memcpy(this->data, src.data, 64);
		}
		/**************************** Constructors ****************************/
		MD5_STRING::MD5_STRING( ) {
			memset(this->data, 0, 64);
		}
		MD5_STRING::MD5_STRING(const char *src,MD5_STRING_TYPES type=LOWERCASE){
			#ifdef __CUDA_ARCH__
				int len = cudaStrlen(src);
			#else
				int len = strlen(src);
			#endif
			memset( this->data,0,64 );
			memcpy( this->data, src, len%64 );
			this->data[56] = 8*len;
			this->type = type;
		}
		MD5_STRING::MD5_STRING( MD5_STRING &src ) {
			(*this) = src;
		}
		MD5_STRING::MD5_STRING( unsigned long int num ) {
			memset(this->data,0,64);
			//input num is 32 bit => max reachable string: ??? 
			//like operator++ the string is treated like little endian, meaning
			//the overflow bits of data[i]+num are summed up in data[i+1]
			for(int i=0; i<54; i++) {
				//if type == LOWERCASE
				this->data[i] = 'a' + (num % 26);
				num /= 26;
				if (num == 0) {
					this->data[i+1] = (char)128;
					this->data[i+2] = 0;
					break;
				}
			}
			#ifdef __CUDA_ARCH__
				this->data[56] = 8*cudaStrlen(this->data);
			#else
				this->data[56] = 8*strlen(this->data);
			#endif
		}
		/****************************** Operators *****************************/
		MD5_STRING& MD5_STRING::operator++() {
			for (int i=0; i<54; i++) {
				if (this->data[i] == 0) {
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


MD5_STRING unmd5( const HASH hash ) {
	MD5_STRING original("", LOWERCASE);
	//for (int i=0; i<256; i++)
	//	printf("%s ", (++original).data );
	while( md5( (++original).data ) != hash ) {
		//printf("%s ", original.data );
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
	
	char msg[64];
	strcpy(msg, original_message);
	//set last bit of 512bit message to 1 and store at bit 480 the length of msg
	//actually 64bit length, but this programm will only work with max 55 byte strings!
	//length is in little endian: 0x0123 4567 89AB CDEF -> Memory: 23 01 57 45 ...
	//therefore the smaller 32bit part of the 64 bit integer is in [56] not [57]
	const DWORD len = strlen(msg);
	msg[len] = (char)128;
	memset( msg+len+1, 0, 64-len-1 );
	msg[56] = 8*len;	//Message Length in bits not bytes!
	/*printf("\nMessage Length: %lu\nSize of unsigned long int: %lu Bytes\n", len, sizeof(DWORD) );
	printf("In Memory:");
	for (int i=0; i<64; i++) {
		if (i%16 == 0)
			printf("\n");
		else if (i%4 == 0)
			printf("\t");
		printf("%x\t", (unsigned char) msg[i]);
	}
	printf("\n");*/
	
	DWORD A = a0;
	DWORD B = b0;
	DWORD C = c0;
	DWORD D = d0;
	DWORD F,g;

	#pragma unroll	//optimize by laying out all operations and memory accesses
	//in program code like done manually in md5.asm
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
	//"test" should convert to: 098f6bcd 4621d373 cade4e83 2627b4f6
	HASH hash;
	((DWORD*)(hash.data))[0] = a0+A;
	((DWORD*)(hash.data))[1] = b0+B;
	((DWORD*)(hash.data))[2] = c0+C;
	((DWORD*)(hash.data))[3] = d0+D;
	return hash;
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
	//char* sr = "test\0";
	HASH hash = md5("test");
	hash.print();
	HASH HashToCrack = "098f6bcd4621d373cade4e832627b4f6";
	//{ 0x09, 0x8f, 0x6b, 0xcd, 0x46, 0x21, 0xd3, 0x73,
	                      // 0xca, 0xde, 0x4e, 0x83, 0x26, 0x27, 0xb4, 0xf6 };
	//HashToCrack.print();
	//printf("test comparison: %i", md5( "test" ) != hash );
	
	//MD5_STRING(num) Debug:
	MD5_STRING testsr(26);
	printf("Number->MD5-String: 26->%s", testsr.data);
	MD5_STRING testsr2(~0);
	printf(", ~0->%s\n", testsr2.data);
	
	/*for(int i=0; i<5; i++) {
		clock_t start_clocks = clock();
		MD5_STRING cracked = unmd5(HashToCrack);
		printf( "Cracked String: \"%s\" found in %f seconds (%i ticks per second)\n",
				cracked.data, (float)(clock()-start_clocks)/CLOCKS_PER_SEC, CLOCKS_PER_SEC );
	}*/

/*	cudaEvent_t start,end;
	cudaEventCreate (&start); cudaEventCreate (&end);
	float elapsedTime;
	cudaEventRecord(start,0);
	//	cudaUnmd5(HashToCrack);
	cudaEventRecord(end,0); cudaEventSynchronize(end);
	gpuErrchk( cudaEventElapsedTime(&elapsedTime,start,end) );
	printf("Elapsed Time: %f ms\n",elapsedTime);*/
	
	return 0;
}

/************************************ TODO *************************************
-for short messages the number of memory accesses could be reduced by adding a
 conditional check, because all the values after the short string are 0 or the
 length of the string!
 */
