#include <stdio.h>	//printf, fopen, fprintf
#include <string.h>	//strlen
#include <stdlib.h>	//malloc,free,calloc
#include <time.h>
#include "md5main.hpp"
#include "md5kernel.hpp"
#include "statistics.hpp"
#include <assert.h>
#include <math.h>
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

uint32_t rolhost(uint32_t num, int bits) {
	return (num << bits) | (num >> (32-bits));
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
			unsigned int len = strlen(this->data);
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
			int len = strlen(src);
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
					this->data[56] = 8*(strlen(this->data)-1);
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
		B += rolhost( A+F+K[i]+((uint32_t*)msg)[g], s[i] );
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

/******************************************************************************
 ************************************* MAIN ***********************************
 ******************************************************************************/
int main(int argc, char** args) {
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
	assert( floor(Mean(stddevdbg,5)*1000)   == floor(3.6*1000) );
	assert( floor(StdDev(stddevdbg,5)*1000) == floor(sqrt(145)/5*1000) );
	
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
	hashstruct hash2; memcpy( hash2.data, HashToCrack.data, 64 );
	md5struct result = cudaUnmd5( hash2 );
	printf("Cracked String \"%s\" found!\n",result.data);
	return 0;
}
