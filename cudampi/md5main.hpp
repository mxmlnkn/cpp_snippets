#ifndef MD5MAIN_H
#define MD5MAIN_H

#include <stdint.h>	//uint32_t

class HASH {
	public:
		char data[16];	//or else the copied argument hashToCrack
		//will not be aligned resulting in a memory error in operator==
		HASH& operator= (HASH const &src);
		HASH();
		HASH( const char hex[32]);
		int operator!=(const HASH &b) const;
		int operator==(const HASH &b) const;
		void print(const char* msg) const;
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
		void operator=(const MD5_STRING &src);
		MD5_STRING( );
		MD5_STRING( const char* src, MD5_STRING_TYPES type );
		MD5_STRING( const MD5_STRING &src );
		MD5_STRING( uint64_t num );
		MD5_STRING& operator++();
		MD5_STRING operator++(int);
		void print() const;
};
 
MD5_STRING unmd5( const uint32_t *hash );
HASH md5(const char* original_message);

#endif

//-compiles with gcc only if copy and assignment operator have const in their arguments!