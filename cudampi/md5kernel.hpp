#ifndef MD5KERNEL_H
#define MD5KERNEL_H

#include <stdint.h>

typedef struct {
	char data[64];
} md5struct;

typedef struct {
	uint32_t data[4];
} hashstruct;

md5struct cudaUnmd5( const hashstruct hash );

#endif