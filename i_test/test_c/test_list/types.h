#ifndef __I_TYPES_H__
#define __I_TYPES_H__

typedef unsigned char  u8;
typedef unsigned short u16;
typedef unsigned int   u32;

typedef char  s8;
typedef short s16;
typedef int   s32;


struct list_head {
	struct list_head *next, *prev;
};

#endif 
