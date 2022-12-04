#ifndef __TYPES_H__ 
#define __TYPES_H__

typedef unsigned char         u8;
typedef unsigned short        u16;
typedef unsigned int          u32; 
typedef unsigned long long    u64; 
#ifdef __i386__
typedef unsigned long         ul32; 
#elif __x86_64__
typedef unsigned long         ul64; 
#endif

typedef char                  s8;
typedef short                 s16;
typedef int                   s32; 
typedef long long             s64; 
#ifdef __i386__
typedef long                  sl32; 
#elif __x86_64__
typedef long                  sl64; 
#endif

#endif
