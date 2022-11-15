#ifndef __I_TYPES_H
#define __I_TYPES_H

#include <stdio.h>
#include <stdlib.h>

#ifndef __initconst
#define __initconst
#endif

#ifndef __init
#define __init
#endif

#ifndef __pure
#define __pure            __attribute__((pure))
#endif

#ifndef __aligned
#define __aligned(x)     __attribute__((aligned(x)))
#endif

#ifndef __alias 
#define __alias(symbol)                 __attribute__((__alias__(#symbol)))
#endif

#ifndef ____cacheline_aligned
#define ____cacheline_aligned
#endif

#ifndef USE_KERNEL
#define swab16(x) ((__u16)(				\
	(((__u16)(x) & (__u16)0x00ffU) << 8) |			\
	(((__u16)(x) & (__u16)0xff00U) >> 8)))

#define __swab32(x) ((__u32)(				\
	(((__u32)(x) & (__u32)0x000000ffUL) << 24) |		\
	(((__u32)(x) & (__u32)0x0000ff00UL) <<  8) |		\
	(((__u32)(x) & (__u32)0x00ff0000UL) >>  8) |		\
	(((__u32)(x) & (__u32)0xff000000UL) >> 24)))

#define __swab64(x) ((__u64)(				\
	(((__u64)(x) & (__u64)0x00000000000000ffULL) << 56) |	\
	(((__u64)(x) & (__u64)0x000000000000ff00ULL) << 40) |	\
	(((__u64)(x) & (__u64)0x0000000000ff0000ULL) << 24) |	\
	(((__u64)(x) & (__u64)0x00000000ff000000ULL) <<  8) |	\
	(((__u64)(x) & (__u64)0x000000ff00000000ULL) >>  8) |	\
	(((__u64)(x) & (__u64)0x0000ff0000000000ULL) >> 24) |	\
	(((__u64)(x) & (__u64)0x00ff000000000000ULL) >> 40) |	\
	(((__u64)(x) & (__u64)0xff00000000000000ULL) >> 56)))

#define __swahw32(x) ((__u32)(			\
	(((__u32)(x) & (__u32)0x0000ffffUL) << 16) |		\
	(((__u32)(x) & (__u32)0xffff0000UL) >> 16)))

#define __swahb32(x) ((__u32)(			\
	(((__u32)(x) & (__u32)0x00ff00ffUL) << 8) |		\
	(((__u32)(x) & (__u32)0xff00ff00UL) >> 8)))

#define local_irq_save(flags)
#define ktime_get_ns()     get_ns()
#define local_irq_restore(flags)
#define pr_info(a, ...)  printf("%s[%04d]" a, __FILE__, __LINE__, ##__VA_ARGS__) 
#define pr_warn(a, ...)  printf("%s[%04d]" a, __FILE__, __LINE__, ##__VA_ARGS__)
#define cond_resched()
#define unlikely
#define __weak

#ifndef __force
#define __force
#endif

#ifndef cpu_to_le32
#define cpu_to_le32(x)    x
#endif

#ifndef __cpu_to_le32
#define __cpu_to_le32     cpu_to_le32
#endif

#ifndef cpu_to_be32
#define cpu_to_be32(x)    __swab32(x) 
#endif

#ifndef __cpu_to_be32
#define __cpu_to_be32     cpu_to_be32 
#endif

#ifndef __be32_to_cpu
#define __be32_to_cpu(x)  __swab32(x) 
#endif

#ifndef __le32_to_cpu
#define __le32_to_cpu(x)  x
#endif

#include <sys/time.h>
#include <stdlib.h>
#include <time.h>


typedef unsigned char         u8; 
typedef signed char           s8;
typedef unsigned short       u16;
typedef signed short         s16;
typedef unsigned int         u32;
typedef signed int           s32;
typedef unsigned long long   u64;
typedef signed long long     s64;

#define get_us()                                     \
({                                                   \
    struct timeval tv;                               \
    gettimeofday(&tv, NULL);                         \
    tv.tv_sec*1000000 + tv.tv_usec;                  \
})

#define get_ns() \
({                                                    \
    struct timespec tv = {0, 0};                      \
    clock_gettime(CLOCK_REALTIME, &tv);               \
    tv.tv_sec*1000000000 + tv.tv_nsec;                \
})

#endif

#endif
