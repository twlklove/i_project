#ifndef __PLATFORM__
#define __PLATFORM__
#include "log.h"

//look up using gcc -dM -E - < /dev/null 
#if defined(__linux__)
#define PLATFORM_ID  "linux"
#elif defined (__WIN32__)
#define PLATFORM_ID  "windows32"
#elif defined (__CYGWIN__)
#define PLATFROM_ID  "cygwin"
#endif

#if defined(__ARM64__)                  // || defined(...) 
#define ARCH_ID "aarch64"
#elif defined(__ARM__)
#define ARCH_ID "arm"
#elif defined(__i386__)
#define ARCH_ID "i386"
#elif defined(__x86_64__)
#define ARCH_ID "x86_64"
#endif

#ifndef PLATFORM_ID
#define PLATFORM_ID "not supported platform"
#endif

#ifndef ARCH_ID
#define ARCH_ID "not supported arch"
#endif

static inline void dump_platform_info()
{     
    const char *p_platform_info = "platform["PLATFORM_ID"]";
    const char *p_arch_info = "arch["ARCH_ID"]";
    DUMP(info, "%s\n", p_arch_info); 
    DUMP(info, "%s\n", p_platform_info);
}

#endif
