#ifndef __LOG_H__
#define __LOG_H__

#include <stdlib.h>
#include <stdio.h>

#ifndef dump
#define dump(a, ...)                printf("%s[%d] mark: " a, __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#define DEFAULT_DUMP_LEN   60
#define DEBUG 1

#if DEBUG
#undef  dump_debug 
#define dump_debug(a, ...)      dump(a, ##__VA_ARGS__)
#else
#define dump_debug(a, ...) 
#endif

#define output_data             printf
#define dump_data(p_data, len)                                             \
({                                                                         \
    s32 i =0;                                                              \
    output_data("len is %d", len);                                         \
    len = len > cfg_dump_len ? cfg_dump_len : len;                         \
    for(i = 0; i < len; i++) {                                             \
        if (0 == i % 16) {                                                 \
            output_data("\n");                                             \
	    output_data("0x%04x: ", i);                                    \
        }                                                                  \
                                                                           \
        if (i + 1 < len) {                                                 \
            output_data("%02x%02x ", p_data[i]&0xFF, p_data[i+1]&0xFF);    \
            i++;                                                           \
        }                                                                  \
        else {                                                             \
            output_data("%02x", p_data[i]&0xFF);                           \
        }                                                                  \
    }                                                                      \
    output_data("\n\n");                                                   \
})


#endif
