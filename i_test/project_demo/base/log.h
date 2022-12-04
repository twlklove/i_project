#ifndef __LOG_H__
#define __LOG_H__

#include <sys/time.h>
#include "types.h"
#define LOG_FILE "i_info.log"

enum DUMP_LEVEL{
    max_log_level = 0,
    fatal = max_log_level,
    err,
    warn,
    info,
    debug,
    min_log_level = debug,
    log_level_num = min_log_level+1,
};

extern const u8 *levels[log_level_num][2];
extern u32 cur_log_level;
extern u32 dump_to_file; // 1:stdout, 2: log_file, 3 : stdout & log_file
extern FILE *p_log_file;

void log_init();
void log_uninit();
void set_dump_level(const u8 *level_name);  
void set_dump_to_file(const u32 value);

#define DUMP(level_num, ...)                                                                                      \
({                                                                                                                 \
    do {                                                                                                          \
        if (!((level_num <= min_log_level) && (level_num >= max_log_level) && (level_num <= cur_log_level))) {    \
            break;                                                                                                \
        }                                                                                                         \
        _DUMP_TO_STDOUT(level_num, __VA_ARGS__);                                                                  \
        _DUMP_TO_FILE(level_num, __VA_ARGS__);                                                                    \
    }while(0);                                                                                                    \
})

// ## :concat two strings
// #  :convert parameter to string
// #@ :convert paremeter to char
// __VA_ARGS__
#define _DO_DUMP(p_file, level_num, ...)                                                         \
({                                                                                                 \
    do {                                                                                          \
        if (NULL == p_file) {                                                                     \
            break;                                                                                \
        }                                                                                         \
	    struct timeval cur_time = {.tv_sec=0, .tv_usec=0};                                        \
	    u32 ret = gettimeofday(&cur_time, NULL);                                                  \
        if(0 != ret) {                                                                            \
	         cur_time.tv_usec = 0;                                                                \
	    }                                                                                         \
        u32 cur_ms = cur_time.tv_usec / 1000;                                                     \
	    if (stdout == p_file){                                                                    \
	        fprintf(p_file, levels[level_num][1]);                                                \
	    }                                                                                         \
        fprintf(p_file, "[%lu.%u][%s][%s:%d:%s] ==> ",                                             \
                cur_time.tv_sec, cur_ms, levels[level_num][0], __FILE__, __LINE__, __FUNCTION__); \
	    fprintf(p_file, __VA_ARGS__);                                                             \
	    if (stdout == p_file){                                                                    \
	        fprintf(p_file, "\033[0m");                                                           \
	    }                                                                                         \
    }while(0);                                                                                    \
})

#define _DUMP_TO_STDOUT(level_num, ...)                                                          \
({                                                                                                \
    do {                                                                                         \
        if (0 == (dump_to_file & 0x01)) {                                                        \
            break;                                                                               \
        }                                                                                        \
        _DO_DUMP(stdout, level_num, __VA_ARGS__);                                                \
    }while(0);                                                                                   \
})


#define _DUMP_TO_FILE(level_num, ...)                                                            \
({                                                                                                \
    do {                                                                                         \
        if (0 == (dump_to_file & 0x02)) {                                                        \
            break;                                                                               \
        }                                                                                        \
        _DO_DUMP(p_log_file, level_num, __VA_ARGS__);                                            \
    }while(0);                                                                                   \
})

#endif

