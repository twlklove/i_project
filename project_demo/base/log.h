#ifndef __LOG_H__
#define __LOG_H__

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

extern const u8 *levels[log_level_num];
extern u32 cur_log_level;
extern u32 dump_to_file; // 1:stdout, 2: log_file, 3 : stdout & log_file
extern FILE *p_log_file;

void log_init();
void log_uninit();
void set_dump_level(const u8 *level_name);  
void set_dump_to_file(const u32 value);

#define DUMP(level_num, ...) (                                                                                            \
{                                                                                                                         \
    do {                                                                                                                  \
        if (!((level_num <= min_log_level) && (level_num >= max_log_level) && (level_num <= cur_log_level))) {            \
            break;                                                                                                        \
        }                                                                                                                 \
        _DUMP_TO_STDOUT(levels[level_num], __VA_ARGS__);                                                                  \
        _DUMP_TO_FILE(levels[level_num], __VA_ARGS__);                                                                    \
    }while(0);                                                                                                            \
})

// ## :concat two strings
// #  :convert parameter to string
// #@ :convert paremeter to char
// __VA_ARGS__
#define _DO_DUMP(p_file, level, ...) (                                                                                    \
{                                                                                                                         \
    do {                                                                                                                  \
        if (NULL == p_file) {                                                                                             \
            break;                                                                                                        \
        }                                                                                                                 \
        fprintf(p_file, "[%s,%s][%s][%s:%d:%s] ==> ", __DATE__, __TIME__,level, __FILE__, __LINE__, __FUNCTION__);        \
        fprintf(p_file, __VA_ARGS__);                                                                                     \
    }while(0);                                                                                                            \
})

#define _DUMP_TO_STDOUT(level, ...) (                                                                                     \
{                                                                                                                         \
    do {                                                                                                                  \
        if (0 == (dump_to_file & 0x01)) {                                                                                 \
            break;                                                                                                        \
        }                                                                                                                 \
        _DO_DUMP(stdout, level, __VA_ARGS__);                                                                             \
    }while(0);                                                                                                            \
})


#define _DUMP_TO_FILE(level, ...) (                                                                                       \
{                                                                                                                         \
    do {                                                                                                                  \
        if (0 == (dump_to_file & 0x02)) {                                                                                 \
            break;                                                                                                        \
        }                                                                                                                 \
        _DO_DUMP(p_log_file, level, __VA_ARGS__);                                                                         \
    }while(0);                                                                                                            \
})

#endif

