#ifndef __INIT_H__
#define __INIT_H__
#include "platform.h"
#include "config.h"
#include "log.h"
#include "cJSON.h"

inline static void init()
{
    parse_log_cfg();
    log_init();
    dump_platform_info();
}

inline static void uninit()
{
    log_uninit();
}

#endif
