#include <stdio.h>
#include <string.h>
#include "types.h"
#include "log.h"

const u8 *levels[log_level_num][2] = {
	                                {"fatal", "\033[0;31m"},
			                {"err",   "\033[1;31m"},
				        {"warn",  "\033[1;33m"},
					{"info",  ""},
					{"debug"  "\033[1;34m"},
				    };

u32 cur_log_level = debug;
u32 dump_to_file = 1; // 1:stdout, 2: log_file, 3 : stdout & log_file
FILE *p_log_file = NULL;

void log_init()
{
    p_log_file = fopen(LOG_FILE, "a+");
    if (NULL == p_log_file) {
        _DO_DUMP(stdout, err, "fail to init log file");
    }
}

void log_uninit() 
{
    if(NULL != p_log_file) {
        fclose(p_log_file);
    }
    p_log_file = NULL;
}

void set_dump_level(const u8 *level_name)                                                                                      
{                                                                                                                         
    int i = 0;                                                                                                            
    for (i = 0; i < log_level_num; i++) {                                                                                 
        if (0 == strncmp(levels[i][0], level_name, strlen(level_name))) {
           break;                                                                                                         
        }                                                                                                                 
    }                                                                                                                     
    if (log_level_num > i) {                                                                                              
        cur_log_level = i;                                                                                            
    }                                                                                                                     
}

void set_dump_to_file(const u32 value)                                                                                      
{                                                                                                                         
     dump_to_file = value;                                                                                            
}
