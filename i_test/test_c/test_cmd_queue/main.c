#include <stdlib.h>
#include <stdio.h>
#include "cmd_queue.h"
#include <unistd.h>

int main() 
{
    init();
    printf("finish init\n");
    int ret = 0; 
	int i = 0;
	cmd_info cmd_info = {.cmd_type=1};
	for (i = 0; i < 60; i++) {
	    cmd_info.id = i;
	    cmd_info.cmd_type = (cmd_info.cmd_type == 0) ? 1: 0;
	    cmd_info.cmd_info = (cmd_info.cmd_type == 0) ? "cmd is start, 123" : "cmd is stop, 321";
	    ret = do_start_comm(&cmd_info);
		int num = 3;
		while ((num--) && (0 != ret)) {
		    ret = do_start_comm(&cmd_info);
			sleep(1);
		}
    }

    printf("finish start_comm\n");
	stop();
}
