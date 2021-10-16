#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include "cmd_queue.h"
 
#define NUM_THREADS 2
#define MAX_CMD_NUM  20

int exit_marker = 0;

typedef struct {
    pthread_mutex_t lock;
    int index_in;
	int index_out;
	int valid_num;
    cmd_info cmd_info[MAX_CMD_NUM]; 
}cmd_queue_type;

cmd_queue_type cmd_queue = {.lock=PTHREAD_MUTEX_INITIALIZER, .index_in = 0, .index_out=0, .valid_num=0};

void start_func(void *args)
{
    cmd_info *p_info = (cmd_info*)args;
    printf("%s, cmd id is : %d, type is %d, info is : %s\n", __FUNCTION__, p_info->id, p_info->cmd_type, p_info->cmd_info);
}

void stop_func(void *args)
{
    cmd_info *p_info = (cmd_info*)args;
    printf("%s, cmd id is : %d, type is %d, info is : %s\n", __FUNCTION__, p_info->id, p_info->cmd_type, p_info->cmd_info);
}

typedef void (*cmd_func)(void *args);
cmd_func cmd_funcs[] = {start_func, stop_func};


void *queue_process(void *args)
{
    int *thread_arg; 
    thread_arg = args;
    printf("Hello from thread %d\n", *thread_arg);
   
    while(1) {
	    if (1 == exit_marker) {
		    return NULL;
		}
        	
		if (0 != cmd_queue.valid_num) { 
		    if (cmd_queue.index_out == MAX_CMD_NUM) {
			   cmd_queue.index_out = 0; 
			}

		    cmd_info *p_info = &(cmd_queue.cmd_info[cmd_queue.index_out]);
		    int cmd_type = p_info->cmd_type;
			cmd_funcs[cmd_type](p_info);

			pthread_mutex_lock(&cmd_queue.lock);
			cmd_queue.index_out += 1;
			cmd_queue.valid_num -= 1;
            pthread_mutex_unlock(&cmd_queue.lock);
		}
		else {
		    sleep(1);
		}	
	}

    return NULL;
}

pthread_t pt;

int init(void)
{
    int rc, t = 0;
     
    rc = pthread_create(&pt, NULL, queue_process, &t); 
    if (rc) 
    { 
        printf("ERROR; return code is %d\n", rc); 
        return EXIT_FAILURE; 
    } 
        
    return EXIT_SUCCESS;
}

int do_start_comm(cmd_info *p_info)
{
    pthread_mutex_lock(&cmd_queue.lock);
    if (cmd_queue.valid_num >= MAX_CMD_NUM) {
	    printf("full\n");
		pthread_mutex_unlock(&cmd_queue.lock);
	    return -1;
	}

	if (cmd_queue.index_in == MAX_CMD_NUM) {
	    cmd_queue.index_in = 0;
	}
    
	cmd_queue.cmd_info[cmd_queue.index_in] = *p_info;
	cmd_queue.index_in += 1;
	cmd_queue.valid_num += 1;
	pthread_mutex_unlock(&cmd_queue.lock);

    return 0;
}

void stop() 
{
    while (cmd_queue.valid_num) {
	    sleep(1);
	}

    exit_marker = 1;
    pthread_join(pt, NULL);
}

