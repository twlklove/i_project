#ifndef __CMD_QUEUE_H__
#define __CMD_QUEUE_H__

typedef struct {
    int id;
    int cmd_type; // 0:start, 1: stop
	char *cmd_info;
}cmd_info;

int init(void);
int do_start_comm(cmd_info *p_info);
void stop();

#endif
