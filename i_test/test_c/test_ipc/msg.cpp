#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/msg.h>
 
struct msg_st
{
    long int msg_type;
    char text[BUFSIZ];
};
 
int msgid = -1;
msgid = msgget((key_t)1234, 0666 | IPC_CREAT); //get or create
if(msgid == -1)

if(msgrcv(msgid, (void*)&data, BUFSIZ, msgtype, 0) == -1) // recv

if(msgctl(msgid, IPC_RMID, 0) == -1) // del

msg_st data;
data.msg_type = 1;  
strcpy(data.text, buffer);
if(msgsnd(msgid, (void*)&data, MAX_TEXT, 0) == -1) // send
