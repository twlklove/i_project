#include <sys/ipc.h>

pid_t pid = fork();
if(pid < 0)
{
    perror("fork error!");
    exit(1);
}

if(pid == 0)
{
    //child 
}
else if(pid > 0)
{
    // parent 
}

