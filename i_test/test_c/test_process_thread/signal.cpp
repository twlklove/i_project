#include <signal.h>

signal(SIGINT, my_exit); 
signal(SIGKILL, my_exit); 
signal(SIGTERM, my_exit);

void my_exit(int signal)
{
    xxx;
}
