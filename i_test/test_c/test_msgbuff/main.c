#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>

#include "dev_core.h"
#include "msg_buff.h"
#include "sys/sysinfo.h"

void signal_handler(int signo, siginfo_t *info, void *context)
{
    printf("Interrupt signal %d received\n", signo);
    printf("signo is %d, errno is %d, si_code is %d, sendpid is %d, senduid is %d\n",
            info->si_signo, info->si_errno, info->si_code, info->si_pid, info->si_uid);


    exit(EXIT_SUCCESS);
}

typedef void (*sa_handler_func)(int signo, siginfo_t *info, void *context);

//getcontext()
void register_signal(int signo, sa_handler_func handler) 
{
    struct sigaction act = { 0 };
 
    act.sa_flags = SA_SIGINFO;
    act.sa_sigaction = handler;
    if (sigaction(signo, &act, NULL) == -1) {
        perror("sigaction");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    int ret = 0;
    int count = 10;
    struct comm_dev *p_dev = NULL;

    register_signal(SIGINT, signal_handler);  //SIGSEGV

    printf("system conf cpu num is %d\n", get_nprocs_conf()); 
    printf("system enable num is %d\n", get_nprocs());  //real cpu num

    workqueue_init();

    do { 
        p_dev = comm_alloc_dev();
        if (!p_dev) {
            ret = -1;
            break;
        }

        ret = comm_register_dev(p_dev);
        if (0 != ret) {
            ret = -1;
            break;
        }
       
        struct msg_buff *msgb;
        do {
            msgb = malloc(sizeof(struct msg_buff));
            if (NULL == msgb) {
                break;
            }

            msgb->len = count;
            comm_recv_data(p_dev, msgb);
            if (0 == count % 2) {
                usleep(5000);
            }
        }while (count--);

    } while (0); 

    sleep(2);

    if (!p_dev) {
        comm_unregister_dev(p_dev);
        comm_free_dev(p_dev);
    }
    
    printf("end\n");
    workqueue_uninit();
 
    return ret;
}
