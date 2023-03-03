#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <pthread.h>
#include <unistd.h>

/*
function 1 :
typedef void (*sighandler_t)(int);
sighandler_t signal(int signum, sighandler_t handler);  //handler也可以是：SIG_IGN屏蔽该信号, SIG_DFL恢复默认行为 
                                                        //signal(SIGINT, handler); // SIGTERM/SIGUSR1/SIGUSR2
int kill(pid_t pid, int sig); // send signal
int sigprocmask(int how, const sigset_t *set, sigset_t *oldset); // how:SIG_BLOCK, SIG_UNBLOCK, SIG_SETMASK

function 2 :
int sigaction(int signum, const struct sigaction *act, struct sigaction *oldact);
struct sigaction {
               void     (*sa_handler)(int); //can be SIG_DFL和SIG_IGN, or signal(...)
               void     (*sa_sigaction)(int, siginfo_t *, void *); //for real time(also be used for none real time), siginfo_t includes info 
               sigset_t   sa_mask; 
               int        sa_flags;  //0: use default; SA_SIGINFO, now use sa_sigaction, not sa_handler
               void     (*sa_restorer)(void);
           };
siginfo_t {
    int      si_signo;
    sigval_t si_value; // data which is included by signal
    ...
};
typedef union sigval
{ 
	int sival_int; 
	void *sival_ptr; 
}sigval_t;

int sigqueue(pid_t pid, int sig, const union sigval value); // send signal
*/


int run_mark = 1;
void handler(int sig_no)
{
	printf("received a signal: %d\n", sig_no);
    run_mark = 0;
}

typedef void (*test_func)(void *);
void test_0_sub(void *p_data)
{
    sigset_t set, oset;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1); // add signal
                              //
#if 1
    int i = 0;
    for(i = 1 ; i <= 64 ; i++)	//look at signal set
    {
    	if(sigismember(&set,i) == 1)
    	{
    		printf("sig in set is %d", i);
    	}
    }
    printf("\n");
#endif
 
    sigprocmask(SIG_SETMASK,&set, &oset);	 //set block set 
    signal(SIGUSR1, handler);  //set signal handler
    sigprocmask(SIG_SETMASK, &oset, NULL);

    while(run_mark) {
        sleep(1);
    }
}

void test_0_parent(void *p_data)
{
    sleep(2);
    pid_t pid = *((int*)p_data);
	kill(pid, SIGUSR1);
    wait(NULL);
}

void handler_info(int signo, siginfo_t *p_data, void *p_unknow)
{
    int sig_value = p_data->si_value.sival_int;
    printf("signo=%d, sig data :%d\n",signo, sig_value);
    run_mark = 0;
}

void test_1_sub(void *p_data)
{
    struct sigaction act;
	sigemptyset(&act.sa_mask);
	act.sa_sigaction=handler_info;
    act.sa_flags=SA_SIGINFO; // for use sa_sigaction
	sigaction(SIGUSR1, &act, NULL);

    while (run_mark) {
        sleep(1);
    }
}

void test_1_parent(void *p_data)
{  
    sleep(2);
    pid_t pid = *((int*)p_data);
    union sigval sigvalue;
    sigvalue.sival_int = 1;
    sigqueue(pid, SIGUSR1, sigvalue);
    printf("send signal: SIGUSR1 success!\n");
    wait(NULL);
}

void test(int *p_ppid, const char *p_test_name, test_func p_parent_func, test_func p_sub_func)
{
    if (0 == *p_ppid) {
        return;
    }

    printf("%s\n", p_test_name);

    do {
	    pid_t pid = fork();

	    if(pid == -1) {
	    	perror("create fork");
	    }
	    else if(pid == 0) {
            p_sub_func(NULL);
            int i = 0;
            while(i++ < 10) {
                sleep(1);
            }
            *p_ppid = pid;
	    }
	    else {
            p_parent_func(&pid);
            printf("\n");
	    }
    }while(0); 
}

int main(int argc, char *argv[])
{
    pid_t pid = getpid();
    test(&pid, "test_0", test_0_parent, test_0_sub);
    test(&pid, "test_1", test_1_parent, test_1_sub); 
    return 0;
}
