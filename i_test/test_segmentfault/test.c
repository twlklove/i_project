#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <time.h>
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <malloc.h>
#include <sys/mman.h>
#include "dump_stack.h"

#ifndef __USE_GNU
#define __USE_GNU
#endif
#include <ucontext.h>

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

static void dump_registers(ucontext_t *context)
{
    printf("R8: %1llx\n", context->uc_mcontext.gregs[0]);
}

static void handler(int sig, siginfo_t *si, void *unused)
{
    printf("Got SIGSEGV at address : %p\n", si->si_addr); 
    ucontext_t context;
    getcontext(&context);
    dump_registers(&context);

    dump_stack();

    exit(EXIT_SUCCESS);
}

void test_0()
{
    char *p_name = NULL;
    *p_name = 1;
}

void test_1()
{
    const int pagesize = sysconf(_SC_PAGE_SIZE);
    if (pagesize == -1)
        handle_error("sysconf");

    /* Allocate a buffer aligned on a page boundary;
       initial protection is PROT_READ | PROT_WRITE. */

    char *buffer = memalign(pagesize, pagesize);
    if (buffer == NULL)
        handle_error("memalign");

    printf("Start of region:        %p, pagesize is %d\n", buffer, pagesize);

    if (mprotect(buffer, pagesize, PROT_READ) == -1)
        handle_error("mprotect");

    for (char *p = buffer ; ; )
        *(p++) = 'a';

    printf("Loop completed\n");     /* Should never happen */
    exit(EXIT_SUCCESS);

}

int main(int argc, char *argv[])
{
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = handler;  
    if (sigaction(SIGSEGV, &sa, NULL) == -1) 
        handle_error("sigaction");

//    test_0();
    test_1();

    return 0;
}
