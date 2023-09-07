#include <stdlib.h>
#include <stdio.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

int main()
{
    pid_t pid = 0;
    int status = 0;
    long data = 0;
    long *addr = (void*)0x55e816f4c048;
    long condition = 0x102;

    pid = fork();
    if (pid == 0) {
        ptrace(PTRACE_TRACEME, pid, NULL, NULL);
        execl("./test/run_test", "run_test", NULL);
    }
    else if (pid > 0) {
        printf("pid is %d\n", pid);
        errno = 0;
        ptrace(PTRACE_ATTACH, pid, NULL, NULL);
         perror("error");
        waitpid(pid, &status, 0);
        ptrace(PTRACE_CONT, pid, NULL, NULL);

        perror("error");
        while(WIFSTOPPED(status)) {
#if 0
                waitpid(pid, &status, 0);
                if (WIFEXITED(status)){
                    ptrace(PTRACE_DEATTACH, pid, NULL, NULL);
                    break;
                }
#endif
            data = ptrace(PTRACE_PEEKDATA, pid, addr, NULL) & 0xFFFFFFFF;
            printf("pid is %d, breakpoint hit at  address %p, data is %lx\n", pid, addr, data);
            sleep(1);
            if (data == condition) {
                printf("pid is %d, breakpoint hit at  address %p, data is %lx\n", pid, addr, data);
                ptrace(PTRACE_POKEDATA, pid, addr, data+1);
                break;
            }

            ptrace(PTRACE_CONT, pid, NULL, NULL);
            waitpid(pid, &status, 0);
        }
    }
    else {
        perror("fork");
        exit(1);
    }

    return 0;
}
