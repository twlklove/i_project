#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>

#define MAX_STACK_DEPTH 20

void i_dump_stack() 
{
    int max_stack_depth = MAX_STACK_DEPTH;
    void *stack_trace_func_addrs[max_stack_depth];
    char **stack_trace_func_names = NULL;

#ifdef CONFIG_STACK_DEPTH
    if (CONFIG_STACK_DEPTH < MAX_STACK_DEPTH) {
        max_stack_depth = CONFIG_STACK_DEPTH;
    }
#endif

    do { 
        int stack_depth = backtrace(stack_trace_func_addrs, max_stack_depth);      //backtrace

        stack_trace_func_names = (char **)backtrace_symbols(stack_trace_func_addrs, stack_depth);      //backtrace_symbols
        if (NULL == stack_trace_func_names) {
            printf("fail to dump stack trace! \n");
            break;
        }

        printf("stack trace is : \n");
        int i = 0;
        for (i = 0; i < stack_depth; ++i) {
            printf(" [%d] %s \r\n", i, stack_trace_func_names[i]);
        }
    } while (0);

    if (NULL != stack_trace_func_names) {
        free(stack_trace_func_names);
        stack_trace_func_names = NULL;
    }

    return;
}

#ifdef DEBUG
void my_func2(void)
{
    i_dump_stack();
}

static void my_func(void)
{
    my_func2();
}

int main(int argc, char *argv[])
{
    my_func();
    return 0;
}
#endif
