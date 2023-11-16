#include <stdlib.h>
#include <stdio.h>
#include <ucontext.h>

void func(void *p_value)
{
    printf("%s\n", (char*)p_value);
}

int main(int argc, char *argv[])
{ 
    char stack[1000] = {0};

    ucontext_t context, rt;
    getcontext(&context);
    getcontext(&rt);

    context.uc_stack.ss_sp = stack;
    context.uc_stack.ss_size = 1000;
    context.uc_link = &rt;

    printf("start\n");
    char *p_value = "hello, world";
    makecontext(&context, (void (*)(void))func, 1, p_value);

    //swapcontext(&rt, &context);
    setcontext(&context);

    printf("finish\n");

    return 0;
}
