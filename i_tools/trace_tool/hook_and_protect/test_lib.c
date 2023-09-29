#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/mman.h>

void test(int *p_data)
{
    int test = 0; 
    int *p_test = (&test + 13);
    printf("%s %p: addr of test is  %p\n", __func__, __func__, &test);
    printf("%s: Hi, I will set 10 on addr %p\n", __func__, p_test);
    *p_test = 10;
}

