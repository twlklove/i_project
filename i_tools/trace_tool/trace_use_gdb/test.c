#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern char __ehdr_start[];

int buf[100] = {0};

void test_1()
{
    int data=20;
    int *p_data = &data+9;
    printf("write addr is %p\n", p_data);
    *(p_data) = 10;
    printf("exit from %s\n", __func__);
}

void test()
{  
    int data = 0;
    static int s_data = 0;
    printf("data addr is %p\n", &data);
    test_1();
   
    printf("start of elf is %p, main addr is %p\n", __ehdr_start, __func__);  //__ehdr_start is start of elf file

    int i = 0;
    for(i = 0; i < 20; i++) {   
        buf[i] = i;
        data = i;
    }
 
    for(i = 0; i < 20; i++) {   
        buf[i] = i;
        s_data=i;
    }

}

int main()
{
    test();

    sleep(3);

    return 0;
}
