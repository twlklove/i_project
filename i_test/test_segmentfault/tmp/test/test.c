#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern char __ehdr_start[];
int buf[100] = {0};

int main()
{
    int data[100] = {0};
    
    printf("buf addr is %p, data addr is %p\n",  buf, data);
    int i = 0;
    while(1) {
        for (i = 0; i < 100; i++) {
            buf[i] = 0x100 + i;
            data[i] = 0x100 + i;
            if (2 == i) { 
                printf("buff[%d] addr is %p, value is %x\n", i, &buf[i], buf[i]);
            }

            sleep(1);
        }
    }

    return 0;
}

