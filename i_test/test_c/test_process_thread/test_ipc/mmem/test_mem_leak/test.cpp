#include <iostream>
#include <thread>
#include "mmem.h"
#include <mcheck.h>

using namespace std;

struct MyData
{
    char name[20];
    int age;
};
void test();
void readMemory();
int main()
{
    mtrace();  //

    readMemory();
	test();

	muntrace(); //
    return 0;
}

void test()
{
    int *p_i = (int*)malloc(10);
	int *p_j = (int*)new int[10];
}
void readMemory()
{
    const char *name = "my_shared_memory";
    int fd = 0;
    int shared_fd = 0;
    const unsigned long size = 4096;
    void *p = NULL; 
    int result = get_memmap(name, &fd, &shared_fd, &p, size);
    if (0 != result) {
        return;
    }

	while(1) {
	    sleep(1);
	}
}


