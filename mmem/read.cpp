#include <iostream>
#include <thread>
#include "mmem.h"

using namespace std;

struct MyData
{
    char name[20];
    int age;
};

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
   
    this_thread::sleep_for(chrono::seconds(10));

    cout << "read shared data: " << endl;
    MyData *share_buffer = (MyData *)p;
    cout << share_buffer->name << " " << share_buffer->age << endl;

    free_memmap(fd, shared_fd, p, size);
}

int main()
{
    readMemory();
    getchar();
    return 0;
}
