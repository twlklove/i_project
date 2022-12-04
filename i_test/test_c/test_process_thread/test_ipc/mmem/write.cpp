#include <iostream>
#include <thread>
#include "mmem.h"

using namespace std;

struct MyData
{
    char name[20];
    int age;
};

void writeMemory()
{
    const char *name = "my_shared_memory";
    MyData share_buffer = { "Tom", 18 };

    int fd = 0;
    int shared_fd = 0;
    const unsigned long size = 4096;
    void *p = NULL; 
    int result = get_memmap(name, &fd, &shared_fd, &p, size);
    if (0 != result) {
        return;
    }
    
    size_t write_size = sizeof(share_buffer);
    
    memcpy(p, &share_buffer, write_size);

    cout << "already write to shared memory, wait ..." << endl;
    this_thread::sleep_for(chrono::seconds(10));

    free_memmap(fd, shared_fd, p, size);
}

int main()
{
    writeMemory();
    return 0;
}
