#include <iostream>
#include <chrono>

#ifdef _WIN32
#include <windows.h>
#elif __linux
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace std;

struct MyData
{
    char name[20];
    int age;
};

int get_memmap(const char *name, int *p_fd, int *p_shared_fd, void **p, int size)
{
    int result = 0;
    if ((NULL == p_fd) || (NULL == p_shared_fd) || (NULL == p)) {
        cout << "null poniter!" << endl; 
        result = -1;
        return result; 
    }

#ifdef __linux
    int fd = open(name, O_CREAT | O_RDWR | O_TRUNC, 00777);
    if (fd < 0) {
       cout << "create file error" << endl;
       result = -1;
    }
    ftruncate(fd, size);

    *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (NULL == *p) {
       close(fd);
       result = -1;
    }
    
    *p_fd = fd;

#elif _WIN32
    HANDLE fd = CreateFile(name,
        GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (INVALID_HANDLE_VALUE == fd) {
        cout << "create file error" << endl;
        result = -1;
        return result;
    }

    HANDLE shared_fd = CreateFileMapping(
        fd, NULL, PAGE_READWRITE, 0, buff_size, shared_file_name);
    if (INVALID_HANDLE_VALUE == shared_file_hd)
    {
        cout << "create file error" << endl;
        CloseHandle(shared_fd);
        result = -1;
        return result; 
    }

    LPVOID *p = MapViewOfFile(shared_fd, FILE_MAP_ALL_ACCESS, 0, 0, buff_size);
    if (NULL == *p) {
        result = -1;
        CloseHandle(shared_fd);
        CloseHandle(fd);
    }

    *p_fd = fd;
    *p_shared_fd = shared_fd; 
#endif
    
    return result;
}

void free_memmap(int fd, int shared_fd, void *p, int size) {
#ifdef __linux
    if (NULL != p) {
        munmap(p, size);
    }

    close(fd);
#elif _WIN32
    if (NULL != p) {
        UnmapViewOfFile(p);;
    }

    CloseHandle(shared_fd);
    CloseHandle(fd);
#endif
}

