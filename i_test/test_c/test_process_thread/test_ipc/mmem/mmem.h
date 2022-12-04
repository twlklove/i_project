#ifndef __MMEM_H__
#define __MMEM_H__

#include <iostream>
#include <thread>
#include <string>
#include "mmem.h"
#ifdef _WIN32
#include <windows.h>
#elif __linux
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif
 
int get_memmap(const char *name, int *p_fd, int *p_shared_fd, void **p, int size);

void free_memmap(int fd, int shared_fd, void *p, int size);

#endif
