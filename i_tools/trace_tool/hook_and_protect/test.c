#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef __USE_GNU
#define __USE_GNU
#include <dlfcn.h>
#undef __USE_GNU
#endif

#include <sys/types.h>
#include <sys/mman.h>


#ifdef DEBUG
#define ALIGNED __attribute__((aligned(4096)))
#else
#define ALIGNED  //(__attribute__((aligned)))
#endif

typedef void (*func)(int *p_data);

func original_func = NULL;
long addr = 0;
int hooked_func(int *p_data)
{ 
    int test_data ALIGNED; 
    printf("%s: the addr of test_data is %p\n", __func__, &test_data);

#ifdef DEBUG
    static long page_size = 0;
    page_size = sysconf(_SC_PAGESIZE);
    static void * start_page_addr = NULL;
    start_page_addr = (void*)((long)&test_data - page_size);
    printf("%s: page size is 0x%lx, the start addr proctected is %p\n", __func__, page_size, start_page_addr);
    mprotect(start_page_addr, page_size, PROT_READ);
#endif

    if (*p_data == 3) {
        pid_t pid = getpid();
        Dl_info info;
        dladdr((void *)original_func, &info);
        const char *functionName = info.dli_sname;
        printf("process %d, function %s\n", pid, functionName);
    }

    original_func(p_data); 
    if (test_data != 0) {
        printf("%s : test_data is changed to %d\n", __func__, test_data);
    }
}

int main()
{
    void *libc = dlopen("./libtest.so", RTLD_LAZY);
    original_func = (func)dlsym(libc, "test");
    func original_func_tmp = original_func;
    int data = 0;

    // prepare for changing 
    long pagesize = sysconf(_SC_PAGESIZE); 
    long function_addr = (long)original_func_tmp;
    long page_offset = function_addr % pagesize;

    // do change
    void *pageStartAddr = (void *)(function_addr - page_offset);
    mprotect(pageStartAddr, pagesize, PROT_READ | PROT_WRITE | PROT_EXEC);
    *(void **)(&original_func_tmp) = hooked_func;
    mprotect(pageStartAddr, pagesize, PROT_READ | PROT_EXEC);

    printf("func addr is changed from %p to %p\n", original_func, original_func_tmp);
    original_func_tmp(&data);

    //void *mem = mmap(NULL, page_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    // recover
    mprotect(pageStartAddr, pagesize, PROT_READ | PROT_WRITE | PROT_EXEC);
    *(void **)(&original_func_tmp) = original_func;
    mprotect(pageStartAddr, pagesize, PROT_READ | PROT_EXEC);

    dlclose(libc);

    return 0;
}
