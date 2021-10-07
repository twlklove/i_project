#include <stdio.h>
#include <stdlib.h>

typedef void (*CALLBACKFUNC)(void *p_args);

void event_callback(void *p_args)
{
    if (NULL == p_args) {
        printf("pointer is NULL\n");
	return;
    }

    printf("%s\n", (char*)p_args);
}


#define MAX_EVENT_NUM 10
CALLBACKFUNC p_back_func[MAX_EVENT_NUM] = {NULL};

int register_event_back(int event_type, CALLBACKFUNC p_func)
{
    if (event_type >= MAX_EVENT_NUM) {
	printf("event_type is invalid\n");
        return -1;
    }
    p_back_func[event_type] = p_func;
    return 0;
}

int main()
{
    int event_type = 1;
    void *p_args="hello, world";
    int ret = register_event_back(event_type, event_callback);
    
    p_back_func[event_type](p_args); 

    return ret;
}

