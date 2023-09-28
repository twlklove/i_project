#include <stdio.h>

#define LOG_MARKER(x) {int line=__LINE__; int level=x;
#define LOG(format, ...)\
do {\
    printf("[level:%d line:%d] "format, level, line,__VA_ARGS__);\
} while(0);}


#define ILOGI LOG_MARKER(1) LOG
#define ILOGE LOG_MARKER(2) LOG

int main() 
{
    int x = 10;
    int y = 20; 
    ILOGI("x is %d, \
            y is %d\n",
                                x, y);

    ILOGE("x is %d\n", x);

    return 0;
}
