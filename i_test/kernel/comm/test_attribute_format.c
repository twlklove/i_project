/* __attribute__ ((format (printf, 2, 3)))
1. 功能：
   __attribute__ format属性可以给被声明的函数加上类似printf或者scanf的特征，
   它可以使编译器检查函数声明和函数实际调用参数之间的格式化字符串是否匹配。
   format属性告诉编译器，按照printf, scanf等标准C函数参数格式规则对该函数的参数进行检查。

2. format的语法格式为：
   format (archetype, string-index, first-to-check) 其中，
        “archetype”指定是哪种风格；
        “string-index”指定传入函数的第几个参数是格式化字符串；
        “first-to-check”指定从函数的第几个参数开始按上述规则进行检查。
3. 具体的使用如下所示：
   __attribute__((format(printf, m, n)))
   __attribute__((format(scanf, m, n)))

　　　　m：第几个参数为格式化字符串(format string);
　　　　n：参数集合中的第一个，即参数“…”里的第一个参数在函数参数总数排在第几。注意，有时函数参数（类成员函数）里还有“隐身”的（this指针）；

   一般函数：
    extern void myprint(const char *format,...) __attribute__((format(printf,1,2)));   //m=1；n=2
    extern void myprint(int l，const char *format,...) __attribute__((format(printf,2,3)));  //m=2；n=3
    
   类成员函数:类成员函数的第一个参数实际上一个“隐身”的“this”指针。
    extern void myprint(int l，const char *format,...) __attribute__((format(printf,3,4)));   
    
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#define __printf(a, b) __attribute__((format(printf, a, b))) // be defined in include/linux/compiler_attributes.h  

static __printf(1, 2)
void myprintf(const char *fmt,...) 
{
    va_list args;
    va_start(args, fmt);

#ifdef DEBUG
    va_list ap;
    va_copy(ap, args);
    
    while (*fmt) {
        switch (*fmt++) {
        case 's':              /* string */
            char *s = va_arg(ap, char *);
            printf("string %s\n", s);
            break;
        case 'd':              /* int */
            int d = va_arg(ap, int);
            printf("int %d\n", d);
            break;
        case 'c':              /* char */
            /* need a cast here since va_arg only
               takes fully promoted types */
            char c = (char) va_arg(ap, int);
            printf("char %c\n", c);
            break;
        }
    }
    va_end(ap);
#else
    vdprintf(1, fmt, args);  //man vdprintf
    va_end(args);
#endif
}

void main()
{
    myprintf("i=%d, %d\n", 6, 7);
    //myprintf("i=%s\n",6);
    //myprintf("i=%s\n","abc");
    //myprintf("%s,%d,%d\n",1,2);
}
