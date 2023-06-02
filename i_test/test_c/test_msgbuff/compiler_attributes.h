#ifndef __COMPILER_ATTRIBUTES_H__
#define __COMPILER_ATTRIBUTES_H__
 
#define __printf(a, b)  __attribute__((__format__(printf, a, b)))

#endif
