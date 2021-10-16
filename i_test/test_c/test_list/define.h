#ifndef __I_DEFINE_H__
#define __I_DEFINE_H__

#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)

#define container_of(ptr, type, member) ({				\
	void *__mptr = (void *)(ptr);					\
	((type *)(__mptr - offsetof(type, member))); })


#endif
