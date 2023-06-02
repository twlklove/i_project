#ifndef __WORKQUEUE_H__
#define __WORKQUEUE_H__

#include <stdbool.h>
#include "list.h"

struct workqueue_struct;
struct work_struct;

#define __INIT_WORK(_work, _func, _onstack)        \
    INIT_LIST_HEAD(&(_work)->entry);               \
    (_work)->func = (_func);

#define INIT_WORK(_work, _func) \
    __INIT_WORK((_work), (_func), 0)

typedef void (*work_func_t)(struct work_struct *work);

struct work_struct {
    u32 data;
    struct list_head entry;
    work_func_t func;
};


#define alloc_ordered_workqueue(fmt, flags, args...)       \
    alloc_workqueue(fmt, flags, 1, ##args)

void destroy_workqueue(struct workqueue_struct *wq);

struct workqueue_struct *alloc_workqueue(const char *fmt, unsigned int flags, int max_active, ...);

#define WQ_UNBOUND    (1 << 1)
#define NR_CPUS        2
#define WORK_CPU_UNBOND NR_CPUS
#define NR_STD_WORKER_POOLS        1 //2

bool queue_work_on(int cpu, struct workqueue_struct *wq, struct work_struct *work);

static inline bool queue_work(struct workqueue_struct *wq, struct work_struct *work)
{
    return queue_work_on(WORK_CPU_UNBOND, wq, work);
}

void workqueue_init(void);
void workqueue_uninit(void);

#endif
