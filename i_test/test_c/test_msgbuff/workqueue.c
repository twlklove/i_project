#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/resource.h>

#include "types.h"
#include "workqueue.h"
#include "list.h"
#include "types.h"
#include "compiler_attributes.h"

struct worker_pool {
    mutex_t mutex;
    struct list_head worklist;

    struct list_head workers;
};

struct worker {
    union {
        struct list_head entry; //whiel idle
    };

    struct work_struct *current_work;
    work_func_t current_func;
    struct pool_workqueue *current_pwq;
    
    pthread_t thread_id;
    struct worker_pool *pool;
    struct list_head node;
};

struct pool_workqueue {
    struct worker_pool *pool;
    struct workqueue_struct *wq;
};

#define nr_node_ids                1U
#define WQ_NAME_LEN                24


struct workqueue_struct {
    //...
    struct list_head list;
    char name[WQ_NAME_LEN];
    u32 flags;
    struct pool_workqueue *cpu_pwqs[NR_CPUS]; // per-cpu pwqs
    struct poll_workqueue *numa_pwq_tbl[]; // unbond pwqs indexed by node
};

struct worker_pool cpu_worker_pools[NR_CPUS][NR_STD_WORKER_POOLS];

static LIST_HEAD(workqueues);

static void init_pwq(struct pool_workqueue *pwq, struct workqueue_struct *wq, struct worker_pool *pool) 
{
    memset(pwq, 0, sizeof(*pwq));
    pwq->pool = pool;
    pwq->wq = wq;
}

static int alloc_and_link_pwqs(struct workqueue_struct *wq)
{
    int highpri = 0;

    if (!((wq->flags) & WQ_UNBOUND)) {
        ;
    }

    wq->cpu_pwqs[0] = malloc(sizeof(struct pool_workqueue));
    if (!wq->cpu_pwqs) {
        return -1;
    }
    
    struct pool_workqueue *pwq =  wq->cpu_pwqs[0];
    struct worker_pool *cpu_pools = cpu_worker_pools[0];
    init_pwq(pwq, wq, &cpu_pools[highpri]);

    return 0;
}

__printf(1, 4)
struct workqueue_struct *alloc_workqueue(const char *fmt, unsigned int flags, int max_active, ...)
{
    int ret = 0; 
    u32 tbl_size = 0;
    struct workqueue_struct *wq = NULL;
    
    if (flags & WQ_UNBOUND) {
        tbl_size = nr_node_ids * sizeof(wq->numa_pwq_tbl[0]);
    }

    wq = malloc(sizeof(*wq) + tbl_size);
    if (!wq) {
        return NULL;
    }

    va_list args;
    va_start(args, max_active);
    vsnprintf(wq->name, sizeof(wq->name), fmt, args);
    va_end(args);

    INIT_LIST_HEAD(&wq->list);

    do {
        if (alloc_and_link_pwqs(wq) < 0) {
            ret = -1;
            break;       
        }

        list_add_tail(&wq->list, &workqueues);
    } while (0);
    
    if (0 != ret) {
        free(wq);
    }

    return wq;
}


void destroy_workqueue(struct workqueue_struct *wq)
{
    list_del_init(&wq->list);
    struct pool_workqueue *pwq =  wq->cpu_pwqs[0];
    free(pwq);
    free(wq);
}

static void insert_work(struct pool_workqueue *pwq, struct work_struct *work, struct list_head *head, u32 extra_flags)
{
    struct worker_pool *pool = pwq->pool; 
    list_add_tail(&work->entry, head);
}

static void __queue_work(int cpu, struct workqueue_struct *wq, struct work_struct *work)
{
    struct pool_workqueue *pwq = NULL; 
    struct list_head *worklist = NULL;
    u32 work_flags = 0;

    int index = 0;
    if (WORK_CPU_UNBOND == cpu) {
        index = 0;
    }

    if (!list_empty(&work->entry)) {
        return;
    } 

    printf("insert work\n");
    pwq = wq->cpu_pwqs[index];
    worklist = &pwq->pool->worklist;   
    insert_work(pwq, work, worklist, work_flags);
}

bool queue_work_on(int cpu, struct workqueue_struct *wq, struct work_struct *work)
{
    bool ret = true;

    __queue_work(cpu, wq, work);

    return ret;
}

static struct pool_workqueue *get_work_pwq(struct work_struct *work)
{
#if 0
    unsigned long data = &work->data;
    if (data & WORK_STRUCT_PWQ) {
        return (void*)(data & WORK_STRUCT_WQ_DATA_MASK);
    }
    else {
        return NULL;
    }
#endif
}

void process_one_work(struct worker *worker, struct work_struct *work)
{ 
    //struct pool_workqueue *pwq = get_work_pwq(work);
    struct worker_pool *pool = worker->pool;

    worker->current_work = work;
    worker->current_func = work->func;
    //worker->current_pwq = pwq;
   
    pthread_mutex_unlock(&pool->mutex);

    if (work->func) {
        worker->current_func(work);
        printf("finish process one work\n");
    }

    list_del_init(&work->entry);

    pthread_mutex_unlock(&pool->mutex);
 
    worker->current_work = NULL;
    worker->current_func = NULL;
    worker->current_pwq = NULL;
}

static bool keep_working(struct worker_pool *pool)
{
    return !list_empty(&pool->worklist);
}

static void cleanup_handler(void *arg)
{
    struct worker *p_worker = arg;
    struct worker_pool *pool = p_worker->pool;
    (void)pthread_mutex_unlock(&pool->mutex);
}

void *worker_thread(void *_worker)
{
    struct worker *p_worker = _worker;
    struct worker_pool *pool = p_worker->pool;

     //getpid/gettid() is kernel thread id, pthread_self() is posix thread id, is better
    printf("process id is %d, posix thread id is %lu, thread id is %d \n", getpid(), pthread_self(), gettid());   
    
    if (0 != setpriority(PRIO_PROCESS, 0, -1)) {  // -19~20
        printf("fail to set priority\n");
    }

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(4, &mask);
    CPU_SET(5, &mask);
    CPU_SET(6, &mask);
    CPU_SET(7, &mask);

    if (0 != sched_setaffinity(0, sizeof(mask), &mask)) {
        printf("fail to set affinity\n");
    } 

    printf("thread is running on cpu %d now\n", sched_getcpu());
 
    pthread_cleanup_push(cleanup_handler, _worker);

    while (1) { 
        pthread_mutex_lock(&pool->mutex);
        while (keep_working(pool)) {
            struct work_struct *work = list_first_entry(&pool->worklist, struct work_struct, entry);
            if (!work) {
                continue;
            }

            process_one_work(p_worker, work);
        }

        pthread_mutex_unlock(&pool->mutex);
        usleep(1000);
    };

    pthread_cleanup_pop(0);
}

static void  worker_attach_to_pool(struct worker *worker, struct worker_pool *pool) 
{
    list_add_tail(&worker->node, &pool->workers);
    worker->pool = pool;
}

static struct worker *create_worker(struct worker_pool *pool)
{
    struct worker *p_worker = malloc(sizeof(struct worker));
    if (NULL == p_worker) {
        return NULL;
    }
  
    
    pthread_mutex_init(&pool->mutex, NULL);
    INIT_LIST_HEAD(&pool->worklist); //
    INIT_LIST_HEAD(&pool->workers);

    INIT_LIST_HEAD(&p_worker->entry);
    INIT_LIST_HEAD(&p_worker->node);
    worker_attach_to_pool(p_worker, pool);

    int ret = pthread_create(&(p_worker->thread_id), NULL, worker_thread, p_worker);
    if (0 != ret) {
        free(p_worker);
        return NULL;
    }

    char name[20] = {0};
    static int id = 0;
    snprintf(name, sizeof(name), "uworker/%d", id++);
    pthread_setname_np(p_worker->thread_id, name); 
                                                         
    return p_worker;
}

#define for_each_online_cpu(cpu)                 \
    for ((cpu) == 0; (cpu) < NR_CPUS; (cpu)++)

#define for_each_cpu_worker_pool(pool, cpu)      \
    for ((pool) = &cpu_worker_pools[cpu][0]; (pool) < &cpu_worker_pools[cpu][NR_STD_WORKER_POOLS]; (pool)++)

void workqueue_init(void)
{
    struct worker *worker = NULL;
    struct worker_pool *pool = NULL;

    int cpu = 0;
    for_each_online_cpu(cpu) {
        for_each_cpu_worker_pool(pool, cpu) {
            worker = create_worker(pool);
            if (NULL == worker) {
                break;
            }
        }
    }
}

static void destroy_worker(struct worker *worker)
{
    struct worker_pool *pool = worker->pool;
    void *res = NULL;

    int ret = pthread_cancel(worker->thread_id);
    if (0 != ret) {
        printf("failt to cancel\n");
    }

    ret = pthread_join(worker->thread_id, &res); // res == PTHREAD_CANCELED
    if (0 != ret) {
        printf("fail to join\n");
    }

    INIT_LIST_HEAD(&pool->worklist);
    list_del(&worker->node);
    list_del_init(&worker->entry);
    free(worker);
}

#define for_each_pool_worker(worker, pool) \
    list_for_each_entry((worker), &(pool)->workers, node)

void workqueue_uninit(void)
{
    struct worker *worker = NULL;
    struct worker_pool *pool = NULL;

    int cpu = 0;
    for_each_online_cpu(cpu) {
        for_each_cpu_worker_pool(pool, cpu) {
            for_each_pool_worker(worker, pool) {
                if(NULL == worker) {
                    continue;
                }

                destroy_worker(worker);
            }
        }
    }
}
