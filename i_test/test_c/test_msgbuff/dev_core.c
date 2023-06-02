#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "dev_core.h"
#include "msg_buff.h"
#include "workqueue.h"

static void comm_rx_work(struct work_struct *work)
{
    struct comm_dev *p_dev = container_of(work, struct comm_dev, rx_work);
    struct msg_buff *msgb;

    while((msgb = msgb_dequeue(&p_dev->rx_q))) {
        printf("len is %d\n", msgb->len);
        free(msgb);
    }
}

struct comm_dev* comm_alloc_dev(void)
{
    struct comm_dev *p_dev = NULL;

    p_dev = malloc(sizeof(struct comm_dev));
    if (!p_dev) {
        return NULL;
    }

    pthread_mutex_init(&p_dev->mutex, NULL); 
    INIT_WORK(&p_dev->rx_work, comm_rx_work);
    msgb_queue_head_init(&p_dev->rx_q); 

    return p_dev;
}

void comm_free_dev(struct comm_dev *comm_dev)
{
    if (!comm_dev) {
        free(comm_dev);
    }
}

int comm_register_dev(struct comm_dev *comm_dev)
{
    int ret = 0;

    sprintf(comm_dev->name, "dev%d", 0);

    do {
        comm_dev->workqueue = alloc_ordered_workqueue("%s", WQ_HIGHPRI, comm_dev->name);
        if (!comm_dev->workqueue) {
            ret = -1;
            break;
        }
    } while (0);

    return ret;
}

void comm_unregister_dev(struct comm_dev *comm_dev)
{
    destroy_workqueue(comm_dev->workqueue);
}

int comm_recv_data(struct comm_dev *comm_dev, struct msg_buff *msgb)
{
    msgb_queue_tail(&comm_dev->rx_q, msgb);
    queue_work(comm_dev->workqueue, &comm_dev->rx_work); 

    return 0;
}

