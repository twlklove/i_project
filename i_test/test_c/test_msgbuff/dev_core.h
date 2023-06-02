#ifndef __DEV_CORE_H__
#define __DEV_CORE_H__

#include "msg_buff.h"
#include "workqueue.h"
#include "types.h"

#define WQ_HIGHPRI   (1 << 4)

struct comm_dev {
    struct list_head list;
    mutex_t mutex;
    char name[8];   

    struct msg_buff_head rx_q;
    struct workqueue_struct *workqueue;
    struct work_struct rx_work;
};

struct comm_dev* comm_alloc_dev(void);
void comm_free_dev(struct comm_dev *comm_dev);

int comm_register_dev(struct comm_dev *comm_dev);
void comm_unregister_dev(struct comm_dev *comm_dev);

int comm_recv_data(struct comm_dev *p_dev, struct msg_buff *msgb);

#endif
