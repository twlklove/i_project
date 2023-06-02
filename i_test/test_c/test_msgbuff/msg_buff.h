#ifndef __MSG_BUFF_H__
#define __MSG_BUFF_H__

#include <pthread.h>
#include "list.h"
#include "types.h"

#define MAX_MSGB_FRAGS 16UL 

typedef struct {
    u32 data_len;  //max_len is 4KB
    u8 *p_data;
}msgb_frag_t;

struct msg_buff {
    union {
        struct {
            /* these two members must be fist */
            struct msg_buff *next;
            struct msg_buff *prev;
        };

        //struct rb_node rbnode;  // for sort
        struct list_head list;
    };

   
    u32 len;
    u32 data_len;

    /* these members must be at the end */
    u8 *head;
    u8 *data;
    u8 *tail;
    u8 *end;
};

struct msgb_shared_info {
    u8 nr_frags;
    struct msg_buff *frag_list;
    msgb_frag_t frags[MAX_MSGB_FRAGS]; // max is 16 * 4KB = 64KB = 2^16
};

struct msg_buff_head {
    /* these two members must be fist */
    struct msg_buff *next;
    struct msg_buff *prev;

    u32 qlen;
    mutex_t mutex;
};

static inline void __msgb_queue_head_init(struct msg_buff_head *list)
{
    list->prev = list->next = (struct msg_buff*)list;
    list->qlen = 0;
}

static inline void msgb_queue_head_init(struct msg_buff_head *list)
{
    pthread_mutex_init(&list->mutex, NULL);
    __msgb_queue_head_init(list); 
}

static inline void __msgb_insert(struct msg_buff *newmsg, struct msg_buff *prev, struct msg_buff *next, struct msg_buff_head *list)
{
    newmsg->next = next;
    newmsg->prev = prev;
    next->prev = newmsg;
    prev->next = newmsg;
    list->qlen++;
}

static inline void __msgb_queue_before(struct msg_buff_head *list, struct msg_buff *next, struct msg_buff *newmsg)
{
    __msgb_insert(newmsg, next->prev, next, list);
}

static inline void __msgb_queue_tail(struct msg_buff_head *list, struct msg_buff *newmsg)
{
    __msgb_queue_before(list, (struct msg_buff*)list, newmsg);
}

static inline void msgb_queue_tail(struct msg_buff_head *list, struct msg_buff *newmsg)
{
    pthread_mutex_lock(&list->mutex);
    __msgb_queue_tail(list, newmsg);
    pthread_mutex_unlock(&list->mutex);
}

static inline struct msg_buff *msgb_peek(const struct msg_buff_head *list)
{
    struct msg_buff *msgb = list->next;
    if (msgb == (struct msg_buff*)list) {
        msgb = NULL;
    }

    return msgb;
}

static inline void __msgb_unlink(struct msg_buff *msgb, struct msg_buff_head *list)
{
    struct msg_buff *next, *prev;
    list->qlen--;
    next = msgb->next;
    prev = msgb->prev;
    msgb->next = msgb->prev = NULL;
    next->prev = prev;
    prev->next = next;
}

static inline struct msg_buff *__msgb_dequeue(struct msg_buff_head *list)
{
    struct msg_buff *msgb = msgb_peek(list);
    if (msgb)
        __msgb_unlink(msgb, list);

    return msgb;
}

static inline struct msg_buff *msgb_dequeue(struct msg_buff_head *list)
{
    struct msg_buff *result = NULL;

    pthread_mutex_lock(&list->mutex);
    result = __msgb_dequeue(list);
    pthread_mutex_unlock(&list->mutex);

    return result;
}
#endif
