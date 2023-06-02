#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/queue.h>

#if 0
In file : /usr/include/x86_64-linux-gnu/sys/queue.h
SLIST:   singly linked lists (SLIST)
LIST:    doubly linked lists (LIST)
STAILQ:  singly linked tail queues (STAILQ)
TAILQ:   doubly linked tail queues (TAILQ)
CIRCLEQ: doubly linked circular queues (CIRCLEQ)

#define	CIRCLEQ_HEAD(name, type)					\
struct name {								\
	struct type *cqh_first;		/* first element */		\
	struct type *cqh_last;		/* last element */		\
}

#define	CIRCLEQ_HEAD_INITIALIZER(head)					\
	{ (void *)&head, (void *)&head }

#define	CIRCLEQ_ENTRY(type)						\
struct {								\
	struct type *cqe_next;		/* next element */		\
	struct type *cqe_prev;		/* previous element */		\
}

#define	CIRCLEQ_INIT(head) do {						\
	(head)->cqh_first = (void *)(head);				\
	(head)->cqh_last = (void *)(head);				\
} while (/*CONSTCOND*/0)
#endif

struct entry { 
    CIRCLEQ_ENTRY(entry) entries;           /* Queue */
    int data;
};

CIRCLEQ_HEAD(circlehead, entry);

int
main(void)
{
    struct entry *n0, *n1, *n2, *n3, *np;
    struct circlehead head;                 /* Queue head */
    int i;

    CIRCLEQ_INIT(&head);                    /* Initialize the queue */

    n0 = malloc(sizeof(struct entry));      /* Insert at the head */
    n0->data = 0;
    CIRCLEQ_INSERT_HEAD(&head, n0, entries);

    n1 = malloc(sizeof(struct entry));      /* Insert at the tail */
    n1->data = 1;
    CIRCLEQ_INSERT_TAIL(&head, n1, entries);

    n2 = malloc(sizeof(struct entry));      /* Insert after */
    n2->data = 2;
    CIRCLEQ_INSERT_AFTER(&head, n1, n2, entries);

    n3 = malloc(sizeof(struct entry));      /* Insert before */
    n3->data = 3;
    CIRCLEQ_INSERT_BEFORE(&head, n2, n3, entries);
    CIRCLEQ_FOREACH(np, &head, entries)
        printf("%i ", np->data);
    printf("\n");

    CIRCLEQ_REMOVE(&head, n2, entries);     /* Deletion */
    free(n2);
                                            /* Forward traversal */
    /* Reverse traversal */
    CIRCLEQ_FOREACH_REVERSE(np, &head, entries)
        printf("%i ", np->data);
    printf("\n");                                        /* Queue deletion */

    n1 = CIRCLEQ_FIRST(&head);
    while (n1 != (void *)&head) {
        n2 = CIRCLEQ_NEXT(n1, entries);
        free(n1);
        n1 = n2;
    } 
    CIRCLEQ_INIT(&head);

    exit(EXIT_SUCCESS);
}
