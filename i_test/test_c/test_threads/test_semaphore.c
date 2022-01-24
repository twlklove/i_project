#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>  //
#include <semaphore.h> //

sem_t sem; 


void *get_service_1(void *thread_id) 
{
    int customer_id = *((int*)thread_id);
    if (customer_id % 2) {
        while(1) {
            printf("customer %d wait ...\n", customer_id);
            if (sem_wait(&sem) == 0) { //
                sleep(10);
                sem_post(&sem);  //
            }
        }
    }
    else {
        while(1) {
            printf("customer %d wait ...\n", customer_id);
            if (sem_wait(&sem) == 0) { //
                sleep(1);
                sem_post(&sem);  //
            }
        }
    }
}

#define CUSTOMER_NUM 3 //10
int main()
{
    sem_init(&sem, 0, CUSTOMER_NUM); //
    pthread_t customers[CUSTOMER_NUM];
    int i, ret;
    for (i = 0; i< CUSTOMER_NUM; i++) {
        int customer_id = i;
        ret = pthread_create(&customers[i], NULL, get_service_1, &customer_id);
        if (0 != ret) {
            printf("fail to create thread\n");
            return -1;
        }
        else {
            printf("customer %d arrived\n", i);
        }

        usleep(10);
    }

    int j;
    for (j = 0; j < CUSTOMER_NUM; j++){
        pthread_join(customers[j], NULL);
    }

    sem_destroy(&sem); //
    return 0;
}
