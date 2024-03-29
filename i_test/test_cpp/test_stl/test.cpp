#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>  //
#include <semaphore.h> //
#include <map>
#include <queue>
#include <iostream>
using namespace std;

#define THREAD_NUM 8 
sem_t sem; 
queue<int> data;

void *service(void *thread_id) 
{
    while(1) {
        if (sem_wait(&sem) == 0) { //
            data.front();
            cout << "pop" << endl;
            data.pop();
            //sem_post(&sem);  //
        }
    }
}

void *service_1(void *thread_id) 
{
    while(1) {
            cout << "push" << endl;
            data.push(1);
            sem_post(&sem);  //
    }
}

typedef void* (*func)(void *thread_id);

func funcs[THREAD_NUM] = {service, service, service_1, service_1,service, service, service_1, service_1};

int main()
{
    sem_init(&sem, 0, 0);//THREAD_NUM); //
    pthread_t threads[THREAD_NUM];
    int i, ret;
    for (i = 0; i< THREAD_NUM; i++) {
        int thread_id = i;
        ret = pthread_create(&threads[i], NULL, funcs[i], &thread_id);
        if (0 != ret) {
            printf("fail to create thread\n");
            return -1;
        }
        else {
            printf("successfull to creat thread %d\n", i);
        }

        usleep(1);
    }

    int j;
    for (j = 0; j < THREAD_NUM; j++){
        pthread_join(threads[j], NULL);
    }

    sem_destroy(&sem); //
    return 0;
}

