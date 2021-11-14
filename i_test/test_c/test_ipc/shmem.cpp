#include <sys/shm.h>
int share_mem_id = 26

key_t key = ftok(".", share_mem_id); // get key value

int shmid = shmget(key, MAX_SIZE, 0644 | IPC_CREAT); // create mem or get mem
if(shmid == -1)
{
    exit(1);
}
 
addr = (char *)shmat(shmid,NULL,0); // map phy mem into vir mem


int ret = shmdt(addr);  //unmap
ret = shmctl(shmid, IPC_RMID, NULL); //free mem
if(shmid == -1)
{
    exit(1);
}

