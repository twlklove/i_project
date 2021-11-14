#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>
 
#define MAX_SIZE 1024
 
char *addr;
int shmid;
 
void my_exit(int signal)
{
     int ret = shmdt(addr);  //unmap
	 if (-1 == ret) {
	     printf("fail to map\n");
		 return
	 }

     ret = shmctl(shmid, IPC_RMID, NULL); //free mem
	 if (-1 == ret) {
	     printf("fail to map\n");
		 return
	 }
	 exit(0);
}
 
int main()
{    
     signal(SIGINT, my_exit); 
	 signal(SIGKILL, my_exit); 
	 signal(SIGTERM, my_exit);
     
     char buffer[MAX_SIZE];
 
     key_t key = ftok(".", 26); // get key value
	 printf("%x\n", key);
 
     shmid = shmget(key, MAX_SIZE, 0644 | IPC_CREAT); // create mem or get mem
     if(shmid == -1)
     {
         perror("shm get error!");
	     exit(1);
     }
 
     pid_t pid = fork();
     if(pid < 0)
     {
         perror("fork error!");
	     exit(1);
     }
 
     if(pid == 0)
     {
          addr = (char *)shmat(shmid,NULL,0); // map phy mem into vir mem

	      while(1)
	      {
	           memset(buffer,0,sizeof(buffer));
	           if(strlen(addr) != 0)
	           {
	               strcpy(buffer,addr);
	               printf("recv:%s\n",buffer);
		           memset(addr,0,MAX_SIZE);
               }
	           sleep(1);
	      }
     }
     else if(pid > 0)
     {
          addr = (char *)shmat(shmid,NULL,0); // 4
	  
	      while(1)
	      {
	           memset(buffer,0,sizeof(buffer));
	           scanf("%s",buffer);
               strcpy(addr,buffer);
			   sleep(1);
	      }
     }
 
     return 0;
}  

