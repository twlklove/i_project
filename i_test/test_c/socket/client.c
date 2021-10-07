#include <sys/stat.h> 
#include <fcntl.h> 
#include <errno.h> 
#include <netdb.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <netinet/in.h> 
#include <arpa/inet.h> 
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <unistd.h>
 
#define SERVER_PORT 6666
#define SERVER_ADDR "127.0.0.1"  // local address 
  
int main()	 
{ 
    int client_socket; 
    struct sockaddr_in server_addr; 
    char sendbuf[200];
    char recvbuf[200];
    int  data_num;
    
    if((client_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) 
    {  
        perror("socket");   
        return 1; 
    }
         
    server_addr.sin_family = AF_INET;      
    server_addr.sin_addr.s_addr = inet_addr(SERVER_ADDR);      
    server_addr.sin_port = htons(SERVER_PORT);       
    if(connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) 
    {   
        perror("connect"); 
        return 1;
    }
          
    while(1) 
    {   
        printf("send message:");

        scanf("%s", sendbuf);
        printf("\n");
        send(client_socket, sendbuf, strlen(sendbuf), 0); 

        if (0 == strcmp(sendbuf, "quit")) {
            break;
	}
          
        printf("读取消息:");   
   
        data_num = recv(client_socket, recvbuf, 200, 0); 
        recvbuf[data_num] = '\0';       
        printf("%s\n", recvbuf); 
    }
          
    close(client_socket);       
    return 0;        
}
