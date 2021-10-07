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

int main()
{
    int server_socket;
    struct sockaddr_in server_addr;
    struct sockaddr_in client_addr;
    int addr_len = sizeof(client_addr);
    int client;
    char buffer[200];
    int data_num;

    if((server_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("socket");
        return 1;
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(SERVER_PORT);
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY); //INADDR_ANY : 0.0.0.0, all addresses
    if(bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        perror("connect");
        return 1;
    }

    if(listen(server_socket, 5) < 0)
    {
        perror("listen");
        return 1;
    }
          
    while(1)
    {
        printf("listen port: %d\n", SERVER_PORT);
        client = accept(server_socket, (struct sockaddr*)&client_addr, (socklen_t*)&addr_len); 
        if(client < 0)
        {
            perror("accept");
            continue;
        }
        
	printf("waiting message...\n");
        printf("IP is %s\n", inet_ntoa(client_addr.sin_addr));
        printf("Port is %d\n", htons(client_addr.sin_port));
                  
        while (1) {
            printf("read message:");
            buffer[0] = '\0';
            data_num = recv(client, buffer, 1024, 0);
            if(data_num < 0)
            {
                perror("recv null");
                continue;
            }

            buffer[data_num] = '\0';
            if(strcmp(buffer, "quit") == 0) {
                break;
	    }
            printf("%s\n", buffer);

            printf("send message:");
            scanf("%s", buffer);
            printf("\n");
            send(client, buffer, strlen(buffer), 0);
            if(strcmp(buffer, "quit") == 0)
            break;
        }
    }

    close(server_socket);
    return 0;
}
