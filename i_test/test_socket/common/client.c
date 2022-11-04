#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/shm.h>

#define MYPORT  8887
#define BUFFER_SIZE 1024

int main(int argc, char *argv[])
{
    char *ip = "127.0.0.1";
    if (argc > 1) {
        ip = argv[1];
    }
#if 1
    struct sockaddr_in ip_addr = {0};
    memset(&ip_addr, 0, sizeof(ip_addr));
    int  len = sizeof(ip_addr);
    char *buf = (char*)(&ip_addr.sin_addr);

    ip_addr.sin_family = AF_INET;
    ip_addr.sin_port = htons(MYPORT);

    if (1 != inet_pton(AF_INET, ip, buf)) {
        printf("ip format is not right\n");
        return -1;
    }
    
    int fd = socket(ip_addr.sin_family, SOCK_STREAM, 0);
    if (-1 == fd) {
        perror("socket");
        exit(1);
    }

    struct sockaddr_in local_addr = {0};
    memset(&local_addr, 0, sizeof(local_addr));
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(0);
    local_addr.sin_port = htonl(INADDR_ANY);
    if (bind(fd, (struct sockaddr *)&local_addr, len) < 0)
    {
        perror("bind");
        exit(1);
    }

    
    if (connect(fd, (struct sockaddr *)&ip_addr, len) < 0)
    {
        perror("connect");
        exit(1);
    }
#endif
#if 0
    struct sockaddr_in6 ip_addr = {0};
    memset(&ip_addr, 0, sizeof(ip_addr));
    int  len = sizeof(ip_addr);
    char *buf = (char*)(&ip_addr.sin6_addr);

    ip_addr.sin6_family = AF_INET;
    ip_addr.sin6_port = htons(MYPORT);

    if (1 != inet_pton(AF_INET, ip, buf)) {
        if (1 != inet_pton(AF_INET6, ip, buf)) {
            printf("ip format is not right\n");
            return -1;
        }
        ip_addr.sin6_family = AF_INET6;
    }
    
    int fd = socket(ip_addr.sin6_family, SOCK_STREAM, 0);
    if (-1 == fd) {
        perror("socket");
        exit(1);
    }

    if (connect(fd, (struct sockaddr *)&ip_addr, len) < 0)
    {
        perror("connect");
        exit(1);
    }
#endif
    char sendbuf[BUFFER_SIZE];
    char recvbuf[BUFFER_SIZE];
    while (fgets(sendbuf, sizeof(sendbuf), stdin) != NULL)
    {
        send(fd, sendbuf, strlen(sendbuf)+1,0);
        if(strcmp(sendbuf,"exit\n")==0)
            break;
        recv(fd, recvbuf, sizeof(recvbuf)+1 ,0);
        fputs(recvbuf, stdout);

        memset(sendbuf, 0, sizeof(sendbuf));
        memset(recvbuf, 0, sizeof(recvbuf));
    }

    close(fd);
    return 0;
}
