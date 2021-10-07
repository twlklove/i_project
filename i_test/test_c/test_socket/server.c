#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <sys/epoll.h>

#define MYPORT  8887
#define QUEUE   20
#define BUFFER_SIZE 1024

void open_server(int argc, char *argv[])
{
    char *ip = "127.0.0.1";
    if (argc > 1) {
        ip = argv[1];
    }

    struct sockaddr_in6 ip_addr = {0};
    int  len = sizeof(ip_addr);
    char *buf = (char*)(&ip_addr.sin6_addr);

    ip_addr.sin6_family = AF_INET;
    ip_addr.sin6_port = htons(MYPORT);

    if (1 != inet_pton(AF_INET, ip, buf)) {
        if (1 != inet_pton(AF_INET6, ip, buf)) {
            printf("ip format is not right\n");
            return;
        }
        ip_addr.sin6_family = AF_INET6;
    }
   
    int epfd = epoll_create(QUEUE);
    if (-1 == epfd) {
         printf("epool fail \n");
         return;
    }

    int fd = socket(AF_INET6, SOCK_STREAM, 0);
    if (-1 == fd) {
        perror("socket");
        close(epfd);
        exit(1);
    }

    struct epoll_event ev, events[QUEUE];
    ev.data.fd = fd;
    ev.events = EPOLLIN | EPOLLET;
    int ret = epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);
    if (0 != ret) {
        printf("fail to epoll ctl\n");
        close(fd);
        close(epfd);
        return;
    }

    char ip_str[INET6_ADDRSTRLEN] = {0};
    printf("%s\n", inet_ntop(ip_addr.sin6_family, buf, ip_str, INET6_ADDRSTRLEN));

    if(-1 == bind(fd, (struct sockaddr *)(&ip_addr), len))
    {
        perror("bind");
        close(fd);
        close(epfd);
        exit(1);
    }

    if(-1 == listen(fd, QUEUE))
    {
        perror("listen");
        close(fd);
        close(epfd);
        exit(1);
    }

    char buffer[BUFFER_SIZE];
    struct sockaddr_in6 client_addr;
    socklen_t length = sizeof(client_addr);
 
    while (1) {
        int nfds = epoll_wait(epfd, events, QUEUE, 200);  // 200ms
        int i = 0;
        for (i = 0; i < nfds; i++) {
            if(fd == events[i].data.fd) {
               int fd_con = accept(fd, (struct sockaddr*)&client_addr, &length);
               if(fd_con < 0) {
                   perror("connect");
                   continue;                   
               } 

               char client_str[INET6_ADDRSTRLEN] = {0};
               printf("%s\n", inet_ntop(client_addr.sin6_family, &(client_addr.sin6_addr), 
                                                            client_str, INET6_ADDRSTRLEN));
               ev.data.fd = fd_con;
               ev.events = EPOLLIN | EPOLLET;
               if (0 != epoll_ctl(epfd, EPOLL_CTL_ADD, fd_con, &ev)) {
                   printf("epoll ctl fail\n");
                   close(fd_con);
               }
            }
                 // already has connected and has received data
            else if ((events[i].events & EPOLLIN) ||(events[i].events & EPOLLPRI)) { 
                int fd_tmp = events[i].data.fd;
                if (fd_tmp < 0) {
                    printf("recev fd_tmp < 0\n");
                    continue;
                }
                 
                memset(buffer,0,sizeof(buffer));
                int recv_len = recv(fd_tmp, buffer, sizeof(buffer),0);
                if(strcmp(buffer,"exit\n")==0) {
                    close(fd_tmp);
                    printf("exit\n");
                    continue;
                }
                fputs(buffer, stdout);

                ev.data.fd = fd_tmp;
                //ev.data.ptr = buf;
                ev.events = EPOLLOUT | EPOLLET;
                if (0 != epoll_ctl(epfd, EPOLL_CTL_MOD, fd_tmp, &ev)) {
                    printf("in err: epoll ctl\n");
                    close(fd_tmp);
                }
                printf("hello\n");
            }
            else if (events[i].events & EPOLLOUT) {  //  has data to send
                int fd_tmp = events[i].data.fd;
                if (fd_tmp < 0) {
                    printf("fd is < 0\n");
                    continue;
                }
printf("hello123\n");                
                char *p_send = "hello, world\n";
                send(fd_tmp, p_send, strlen(p_send), 0);
                ev.data.fd = fd_tmp;
                ev.events = EPOLLIN | EPOLLET;
                if (0 != epoll_ctl(epfd, EPOLL_CTL_MOD, fd_tmp, &ev)) {
                    printf("out err: epoll ctl\n");
                    close(fd_tmp);
                }
            }
            else {
                printf("err: %d\n", events[i].events);
                close(events[i].data.fd);
            }
        }
    }
    
    close(fd);
    close(epfd); 
}

int main(int argc, char *argv[])
{
    open_server(argc, argv);
    return 0;
}
