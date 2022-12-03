#include <stdio.h>
#include <unistd.h>

#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>

#include<string.h>
#include <iostream>

using namespace std;

int main()
{ 
    int sock=-1;
    if((sock=socket(AF_INET,SOCK_DGRAM,0))==-1)
    {
        cout<<"socket error..."<<endl;
        return -1;
    }

#if 0   // bind to special eth
#define ETH "ens33"
    struct ifreq ifr;
    memset(&ifr, 0, sizeof(ifr));
    strncpy(ifr.ifr_name, ETH, strlen(ETH)+1);
    setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, &ifr, sizeof(ifr));
#endif

    int opt_value = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt_value, sizeof(opt_value));

    // must bind for udp
    struct sockaddr_in addrto;
    socklen_t len=sizeof(addrto);
    bzero(&addrto,sizeof(struct sockaddr_in));
    addrto.sin_family=AF_INET;
    addrto.sin_addr.s_addr=htonl(INADDR_ANY);
    addrto.sin_port=htons(6000);

    if(bind(sock,(struct sockaddr*)&(addrto),len)==-1)
    {
        cout<<"bind error..."<<endl;
        return -1;
    } 
    
    char msg[100]={0}; 
    struct in_addr addr;
    while(1)
    {
        bzero(&addrto,sizeof(struct sockaddr_in));
        int ret=recvfrom(sock, msg, 100, 0, (struct sockaddr*)&addrto, &len);
        if(ret<=0)
        {
            cout<<"read error..."<<endl;
        }
        else
        {
            // inet_addr(), inet_aotn(), inet_ntoa()
            addr.s_addr=addrto.sin_addr.s_addr;
            printf("addr %s, port %d : %s\n", inet_ntoa(addr), ntohs(addrto.sin_port), msg);
        }
        sleep(1);
    }

    return 0;
}

