#include<netdb.h>
#include<netinet/in.h>
#include<arpa/inet.h>
#include<string.h>
#include <iostream>
#include <unistd.h>
using namespace std;

int main()
{
    int ret = 0;
    int sock=-1;
    if((sock=socket(AF_INET,SOCK_DGRAM,0))==-1)
    {
        cout<<"sock error"<<endl;
        return -1;
    }

    const int opt=-1;
    int nb=0;
    nb=setsockopt(sock,SOL_SOCKET,SO_BROADCAST,(char*)&opt,sizeof(opt));//设置套接字类型
    if(nb==-1)
    {
        cout<<"set socket error...\n"<<endl;
        return -1;
    }

    struct sockaddr_in addrto;
    bzero(&addrto,sizeof(struct sockaddr_in));
    addrto.sin_family=AF_INET;
    addrto.sin_addr.s_addr=htonl(INADDR_BROADCAST);//套接字地址为广播地址
    addrto.sin_port=htons(6000);//套接字广播端口号为6000
    int nlen=sizeof(addrto);
    char msg[]={"broadcast the message ==> hello world"};
    
    while(1)
    {
        ret=sendto(sock, msg, strlen(msg), 0, (sockaddr*)&addrto,nlen); //向广播地址发布消息
        if(ret<0)
        {
            cout<<"send error...\n"<<endl;
            break;
        }
        else 
        {
            printf("ok\n");
        }

        sleep(1);
    }

    return ret;
}
