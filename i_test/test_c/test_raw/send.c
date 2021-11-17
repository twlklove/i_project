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
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/if.h>
#include <sys/ioctl.h>

int main()
{  
    int ret = 0;

    char *p_eth = "ens33"; 
    struct ifreq req;
    int sd = socket(PF_INET,SOCK_DGRAM,0);
    strncpy(req.ifr_name, p_eth, sizeof(p_eth)); 
    ret=ioctl(sd, SIOCGIFINDEX, &req);   
    close(sd);
    if (ret==-1)
    {
       printf("Get eth0 index err, %d \n", ret);
    }
   
    printf("name is %s, index is %d\n", req.ifr_name, req.ifr_ifindex); 
    struct sockaddr_ll stTagAddr;
    memset(&stTagAddr, 0 , sizeof(stTagAddr));
    stTagAddr.sll_family    = AF_PACKET;
    stTagAddr.sll_protocol  = htons(ETH_P_ALL);
    stTagAddr.sll_ifindex   = req.ifr_ifindex; //网卡eth0的index, dst eth
    //stTagAddr.sll_pkttype   = PACKET_OUTGOING; //标识包的类型为发出去的包
    stTagAddr.sll_halen     = 6;    //目标MAC地址长度为6
   
    //00:0c:29:89:26:ba 
    //填写目标MAC地址
    stTagAddr.sll_addr[0]   = 0x00;
    stTagAddr.sll_addr[1]   = 0x0c;
    stTagAddr.sll_addr[2]   = 0x29;
    stTagAddr.sll_addr[3]   = 0x89;
    stTagAddr.sll_addr[4]   = 0x26;
    stTagAddr.sll_addr[5]   = 0xba;

    //int SockFd = socket(PF_PACKET, SOCK_RAW, xx);
    int SockFd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL)); 
    //int SockFd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_IP)); 
    if (-1 == SockFd)
    {
      printf("create socket error.\n");
      return 1;
    }    

    char buff[2048] = "hello, world"; 
    int count = 100;
    int len = 0;
    while (count--) {
        //int len = sendto(SockFd, buff, sizeof(buff), 0, NULL, NULL);
	len = sendto(SockFd, buff, strlen(buff)+1, 0, (const struct sockaddr *)&stTagAddr, sizeof(stTagAddr));    
	sleep(1);
	printf("\n");
    }

    close(SockFd);
    return 0;
}


