#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include<memory.h>
#include <sys/ioctl.h>	
#include<stdlib.h>
#include <linux/if_ether.h>
#include <linux/if_packet.h> 
#include<arpa/inet.h>
#include<netinet/if_ether.h>
#include<errno.h>
#include<netinet/ether.h>
#include<net/if.h>
#include<string.h>

#define dump printf
#define dump_data(p_data, len) \
({\
    int i =0;\
    for(i = 0; i < len; i++) { \
        if ((0 != i) && (0 == i % 16)) {\
	    dump("\n");\
	}\
        dump("%02x ", p_data[i]&0xFF);\
    }\
    dump("\n");\
})

int main(int argc, char **argv) {
	int sock, n;
	char buffer[1024] = {0};
	unsigned char send_msg[1024] = {
		//--------------组MAC--------14------
		0x00, 0x0c, 0x29, 0x89, 0x26, 0xc4,
	        0x00, 0x0c, 0x29, 0x89, 0x26, 0xce,
		0x08, 0x00,                         //类型：0x0800 IP协议
	};
	if ((sock = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_IP))) < 0)
	{
		perror("socket");
		exit(1);
	}
	struct sockaddr_ll client;
	socklen_t addr_length = sizeof(struct sockaddr_ll);
	struct sockaddr_ll sll;					//原始套接字地址结构
	struct ifreq req;					//网络接口地址	
	strncpy(req.ifr_name, "ens33", IFNAMSIZ);			//指定网卡名称
	if(-1 == ioctl(sock, SIOCGIFINDEX, &req))	//获取网络接口
	{
		perror("ioctl");
		close(sock);
		exit(-1);
	}	

	bzero(&sll, sizeof(sll));
	sll.sll_ifindex = req.ifr_ifindex;
        uint8_t sendbuffer[1024]; 
	int count = 10000;
	while (count--) {
		usleep(1000);
		n = recvfrom(sock, buffer,1024,0, (struct sockaddr *)&client, &addr_length);
		if (n < 14) {
		    continue;
		}

		printf("recv data is :\n");
                dump_data(buffer, n);

		int num = n-14;
		memcpy(sendbuffer, buffer, n);
		char data[1024] = {0};
                memcpy(data,sendbuffer+14,num);	
			
		int len = sprintf(send_msg+14, "%s", "hello, world");
		len = sendto(sock, send_msg, 14+len, 0 , (struct sockaddr *)&sll, sizeof(sll));
		if(len == -1)
		{
		    perror("sendto");
		}	
	}
}
