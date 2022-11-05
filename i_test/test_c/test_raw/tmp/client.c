#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <net/if.h>				
#include <sys/ioctl.h>			
#include <sys/socket.h>
#include <netinet/ether.h>		
#include <netpacket/packet.h>	
#include <arpa/inet.h>
#include <sys/types.h>
#include<unistd.h>    //close
int main(int argc, char *argv[])
{
	//1.创建通信用的原始套接字
	int n;
	char buffer[1024];
	int sock_raw_fd = socket(PF_PACKET, SOCK_RAW, htons(ETH_P_IP));
	uint8_t sendbuffer[1024]; 	
	//2.根据各种协议首部格式构建发送数据报
	unsigned char send_msg[1024] = {
		//--------------组MAC--------14------
		0x00, 0x0c, 0x29, 0x89, 0x26, 0xce,
		0x00, 0x0c, 0x29, 0x89, 0x26, 0xc4, 
		0x08, 0x00,                         //类型：0x0800 IP协议
	};	

	struct ifreq req;					//网络接口地址	
	strncpy(req.ifr_name, "ens33", IFNAMSIZ);			//指定网卡名称
	if(-1 == ioctl(sock_raw_fd, SIOCGIFINDEX, &req))	//获取网络接口
	{
		perror("ioctl");
		close(sock_raw_fd);
		exit(-1);
	}
	struct sockaddr_ll sll;					//原始套接字地址结构
			
	/*将网络接口赋值给原始套接字地址结构*/
	bzero(&sll, sizeof(sll));
	sll.sll_ifindex = req.ifr_ifindex;


	int count=10000;
	
	while(count--)
	{
		usleep(1000);
		int len = sprintf(send_msg+14, "%s", "test");
		len = sendto(sock_raw_fd, send_msg, 14+len, 0 , (struct sockaddr *)&sll, sizeof(sll));
		if(len == -1)
		{
			perror("sendto");
		}	

		socklen_t addr_length = sizeof(struct sockaddr_ll);
		n = recvfrom(sock_raw_fd, buffer,1024,0, (struct sockaddr *)&sll, &addr_length);
		if ((n < 14) || (buffer[14] != 'h')) {
			continue;
		}

		int num = n-14;
		memcpy(sendbuffer, buffer, n);
		char data[1024]={0};
                memcpy(data,sendbuffer+14,num);
		printf("recv data is : %s\n",data);
		sleep(3);
	}
	return 0;
}
